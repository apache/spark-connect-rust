# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Launch the native server inside Spark and run reproducible lifecycle/API acceptance."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time


SHUTDOWN_COMPLETE = "Embedded Spark Engine shutdown complete"


def shutdown_error(content, exit_code, marker=SHUTDOWN_COMPLETE):
    if re.search(r"shutdown failed|JNI stop is incomplete|native stop incomplete", content,
                 flags=re.IGNORECASE):
        return "embedded server reported failed or incomplete shutdown"
    if re.search(r"JNI cleanup is incomplete|Java engine handle cleanup failed", content,
                 flags=re.IGNORECASE):
        return "embedded server reported failed resource cleanup"
    if marker not in content.splitlines():
        return "embedded server omitted successful cleanup marker"
    if exit_code not in (0, -signal.SIGTERM, 128 + signal.SIGTERM):
        return "embedded server exited abnormally"
    return None


def listener_owner_pids(port, *, proc_root=Path("/proc")):
    """Inspect listener ownership with lsof, falling back to Linux /proc."""
    lsof = shutil.which("lsof")
    if lsof is not None:
        ownership = subprocess.run(
            [lsof, "-nP", "-iTCP:" + str(port), "-sTCP:LISTEN", "-Fp"],
            capture_output=True, text=True, check=True, timeout=10)
        return sorted({int(line[1:]) for line in ownership.stdout.splitlines()
                       if line.startswith("p")})
    if not (proc_root / "net" / "tcp").is_file():
        raise RuntimeError("listener ownership check requires lsof or Linux /proc")
    inodes = set()
    for table in (proc_root / "net" / "tcp", proc_root / "net" / "tcp6"):
        if not table.is_file():
            continue
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add("socket:[" + fields[9] + "]")
    owners = set()
    for process in proc_root.iterdir():
        if not process.name.isdecimal():
            continue
        try:
            for descriptor in (process / "fd").iterdir():
                try:
                    if os.readlink(descriptor) in inodes:
                        owners.add(int(process.name))
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return sorted(owners)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path, help="built JNI .dylib or .so")
    parser.add_argument("--native-jar", type=Path,
                        help="also exercise classpath loading from this native platform/all JAR")
    parser.add_argument("--spark-home", type=Path, default=os.environ.get("SPARK_HOME"))
    parser.add_argument("--python", default=sys.executable, help="official PySpark client/worker Python")
    parser.add_argument("--rust-example", type=Path, help="optional prebuilt official Rust client example")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.spark_home is None:
        parser.error("set SPARK_HOME or pass --spark-home")
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE"):
        parser.error("acceptance checks require Python assertions; unset PYTHONOPTIMIZE")
    root = Path(__file__).resolve().parents[1]
    library = args.library.resolve(strict=True)
    native_jar = args.native_jar.resolve(strict=True) if args.native_jar else None
    spark_home = args.spark_home.resolve(strict=True)
    reports = root / ".build" / "jvm-native"
    reports.mkdir(parents=True, exist_ok=True)
    # Keep every run's logs/client report separate so stale evidence cannot pass a rerun.
    build = Path(tempfile.mkdtemp(prefix="engine-", dir=reports))
    output = args.output or reports / "engine-acceptance.json"
    spark_submit = spark_home / "bin" / "spark-submit"
    if not spark_submit.is_file():
        parser.error("Spark launcher does not exist: " + str(spark_submit))
    production_jar = root / ".build" / "jvm" / "spark-connect-native-bridge.jar"
    tests_jar = root / ".build" / "jvm" / "bridge-test.jar"
    for artifact in (production_jar, tests_jar):
        if not artifact.is_file():
            parser.error("missing Java artifact; first run jvm/build.sh --tests: " + str(artifact))
    env = os.environ.copy()
    env.update({"SPARK_HOME": str(spark_home), "SPARK_LOCAL_IP": "127.0.0.1",
                "SPARK_CONNECT_MODE": "0", "PYSPARK_PYTHON": args.python, "TZ": "UTC"})
    env.pop("SPARK_REMOTE", None)
    env.pop("PYSPARK_REMOTE_INIT_CONF_LEN", None)
    common = [str(spark_submit), "--master", "local[2]",
              "--driver-memory", "1g", "--driver-java-options", "-Xcheck:jni",
              "--conf", "spark.ui.enabled=false", "--conf", "spark.driver.host=127.0.0.1",
              "--conf", "spark.driver.bindAddress=127.0.0.1", "--conf", "spark.sql.shuffle.partitions=2",
              "--conf", "spark.pyspark.python=" + args.python,
              "--conf", "spark.sql.warehouse.dir=" + str(build / "warehouse")]
    marker = SHUTDOWN_COMPLETE
    report = {"backend": "spark-jvm", "library": str(library), "spark_home": str(spark_home),
              "native_jar": str(native_jar) if native_jar else None,
              "run_directory": str(build), "results": []}
    server = None
    log = None
    address = None

    def run(name, command, timeout=180):
        started = time.monotonic()
        path = build / (name + ".log")
        try:
            completed = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                       text=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            def decoded(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            path.write_text(decoded(error.stdout) + decoded(error.stderr))
            report["results"].append({"name": name, "command": command, "timed_out": True,
                                      "seconds": round(time.monotonic() - started, 3),
                                      "log": str(path)})
            raise RuntimeError(name + " timed out; see " + str(path)) from error
        path.write_text(completed.stdout + completed.stderr)
        result = {"name": name, "command": command, "exit_code": completed.returncode,
                  "seconds": round(time.monotonic() - started, 3), "log": str(path)}
        report["results"].append(result)
        print(json.dumps(result), flush=True)
        if completed.returncode != 0:
            raise RuntimeError(name + " failed; see " + str(path))
        if re.search(r"WARNING in native method|FATAL ERROR in native method", path.read_text()):
            raise RuntimeError(name + " triggered a JNI check warning")
        return completed

    try:
        for name, class_name, success in (
            ("engine-smoke", "EngineSmokeTest", "ENGINE_SMOKE_OK"),
            ("vector-smoke", "VectorBatchSmokeTest", "VECTOR_BATCH_SMOKE_OK"),
        ):
            completed = run(name, common + ["--class",
                "org.apache.spark.sql.connect.nativebridge.engine." + class_name,
                str(tests_jar)])
            if success not in completed.stdout:
                raise RuntimeError(name + " success marker missing")
        engine_lifecycle = run("engine-lifecycle", common + ["--class",
            "org.apache.spark.sql.connect.nativebridge.engine.NativeEngineLifecycleTest",
            str(tests_jar), str(library)])
        if "NATIVE_ENGINE_LIFECYCLE_OK" not in engine_lifecycle.stdout:
            raise RuntimeError("native engine lifecycle success marker missing")
        cleanup_failure = run("cleanup-failure", common + ["--class",
            "org.apache.spark.sql.connect.nativebridge.engine.CleanupFailureTest",
            str(tests_jar), str(library)])
        if "CLEANUP_FAILURE_OK" not in cleanup_failure.stdout:
            raise RuntimeError("cleanup failure retention success marker missing")
        # A private JVM temp root lets the check prove extracted library cleanup at exit.
        native_temp = build / "native-temp"
        native_temp.mkdir()
        packaged_options = (["--jars", str(native_jar), "--conf",
                             "spark.driver.extraJavaOptions=-Xcheck:jni -Djava.io.tmpdir="
                             + shlex.quote(str(native_temp))] if native_jar else [])
        command = common + packaged_options + ["--class",
                            "org.apache.spark.sql.connect.nativebridge.engine.EmbeddedSparkEngine",
                            str(production_jar),
                            "--bundled" if native_jar else str(library), "127.0.0.1:0"]
        log_path = build / "server.log"
        log = log_path.open("w")
        server = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        report.update({"server_command": command, "server_pid": server.pid, "server_log": str(log_path)})
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            content = log_path.read_text(errors="replace")
            listening = re.search(r"Embedded Spark Connect listening on (\S+)", content)
            identity = re.search(r"Embedded Spark Connect JVM PID (\d+); Java Connect listener absent", content)
            if listening and identity:
                address = listening.group(1)
                report["reported_jvm_pid"] = int(identity.group(1))
                if int(identity.group(1)) != server.pid:
                    raise RuntimeError("Spark launcher and JVM PID differ")
                break
            if server.poll() is not None:
                raise RuntimeError("embedded server exited during startup; see " + str(log_path))
            time.sleep(0.1)
        else:
            raise RuntimeError("embedded server startup timed out; see " + str(log_path))
        owners = listener_owner_pids(int(address.rsplit(":", 1)[1]))
        report["listener_owner_pids"] = owners
        if owners != [server.pid]:
            raise RuntimeError("native listener is not owned exclusively by the Spark JVM")
        if native_jar:
            extracted = list(native_temp.rglob("libspark_connect_jni.*"))
            if len(extracted) != 1:
                raise RuntimeError("expected one extracted native library while server is running")
            if hashlib.sha256(extracted[0].read_bytes()).digest() != hashlib.sha256(library.read_bytes()).digest():
                raise RuntimeError("packaged library differs from selected native artifact")
            report["extracted_native_library"] = str(extracted[0])
        env["SPARK_REMOTE"] = "sc://" + address
        report["remote"] = env["SPARK_REMOTE"]
        client_report = build / "engine-client.json"
        run("engine-client", [args.python, "-B", "jvm/engine_acceptance.py", "--remote",
                               env["SPARK_REMOTE"], "--output", str(client_report)], timeout=600)
        evidence = json.loads(client_report.read_text())
        if (evidence.get("passed") is not True or not evidence.get("checks")
                or any(item.get("passed") is not True for item in evidence["checks"])):
            raise RuntimeError("direct engine client report did not pass")
        report["engine_client_report"] = str(client_report)
        if args.rust_example:
            run("official-rust", [str(args.rust_example.resolve(strict=True))])
    except Exception as error:
        report["error"] = repr(error)
    except KeyboardInterrupt:
        report["error"] = "acceptance interrupted"
    finally:
        if server is not None:
            if server.poll() is not None and "error" not in report:
                report["error"] = "embedded server exited before requested shutdown"
            if server.poll() is None:
                server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
                report["forced_kill"] = True
            report["server_exit_code"] = server.returncode
        if log is not None:
            log.close()
        if report.get("extracted_native_library"):
            extracted = Path(report["extracted_native_library"])
            report["extracted_library_cleaned"] = not extracted.exists() and not extracted.parent.exists()
            if not report["extracted_library_cleaned"]:
                report.setdefault("error", "extracted native library or private directory leaked at JVM exit")
        if address is not None:
            try:
                with socket.create_connection(("127.0.0.1", int(address.rsplit(":", 1)[1])), timeout=1):
                    report["listener_leaked"] = True
            except OSError:
                report["listener_closed"] = True
        if "server_log" in report:
            content = Path(report["server_log"]).read_text(errors="replace")
            report["jni_check_clean"] = not bool(re.search(
                r"WARNING in native method|FATAL ERROR in native method", content))
            if not report["jni_check_clean"]:
                report["error"] = "embedded server triggered a JNI check warning"
            report["shutdown_complete"] = marker in content.splitlines()
            failure = shutdown_error(content, report["server_exit_code"], marker)
            if failure is not None:
                report.setdefault("error", failure)
        report["passed"] = "error" not in report and not report.get("forced_kill") and not report.get("listener_leaked")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"output": str(output), "passed": report["passed"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
