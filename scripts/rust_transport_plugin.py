"""pytest plugin: run the official Spark Connect tests through our Rust transport.

The official test files import the *reference* pyspark (they live in its tree), which
builds the protobuf plans. This plugin monkeypatches ``SparkConnectClient`` so its gRPC
stub is our Rust-backed transport instead of grpcio: the reference builds byte-identical
plan protos, and our Rust ``spark-connect-core`` carries them over the wire and decodes
the Arrow results. That makes the official suite genuinely exercise our Rust client with
no vendoring.

Usage (see scripts/run_rust_official_tests.sh):
    RUST_PYSPARK_SO=/path/to/python/pyspark/_pyspark.so \
    SPARK_CONNECT_TESTING_REMOTE=sc://localhost:15002 SPARK_TESTING=1 \
    SPARK_SKIP_CONNECT_COMPAT_TESTS=1 \
    PYTHONPATH=/repo/scripts:/path/to/spark-v4.2.0/python \
    python -m pytest -p rust_transport_plugin <official test file>
"""

import importlib.util
import os

# Load our compiled extension directly by file so it works even though the reference
# pyspark (which has no _pyspark) is the one on the path.
_SO = os.environ.get("RUST_PYSPARK_SO")
if not _SO or not os.path.exists(_SO):
    raise RuntimeError("RUST_PYSPARK_SO must point to the built _pyspark extension (.so/.dylib)")
# The extension's init symbol is PyInit__pyspark, so it must be loaded as "_pyspark".
_spec = importlib.util.spec_from_file_location("_pyspark", _SO)
_rust = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rust)


_GRPC_DETAILS_KEY = "grpc-status-details-bin"


def _rust_rpc_to_exception(e):
    """Turn a Rust RustRpcError into a real grpc.RpcError.

    The Rust side hands us (grpc_code:int, status_details:bytes, message:str) where
    status_details is the serialized google.rpc.Status (the grpc-status-details-bin
    trailer). We wrap it as a grpc.RpcError so the reference client's own
    _handle_rpc_error runs natively: it reads rpc_status.from_call(error), parses the
    ErrorInfo, enriches via FetchErrorDetails, and calls convert_exception - giving the
    exact same typed exception AND full message the reference produces.
    """
    import grpc

    code, details, message = e.args

    class _RustRpcError(grpc.RpcError):
        def code(self):
            for c in grpc.StatusCode:
                if c.value[0] == code:
                    return c
            return grpc.StatusCode.UNKNOWN

        def details(self):
            # grpc_status.rpc_status.from_call requires call.details() to match the
            # message inside the status-details proto exactly, so read it from there.
            if details:
                try:
                    from google.rpc import status_pb2

                    return status_pb2.Status.FromString(bytes(details)).message
                except Exception:
                    pass
            return message

        def trailing_metadata(self):
            return ((_GRPC_DETAILS_KEY, bytes(details)),) if details else ()

        def initial_metadata(self):
            return ()

    return _RustRpcError()


class _RustStub:
    """A grpcio-compatible SparkConnectServiceStub backed by our Rust transport."""

    def __init__(self, url):
        self._c = _rust.RustConnectStub(url)

    def _pb(self):
        # Imported lazily; the reference pyspark provides these pb2 types.
        from pyspark.sql.connect.proto import base_pb2

        return base_pb2

    def _stream(self, rust_stream):
        # Lazily yield pb2 ExecutePlanResponse from a Rust ResponseStream, converting a
        # per-message gRPC error into a real grpc.RpcError so the reference client's
        # reattachable iterator/error handling behaves exactly as upstream.
        pb = self._pb()
        it = iter(rust_stream)
        while True:
            try:
                b = next(it)
            except StopIteration:
                return
            except _rust.RustRpcError as e:
                raise _rust_rpc_to_exception(e) from None
            yield pb.ExecutePlanResponse.FromString(b)

    def ExecutePlan(self, request, metadata=None, **kw):
        try:
            rust_stream = self._c.execute_plan(request.SerializeToString())
        except _rust.RustRpcError as e:
            raise _rust_rpc_to_exception(e) from None
        return self._stream(rust_stream)

    def AnalyzePlan(self, request, metadata=None, **kw):
        pb = self._pb()
        try:
            raw = self._c.analyze_plan(request.SerializeToString())
        except _rust.RustRpcError as e:
            raise _rust_rpc_to_exception(e) from None
        return pb.AnalyzePlanResponse.FromString(raw)

    def Config(self, request, metadata=None, **kw):
        pb = self._pb()
        try:
            raw = self._c.config(request.SerializeToString())
        except _rust.RustRpcError as e:
            raise _rust_rpc_to_exception(e) from None
        return pb.ConfigResponse.FromString(raw)

    def Interrupt(self, request, metadata=None, **kw):
        pb = self._pb()
        try:
            raw = self._c.interrupt(request.SerializeToString())
        except _rust.RustRpcError as e:
            raise _rust_rpc_to_exception(e) from None
        return pb.InterruptResponse.FromString(raw)

    def FetchErrorDetails(self, request, metadata=None, **kw):
        pb = self._pb()
        # Must surface a real grpc.RpcError on failure, exactly like the grpcio stub:
        # the reference client enriches errors via _fetch_enriched_error(), which wraps
        # this call in `except grpc.RpcError: return None` and falls back to the original
        # execute error. If we let a raw RustRpcError escape instead, that fallback never
        # runs and the enrichment failure masks the real error (e.g. a worker "Segmentation
        # fault" gets replaced by the server-side NPE that FetchErrorDetails itself raises).
        try:
            raw = self._c.fetch_error_details(request.SerializeToString())
        except _rust.RustRpcError as e:
            raise _rust_rpc_to_exception(e) from None
        return pb.FetchErrorDetailsResponse.FromString(raw)

    def ReattachExecute(self, request, metadata=None, **kw):
        try:
            rust_stream = self._c.reattach_execute(request.SerializeToString())
        except _rust.RustRpcError as e:
            raise _rust_rpc_to_exception(e) from None
        return self._stream(rust_stream)

    def ReleaseExecute(self, request, metadata=None, **kw):
        pb = self._pb()
        return pb.ReleaseExecuteResponse.FromString(
            self._c.release_execute(request.SerializeToString())
        )

    def ReleaseSession(self, request, metadata=None, **kw):
        # Best-effort: the server reclaims idle sessions; a no-op keeps close() working.
        return self._pb().ReleaseSessionResponse()

    def close(self):
        # Cancel in-flight Rust streams (mirrors grpcio channel.close()) so a reader
        # blocked on a long-lived stream (e.g. the streaming-query listener bus) unblocks.
        self._c.close()


def pytest_configure(config):
    import pyspark.sql.connect.client.core as core

    url = os.environ.get("SPARK_CONNECT_TESTING_REMOTE", "sc://localhost:15002")
    orig_init = core.SparkConnectClient.__init__

    def patched_init(self, connection, *args, **kwargs):
        orig_init(self, connection, *args, **kwargs)
        try:
            endpoint = self._builder.endpoint
            client_url = f"sc://{endpoint}"
        except Exception:
            client_url = url
        # Route transport through Rust. Reattachable execute stays enabled (upstream
        # default) so operations are registered/interruptible and streaming works like
        # upstream; ReattachExecute/ReleaseExecute are implemented on the Rust stub.
        self._stub = _RustStub(client_url)

    core.SparkConnectClient.__init__ = patched_init

    # The reference client's close() calls self._channel.close(), which for grpcio cancels
    # in-flight RPCs. Our transport bypasses that channel, so also cancel the Rust stub's
    # streams here - otherwise a non-daemon streaming-query listener thread blocked reading
    # a long-lived stream never terminates and hangs interpreter shutdown (threading._shutdown).
    orig_close = core.SparkConnectClient.close

    def patched_close(self):
        try:
            stub = getattr(self, "_internal_stub", None)
            if stub is not None and hasattr(stub, "close"):
                stub.close()
        except Exception:
            pass
        return orig_close(self)

    core.SparkConnectClient.close = patched_close
