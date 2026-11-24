// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

package org.apache.spark.sql.connect.nativebridge.engine;

import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.Properties;

/** JDK-only resource, integrity and actual native-loading checks. */
public final class NativeLibraryLoaderTest {
  private static final String ADAPTER_PROPERTIES =
      "META-INF/spark-connect-native-adapter.properties";
  private static int checks;

  private NativeLibraryLoaderTest() {}

  public static void main(String[] arguments) throws Exception {
    if (arguments.length == 1 && arguments[0].equals("--load-packaged")) {
      Path library = NativeLibraryLoader.loadBundled();
      require(Files.isRegularFile(library), "packaged native library was not extracted");
      require(NativeLibraryLoader.loadBundled().equals(library),
          "packaged load was not idempotent");
      System.out.println("NATIVE_PACKAGED_LOAD_OK");
      return;
    }
    require(arguments.length <= 1, "usage: NativeLibraryLoaderTest [absolute-fixture-library]");
    platforms();
    resources();
    if (arguments.length == 1) nativeFixture(Path.of(arguments[0]));
    System.out.println("NATIVE_LIBRARY_LOADER_OK checks=" + checks);
  }

  private static void platforms() throws Exception {
    for (String os : new String[] {"Linux", "LINUX"}) {
      for (String arch : new String[] {"amd64", "x86_64"}) {
        require(NativeLibraryLoader.platform(os, arch).equals("linux-x86_64"), "Linux alias");
      }
      for (String arch : new String[] {"arm64", "aarch64"}) {
        require(NativeLibraryLoader.platform(os, arch).equals("linux-aarch64"), "ARM alias");
      }
    }
    for (String os : new String[] {"Mac OS X", "macOS", "Darwin"}) {
      require(NativeLibraryLoader.platform(os, "arm64").equals("macos-aarch64"), "macOS alias");
      require(NativeLibraryLoader.platform(os, "amd64").equals("macos-x86_64"), "macOS x86 alias");
    }
    expect(UnsupportedOperationException.class, "operating system",
        () -> NativeLibraryLoader.platform("Windows 11", "amd64"));
    expect(UnsupportedOperationException.class, "architecture",
        () -> NativeLibraryLoader.platform("Linux", "riscv64"));
    expect(IllegalArgumentException.class, "absolute",
        () -> NativeLibraryLoader.loadAbsolute(Path.of("relative.so")));
  }

  private static void resources() throws Exception {
    byte[] bytes = new byte[] {1, 4, 9, 16, 25};
    try (Fixture fixture = new Fixture(bytes)) {
      Path extracted = fixture.extract();
      try {
        require(Arrays.equals(bytes, Files.readAllBytes(extracted)), "extracted bytes changed");
        require(Files.getPosixFilePermissions(extracted.getParent()).equals(
            PosixFilePermissions.fromString("rwx------")), "extraction directory is not private");
        require(Files.getPosixFilePermissions(extracted).equals(
            PosixFilePermissions.fromString("rw-------")), "extracted library is not private");
      } finally {
        Files.delete(extracted);
        Files.delete(extracted.getParent());
      }
    }
    try (Fixture fixture = new Fixture(bytes)) {
      Files.delete(fixture.root.resolve(ADAPTER_PROPERTIES));
      expect(IllegalStateException.class, "missing classpath resource", fixture::extract);
    }
    try (Fixture fixture = new Fixture(bytes)) {
      Files.delete(fixture.root.resolve(fixture.prefix + "native.properties"));
      expect(IllegalStateException.class, "matching platform native JAR", fixture::extract);
    }
    try (Fixture fixture = new Fixture(bytes)) {
      Files.delete(fixture.root.resolve(fixture.prefix + fixture.libraryName));
      expect(IllegalStateException.class, "missing classpath resource", fixture::extract);
    }
    try (Fixture fixture = new Fixture(bytes)) {
      fixture.nativeMetadata.setProperty("artifact.version", "other-version");
      fixture.writeNativeMetadata();
      expect(IllegalStateException.class, "artifact version mismatch", fixture::extract);
    }
    try (Fixture fixture = new Fixture(bytes)) {
      fixture.adapterMetadata.setProperty("native.abi.version", "2");
      fixture.writeAdapterMetadata();
      expect(IllegalStateException.class, "adapter native ABI mismatch", fixture::extract);
    }
    try (Fixture fixture = new Fixture(bytes)) {
      fixture.nativeMetadata.setProperty("native.abi.version", "2");
      fixture.writeNativeMetadata();
      expect(IllegalStateException.class, "native ABI mismatch", fixture::extract);
    }
    try (Fixture fixture = new Fixture(bytes)) {
      fixture.nativeMetadata.setProperty("library.name", "../wrong.so");
      fixture.writeNativeMetadata();
      expect(IllegalStateException.class, "library name mismatch", fixture::extract);
    }
    try (Fixture fixture = new Fixture(bytes)) {
      fixture.nativeMetadata.setProperty("library.sha256", "bad-digest");
      fixture.writeNativeMetadata();
      expect(IllegalStateException.class, "64 lowercase hexadecimal", fixture::extract);
    }
    try (Fixture fixture = new Fixture(bytes)) {
      Files.write(fixture.root.resolve(fixture.prefix + fixture.libraryName), new byte[] {0});
      expect(IllegalStateException.class, "SHA-256 mismatch", fixture::extract);
    }
    try (Fixture fixture = new Fixture(new byte[0])) {
      expect(IllegalStateException.class, "resource is empty", fixture::extract);
    }
    try (Fixture fixture = new Fixture(bytes)) {
      fixture.nativeMetadata.remove("native.abi.version");
      fixture.writeNativeMetadata();
      expect(IllegalStateException.class, "missing native artifact property", fixture::extract);
    }
    try (Fixture fixture = new Fixture(bytes)) {
      Files.writeString(fixture.root.resolve(ADAPTER_PROPERTIES), "x".repeat(16 * 1024 + 1));
      expect(IllegalStateException.class, "exceeds 16 KiB", fixture::extract);
    }
    for (String duplicate : new String[] {ADAPTER_PROPERTIES, "native.properties", "library"}) {
      try (Fixture fixture = new Fixture(bytes)) {
        Path second = Files.createTempDirectory("native-duplicate-test-");
        try {
          String name = duplicate.equals(ADAPTER_PROPERTIES) ? duplicate : fixture.prefix
              + (duplicate.equals("library") ? fixture.libraryName : duplicate);
          Path destination = second.resolve(name);
          Files.createDirectories(destination.getParent());
          Files.copy(fixture.root.resolve(name), destination);
          try (URLClassLoader loader = new URLClassLoader(new URL[] {
              fixture.root.toUri().toURL(), second.toUri().toURL()}, null)) {
            expect(IllegalStateException.class, "multiple classpath resources",
                () -> NativeLibraryLoader.extract(loader, fixture.os, fixture.arch));
          }
        } finally {
          removeTree(second);
        }
      }
    }
  }

  private static void nativeFixture(Path library) throws Exception {
    require(library.isAbsolute() && Files.isRegularFile(library),
        "fixture library must be an absolute file");
    try (Fixture fixture = new Fixture(Files.readAllBytes(library));
         URLClassLoader loader = fixture.loader()) {
      Path loaded = NativeLibraryLoader.loadBundled(loader, fixture.os, fixture.arch);
      require(fixtureValue() == 37, "fixture JNI symbol did not resolve after packaged loading");
      require(NativeLibraryLoader.loadBundled(loader, fixture.os, fixture.arch).equals(loaded),
          "repeat packaged load extracted or loaded another copy");
      require(NativeLibraryLoader.loadAbsolute(loaded).equals(loaded),
          "same absolute library was not reused");
      expect(IllegalStateException.class, "different Spark Connect native library",
          () -> NativeLibraryLoader.loadAbsolute(library));
    }
  }

  private static native int fixtureValue();

  private static void require(boolean condition, String message) {
    if (!condition) throw new AssertionError(message);
    checks++;
  }

  @FunctionalInterface
  private interface CheckedRunnable {
    void run() throws Exception;
  }

  private static void expect(Class<? extends Throwable> type, String message,
      CheckedRunnable action) throws Exception {
    try {
      action.run();
    } catch (Throwable failure) {
      require(type.isInstance(failure), "unexpected exception " + failure);
      require(failure.getMessage() != null && failure.getMessage().contains(message),
          "missing diagnostic '" + message + "' in " + failure);
      return;
    }
    throw new AssertionError("expected " + type.getSimpleName() + " containing " + message);
  }

  private static void removeTree(Path root) throws IOException {
    try (var entries = Files.walk(root)) {
      for (Path entry : entries.sorted(Comparator.reverseOrder()).toList()) Files.delete(entry);
    }
  }

  private static final class Fixture implements AutoCloseable {
    final Path root = Files.createTempDirectory("native-resource-test-");
    final String os = System.getProperty("os.name");
    final String arch = System.getProperty("os.arch");
    final String platform = NativeLibraryLoader.platform(os, arch);
    final String prefix = "META-INF/native/" + platform + "/";
    final String libraryName = "libspark_connect_jni."
        + (platform.startsWith("macos-") ? "dylib" : "so");
    final Properties adapterMetadata = new Properties();
    final Properties nativeMetadata = new Properties();

    Fixture(byte[] library) throws Exception {
      Files.createDirectories(root.resolve(prefix));
      adapterMetadata.setProperty("artifact.version", "test-1.0");
      adapterMetadata.setProperty("native.abi.version", "1");
      nativeMetadata.putAll(adapterMetadata);
      nativeMetadata.setProperty("library.name", libraryName);
      nativeMetadata.setProperty("library.sha256",
          HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(library)));
      writeAdapterMetadata();
      writeNativeMetadata();
      Files.write(root.resolve(prefix + libraryName), library);
    }

    void writeAdapterMetadata() throws IOException {
      writeProperties(root.resolve(ADAPTER_PROPERTIES), adapterMetadata);
    }

    void writeNativeMetadata() throws IOException {
      writeProperties(root.resolve(prefix + "native.properties"), nativeMetadata);
    }

    Path extract() throws Exception {
      try (URLClassLoader resources = loader()) {
        return NativeLibraryLoader.extract(resources, os, arch);
      }
    }

    URLClassLoader loader() throws IOException {
      return new URLClassLoader(new URL[] {root.toUri().toURL()}, null);
    }

    @Override
    public void close() throws IOException {
      removeTree(root);
    }

    private static void writeProperties(Path file, Properties values) throws IOException {
      try (var output = Files.newBufferedWriter(file, StandardCharsets.ISO_8859_1)) {
        values.store(output, "test metadata");
      }
    }
  }
}
