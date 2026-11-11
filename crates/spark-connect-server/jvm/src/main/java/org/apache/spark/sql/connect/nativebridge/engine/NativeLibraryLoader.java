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

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Enumeration;
import java.util.HexFormat;
import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

/** Loads a matched native artifact from this adapter's defining class loader. */
final class NativeLibraryLoader {
  private static final String ADAPTER_PROPERTIES =
      "META-INF/spark-connect-native-adapter.properties";
  private static final String ABI_VERSION = "1";
  private static final int MAX_METADATA_BYTES = 16 * 1024;
  private static final long MAX_LIBRARY_BYTES = 512L * 1024 * 1024;
  private static Path loadedPath;
  private static ClassLoader loadedResourceLoader;
  private static String loadedPlatform;

  private NativeLibraryLoader() {}

  static Path loadBundled() {
    return loadBundled(NativeLibraryLoader.class.getClassLoader(),
        System.getProperty("os.name"), System.getProperty("os.arch"));
  }

  static synchronized Path loadAbsolute(Path library) {
    Objects.requireNonNull(library, "library");
    if (!library.isAbsolute()) {
      throw new IllegalArgumentException("native library path must be absolute");
    }
    try {
      Path resolved = library.toRealPath();
      if (loadedPath != null) {
        if (loadedPath.equals(resolved)) return loadedPath;
        throw new IllegalStateException(
            "a different Spark Connect native library is already loaded: " + loadedPath);
      }
      System.load(resolved.toString());
      loadedPath = resolved;
      return resolved;
    } catch (IOException error) {
      throw new IllegalStateException("cannot read Spark Connect native library " + library, error);
    }
  }

  // Package visibility permits deterministic resource and platform tests without Spark jars.
  static synchronized Path loadBundled(ClassLoader resources, String osName, String osArch) {
    String platform = platform(osName, osArch);
    if (loadedPath != null) {
      if (loadedResourceLoader == resources && platform.equals(loadedPlatform)) return loadedPath;
      throw new IllegalStateException("a Spark Connect native library is already loaded from "
          + loadedPath + "; use one native artifact source per adapter class loader");
    }
    Path library = extract(resources, osName, osArch);
    try {
      System.load(library.toString());
      loadedPath = library;
      loadedResourceLoader = resources;
      loadedPlatform = platform;
      return library;
    } catch (RuntimeException | Error failure) {
      deleteExtraction(library, failure);
      throw failure;
    }
  }

  static String platform(String osName, String osArch) {
    String os = Objects.requireNonNull(osName, "os.name").toLowerCase(Locale.ROOT);
    String arch = Objects.requireNonNull(osArch, "os.arch").toLowerCase(Locale.ROOT);
    String operatingSystem;
    if (os.equals("linux")) operatingSystem = "linux";
    else if (os.equals("mac os x") || os.equals("macos") || os.equals("darwin")) {
      operatingSystem = "macos";
    } else {
      throw new UnsupportedOperationException("unsupported Spark Connect native operating system: "
          + osName + "; packaged libraries support Linux and macOS");
    }
    String architecture;
    if (arch.equals("amd64") || arch.equals("x86_64")) architecture = "x86_64";
    else if (arch.equals("aarch64") || arch.equals("arm64")) architecture = "aarch64";
    else {
      throw new UnsupportedOperationException("unsupported Spark Connect native architecture: "
          + osArch + "; packaged libraries support x86_64 and aarch64");
    }
    return operatingSystem + "-" + architecture;
  }

  static Path extract(ClassLoader resources, String osName, String osArch) {
    Objects.requireNonNull(resources, "resource class loader");
    String platform = platform(osName, osArch);
    String libraryName = "libspark_connect_jni." + (platform.startsWith("macos-") ? "dylib" : "so");
    String prefix = "META-INF/native/" + platform + "/";
    Path directory = null;
    Path library = null;
    try {
      Properties adapter = properties(uniqueResource(resources, ADAPTER_PROPERTIES));
      String version = required(adapter, "artifact.version");
      requireEquals(ABI_VERSION, required(adapter, "native.abi.version"), "adapter native ABI");
      Properties nativeArtifact = properties(
          uniqueResource(resources, prefix + "native.properties"));
      requireEquals(version, required(nativeArtifact, "artifact.version"),
          "native artifact version");
      requireEquals(ABI_VERSION, required(nativeArtifact, "native.abi.version"), "native ABI");
      requireEquals(libraryName, required(nativeArtifact, "library.name"), "native library name");
      String expectedHash = required(nativeArtifact, "library.sha256");
      if (!expectedHash.matches("[0-9a-f]{64}")) {
        throw new IOException(
            "native library SHA-256 must contain 64 lowercase hexadecimal digits");
      }
      URL resource = uniqueResource(resources, prefix + libraryName);
      directory = Files.createTempDirectory("spark-connect-native-",
          PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rwx------")));
      directory = directory.toRealPath();
      // Delete-on-exit is ordered in reverse registration order: file before directory.
      directory.toFile().deleteOnExit();
      library = Files.createFile(directory.resolve(libraryName),
          PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rw-------")));
      library.toFile().deleteOnExit();
      MessageDigest digest = sha256();
      long total = 0;
      try (InputStream input = resource.openStream();
           OutputStream output = Files.newOutputStream(library, StandardOpenOption.WRITE)) {
        byte[] buffer = new byte[64 * 1024];
        int read;
        while ((read = input.read(buffer)) != -1) {
          total += read;
          if (total > MAX_LIBRARY_BYTES) throw new IOException("native library exceeds 512 MiB");
          digest.update(buffer, 0, read);
          output.write(buffer, 0, read);
        }
      }
      if (total == 0) throw new IOException("native library resource is empty");
      String actualHash = HexFormat.of().formatHex(digest.digest());
      if (!expectedHash.equals(actualHash)) {
        throw new IOException("native library SHA-256 mismatch for " + platform);
      }
      return library;
    } catch (IOException | RuntimeException failure) {
      if (library != null) deleteExtraction(library, failure);
      else if (directory != null) {
        try {
          Files.deleteIfExists(directory);
        } catch (IOException cleanup) {
          failure.addSuppressed(cleanup);
        }
      }
      throw new IllegalStateException("cannot load bundled Spark Connect native library for "
          + platform + ": " + failure.getMessage(), failure);
    }
  }

  private static URL uniqueResource(ClassLoader resources, String name) throws IOException {
    Enumeration<URL> matches = resources.getResources(name);
    if (!matches.hasMoreElements()) {
      throw new IOException("missing classpath resource " + name
          + "; include the adapter JAR and matching platform native JAR");
    }
    URL first = matches.nextElement();
    if (matches.hasMoreElements()) {
      throw new IOException("multiple classpath resources for " + name
          + "; include exactly one matching artifact version");
    }
    return first;
  }

  private static Properties properties(URL resource) throws IOException {
    byte[] bytes;
    try (InputStream input = resource.openStream()) {
      bytes = input.readNBytes(MAX_METADATA_BYTES + 1);
    }
    if (bytes.length > MAX_METADATA_BYTES) {
      throw new IOException("native artifact metadata exceeds 16 KiB");
    }
    Properties values = new Properties();
    values.load(new ByteArrayInputStream(bytes));
    return values;
  }

  private static String required(Properties values, String key) throws IOException {
    String value = values.getProperty(key);
    if (value == null || value.isEmpty()) {
      throw new IOException("missing native artifact property " + key);
    }
    return value;
  }

  private static void requireEquals(String expected, String actual, String description)
      throws IOException {
    if (!expected.equals(actual)) {
      throw new IOException(description + " mismatch: expected " + expected + ", found " + actual);
    }
  }

  private static MessageDigest sha256() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException impossible) {
      throw new AssertionError("JDK must support SHA-256", impossible);
    }
  }

  private static void deleteExtraction(Path library, Throwable failure) {
    try {
      Files.deleteIfExists(library);
      Files.deleteIfExists(library.getParent());
    } catch (IOException cleanup) {
      failure.addSuppressed(cleanup);
    }
  }
}
