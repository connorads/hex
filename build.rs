use std::path::{Path, PathBuf};
use std::process::Command;

/// fermionresearch/phonon-coreml release 1.1.2 (Apache-2.0).
const PHONON_COREML_REPOSITORY: &str = "https://github.com/fermionresearch/phonon-coreml";
const PHONON_COREML_COMMIT: &str = "bf59e574639b060e30ee1a3f9c0e003fed5744c8";
/// The release's prebuilt greedy-TDT decoder (universal static C library).
const PHONON_TDT_LIBRARY: &str = "Binaries/PhononTDT.xcframework/macos-arm64_x86_64/libPhononTDT.a";
const PHONON_TDT_SHA256: &str = "22fe6f7b949a4a56b4538f418b9b6142d6488e903a79eee9ad6bd8aca62930e5";
const PHONON_COREML_SOURCES: &[&str] = &[
    "AudioSource.swift",
    "DecoderBin.swift",
    "LogMel.swift",
    "Segmenter.swift",
    "Transcriber.swift",
    "Windower.swift",
    "Words.swift",
];

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
            // transcribe-cpp's static manifest records OpenBLAS as an absolute
            // link argument, which Cargo does not propagate through the -sys
            // crate to this final binary. Keep it after the static archives so
            // GNU ld's --as-needed does not discard it before seeing cblas_*.
            println!("cargo:rustc-link-arg=-lopenblas");
        }
        return;
    }
    let output = Command::new("xcode-select")
        .arg("-p")
        .output()
        .expect("xcode-select is required to link ScreenCaptureKit");
    assert!(output.status.success(), "xcode-select -p failed");
    let developer_dir = String::from_utf8(output.stdout).expect("Xcode path must be UTF-8");
    compile_apple_speech_bridge();
    if std::env::var("TARGET").as_deref() == Ok("aarch64-apple-darwin") {
        compile_phonon_bridge();
    }
    compile_indicator_shader();
    compile_permission_guide();
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    for runtime in ["swift-5.5/macosx", "swift/macosx"] {
        let swift_runtime = format!(
            "{}/Toolchains/XcodeDefault.xctoolchain/usr/lib/{runtime}",
            developer_dir.trim()
        );
        println!("cargo:rustc-link-arg=-Wl,-rpath,{swift_runtime}");
    }
}

fn compile_permission_guide() {
    println!("cargo:rerun-if-changed=native/permission_guide.m");
    println!("cargo:rustc-link-lib=framework=AppKit");
    println!("cargo:rustc-link-lib=framework=ApplicationServices");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");

    cc::Build::new()
        .file("native/permission_guide.m")
        .flag("-fobjc-arc")
        .compile("hex_permission_guide");
}

fn compile_apple_speech_bridge() {
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let output_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let source = manifest_dir.join("src/apple_speech.swift");
    let object = output_dir.join("apple_speech.o");
    let sdk = Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-path"])
        .output()
        .expect("xcrun is required to compile Apple Speech support");
    assert!(sdk.status.success(), "could not locate the macOS SDK");
    let target = match std::env::var("TARGET").as_deref() {
        Ok("aarch64-apple-darwin") => "arm64-apple-macosx15.0",
        Ok("x86_64-apple-darwin") => "x86_64-apple-macosx15.0",
        Ok(target) => panic!("unsupported macOS target for Apple Speech: {target}"),
        Err(error) => panic!("TARGET is unavailable: {error}"),
    };
    println!("cargo:rerun-if-changed={}", source.display());
    let swift = Command::new("xcrun")
        .args([
            "swiftc",
            "-parse-as-library",
            "-emit-object",
            "-module-name",
            "HEXAppleSpeechBridge",
            "-sdk",
        ])
        .arg(String::from_utf8_lossy(&sdk.stdout).trim())
        .args(["-target", target])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .expect("xcrun swiftc is required to compile Apple Speech support");
    assert!(swift.success(), "Apple Speech bridge compilation failed");
    println!("cargo:rustc-link-arg={}", object.display());
    println!("cargo:rustc-link-lib=framework=AVFAudio");
    println!("cargo:rustc-link-lib=framework=Speech");
}

/// Compiles the pinned PhononCoreML sources together with `src/phonon.swift`
/// into one object and links the release's static decoder. Nothing from the
/// package is shipped as a separate dynamic library. Set
/// `HEX_PHONON_COREML_SOURCE` to a checkout of the pinned commit to build
/// offline; otherwise the commit is fetched into `OUT_DIR`.
fn compile_phonon_bridge() {
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let output_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let bridge = manifest_dir.join("src/phonon.swift");
    println!("cargo:rerun-if-changed={}", bridge.display());
    println!("cargo:rerun-if-env-changed=HEX_PHONON_COREML_SOURCE");
    let package = match std::env::var_os("HEX_PHONON_COREML_SOURCE") {
        Some(path) => PathBuf::from(path),
        None => fetch_phonon_coreml(&output_dir.join("phonon-coreml")),
    };
    let decoder = package.join(PHONON_TDT_LIBRARY);
    let digest = Command::new("shasum")
        .args(["-a", "256"])
        .arg(&decoder)
        .output()
        .expect("shasum is required to verify the Phonon decoder");
    assert!(
        digest.status.success(),
        "could not hash {}",
        decoder.display()
    );
    assert_eq!(
        String::from_utf8_lossy(&digest.stdout)
            .split_whitespace()
            .next(),
        Some(PHONON_TDT_SHA256),
        "{} does not match the pinned phonon-coreml release",
        decoder.display()
    );
    let sdk = Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-path"])
        .output()
        .expect("xcrun is required to compile Phonon support");
    assert!(sdk.status.success(), "could not locate the macOS SDK");
    let object = output_dir.join("phonon.o");
    let sources = package.join("Sources/PhononCoreML");
    let swift = Command::new("xcrun")
        .args([
            "swiftc",
            "-parse-as-library",
            "-emit-object",
            "-whole-module-optimization",
            "-O",
            "-module-name",
            "HEXPhononBridge",
            "-sdk",
        ])
        .arg(String::from_utf8_lossy(&sdk.stdout).trim())
        .args(["-target", "arm64-apple-macosx15.0", "-I"])
        .arg(decoder.parent().unwrap().join("Headers"))
        .arg(&bridge)
        .args(
            PHONON_COREML_SOURCES
                .iter()
                .map(|source| sources.join(source)),
        )
        .arg("-o")
        .arg(&object)
        .status()
        .expect("xcrun swiftc is required to compile Phonon support");
    assert!(swift.success(), "Phonon bridge compilation failed");
    println!("cargo:rustc-link-arg={}", object.display());
    println!("cargo:rustc-link-arg={}", decoder.display());
    for framework in ["Accelerate", "AVFoundation", "CoreML"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
}

fn fetch_phonon_coreml(checkout: &Path) -> PathBuf {
    let git = |arguments: &[&str]| {
        let status = Command::new("git")
            .arg("-C")
            .arg(checkout)
            .args(arguments)
            .status()
            .expect("git is required to fetch phonon-coreml");
        assert!(status.success(), "git {} failed", arguments.join(" "));
    };
    let head = Command::new("git")
        .arg("-C")
        .arg(checkout)
        .args(["rev-parse", "HEAD"])
        .output();
    if head.is_ok_and(|head| String::from_utf8_lossy(&head.stdout).trim() == PHONON_COREML_COMMIT) {
        return checkout.to_path_buf();
    }
    let _ = std::fs::remove_dir_all(checkout);
    std::fs::create_dir_all(checkout).unwrap();
    git(&["init", "--quiet"]);
    git(&["remote", "add", "origin", PHONON_COREML_REPOSITORY]);
    git(&[
        "fetch",
        "--quiet",
        "--depth",
        "1",
        "origin",
        PHONON_COREML_COMMIT,
    ]);
    git(&["checkout", "--quiet", "--detach", "FETCH_HEAD"]);
    checkout.to_path_buf()
}

fn compile_indicator_shader() {
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let output_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let source = manifest_dir.join("src/dictation_indicator.metal");
    let air = output_dir.join("dictation_indicator.air");
    let library = output_dir.join("dictation_indicator.metallib");
    println!("cargo:rerun-if-changed={}", source.display());

    let metal = Command::new("xcrun")
        .args(["--sdk", "macosx", "metal", "-c"])
        .arg(&source)
        .arg("-o")
        .arg(&air)
        .status()
        .expect("xcrun metal is required to build the dictation indicator");
    assert!(metal.success(), "Metal shader compilation failed");

    let metallib = Command::new("xcrun")
        .args(["--sdk", "macosx", "metallib"])
        .arg(&air)
        .arg("-o")
        .arg(&library)
        .status()
        .expect("xcrun metallib is required to build the dictation indicator");
    assert!(metallib.success(), "Metal library creation failed");
}
