// C ABI over the PhononCoreML Swift package (compiled into this module by
// build.rs). Rust owns each session through `crate::phonon::Phonon`, which
// serialises every call; the lock here keeps a misuse from racing the engine.
import CoreML
import Foundation

private struct PhononBridgeError: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? { message }
}

private struct BridgeSpan: Codable {
    let startMs: Int64
    let endMs: Int64
    let text: String
}

private struct BridgeTranscript: Codable {
    let text: String
    let segments: [BridgeSpan]
    let words: [BridgeSpan]
}

private struct BridgeLoadReport: Codable {
    let compileSeconds: Double
    let loadSeconds: Double
    let functionLoadSeconds: [String: Double]
    let prewarmSeconds: Double
    let singleShotMaxSamples: Int
    let eagerFunctions: [Double]
}

/// Hex keeps one window up to this many samples. PhononCoreML 1.1.2 plans a
/// single window through 35.0 s but selects the encoder for
/// `frames * 0.01 + 0.02` seconds, so 559,680...560,000 samples choose no
/// function and fail. 559,520 samples (34.97 s) is the largest multiple of the
/// 160-sample hop whose selection still fits the 35 s function; longer audio
/// uses the package's own pause-cut 15 s windows, so no audio is dropped.
let hexPhononSingleShotMaxSamples = 559_520

/// Inputs shorter than two 10 ms feature frames cannot hold a word, and the
/// package's front end has no frames to normalise below 160 samples.
let hexPhononMinimumSamples = 320

private final class PhononSession {
    let transcriber: Transcriber
    let report: BridgeLoadReport
    let lock = NSLock()

    init(bundle: URL, singleShotMaxSamples: Int, eagerAllFunctions: Bool) throws {
        let manifest = try JSONDecoder().decode(
            Manifest.self,
            from: Data(contentsOf: bundle.appendingPathComponent("manifest.json"))
        )
        try PhononSession.compileBesideBundle(bundle: bundle, manifest: manifest)
        var options = Transcriber.Options()
        options.singleShotMaxSeconds = Double(singleShotMaxSamples) / 16_000
        // Optionally prepare every encoder function now, during model
        // preparation, so the first dictation of each length never pays the
        // Neural Engine compile. Otherwise the package loads its 15 s function
        // eagerly and the others on first use.
        if eagerAllFunctions { options.eagerFunctions = manifest.functions_s }
        let transcriber = try Transcriber(bundle: bundle, options: options)
        let prewarmStarted = Date()
        _ = try transcriber.transcribe(PhononSession.prewarmSignal())
        transcriber.resetTiming()
        self.transcriber = transcriber
        report = BridgeLoadReport(
            compileSeconds: transcriber.compileSeconds,
            loadSeconds: transcriber.loadSeconds,
            functionLoadSeconds: Dictionary(
                uniqueKeysWithValues: transcriber.functionLoadSeconds.map { ("\($0.key)", $0.value) }
            ),
            prewarmSeconds: Date().timeIntervalSince(prewarmStarted),
            singleShotMaxSamples: singleShotMaxSamples,
            eagerFunctions: transcriber.loadedFunctions.map(Double.init)
        )
    }

    /// The package reuses `<bundle>/<package>.mlmodelc` when present. Compile
    /// it there once, publishing atomically, so the compiled model lives at a
    /// stable path inside Hex's own models directory and is deleted with it.
    static func compileBesideBundle(bundle: URL, manifest: Manifest) throws {
        let name = (manifest.multifunction as NSString).deletingPathExtension
        let compiled = bundle.appendingPathComponent(name).appendingPathExtension("mlmodelc")
        let files = FileManager.default
        if files.fileExists(atPath: compiled.path) { return }
        let temporary = try MLModel.compileModel(at: bundle.appendingPathComponent(manifest.multifunction))
        let staged = bundle.appendingPathComponent(".\(name)-\(UUID().uuidString).mlmodelc")
        defer {
            try? files.removeItem(at: temporary)
            try? files.removeItem(at: staged)
        }
        try files.moveItem(at: temporary, to: staged)
        if rename(staged.path, compiled.path) != 0 && !files.fileExists(atPath: compiled.path) {
            throw PhononBridgeError("Could not store the compiled Phonon model: \(String(cString: strerror(errno)))")
        }
    }

    /// One second of deterministic low-level noise: above the package's
    /// silence gate, so the prewarm runs the mel, encoder and decoder path.
    static func prewarmSignal() -> [Float] {
        var state: UInt32 = 0x1234_5678
        return (0..<16_000).map { _ in
            state = state &* 1_664_525 &+ 1_013_904_223
            return (Float(state >> 8) / Float(1 << 24) - 0.5) * 0.02
        }
    }

    func transcribe(_ samples: [Float]) throws -> BridgeTranscript {
        lock.lock()
        defer { lock.unlock() }
        if samples.count < hexPhononMinimumSamples {
            return BridgeTranscript(text: "", segments: [], words: [])
        }
        let result = try transcriber.transcribe(samples)
        let duration = Int64((Double(samples.count) * 1000 / 16_000).rounded(.up))
        let words = try result.words.map { try span($0.text, $0.start, $0.end, limit: duration) }
        let segments = try result.segments.map { try span($0.text, $0.start, $0.end, limit: duration) }
        return BridgeTranscript(text: result.text, segments: segments, words: words)
    }

    private func span(_ text: String, _ start: Double, _ end: Double, limit: Int64) throws -> BridgeSpan {
        guard start.isFinite, end.isFinite else {
            throw PhononBridgeError("Phonon returned a non-finite timestamp.")
        }
        let startMs = min(max(Int64((start * 1000).rounded()), 0), limit)
        let endMs = min(max(Int64((end * 1000).rounded()), startMs), limit)
        return BridgeSpan(startMs: startMs, endMs: endMs, text: text)
    }
}

private func writeError(
    _ error: Error,
    to destination: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) {
    destination?.pointee = strdup(
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    )
}

private func encode<T: Encodable>(_ value: T) throws -> UnsafeMutablePointer<CChar> {
    let data = try JSONEncoder().encode(value)
    guard let json = String(data: data, encoding: .utf8), let pointer = strdup(json) else {
        throw PhononBridgeError("Could not encode the Phonon result.")
    }
    return pointer
}

/// Loads, compiles if needed, and prewarms a verified bundle. Returns a
/// retained session, or null with `error` set.
@_cdecl("hex_phonon_create")
public func hexPhononCreate(
    _ directory: UnsafePointer<CChar>?,
    _ singleShotMaxSamples: Int,
    _ eagerAllFunctions: Int32,
    _ report: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    _ error: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutableRawPointer? {
    guard let directory else {
        writeError(PhononBridgeError("Phonon needs a model directory."), to: error)
        return nil
    }
    guard singleShotMaxSamples > 0 else {
        writeError(PhononBridgeError("Phonon needs a positive single-window limit."), to: error)
        return nil
    }
    do {
        let session = try autoreleasepool {
            try PhononSession(
                bundle: URL(fileURLWithPath: String(cString: directory), isDirectory: true),
                singleShotMaxSamples: singleShotMaxSamples,
                eagerAllFunctions: eagerAllFunctions != 0
            )
        }
        report?.pointee = try? encode(session.report)
        return Unmanaged.passRetained(session).toOpaque()
    } catch let failure {
        writeError(failure, to: error)
        return nil
    }
}

/// Transcribes mono 16 kHz samples exactly as supplied: no resampling,
/// padding or trimming. Returns canonical transcript JSON, or null with
/// `error` set.
@_cdecl("hex_phonon_transcribe")
public func hexPhononTranscribe(
    _ opaque: UnsafeMutableRawPointer?,
    _ samples: UnsafePointer<Float>?,
    _ count: Int,
    _ error: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutablePointer<CChar>? {
    guard let opaque, count >= 0, count == 0 || samples != nil else {
        writeError(PhononBridgeError("Phonon received invalid audio."), to: error)
        return nil
    }
    let session = Unmanaged<PhononSession>.fromOpaque(opaque).takeUnretainedValue()
    let audio = count == 0 ? [] : Array(UnsafeBufferPointer(start: samples, count: count))
    guard audio.allSatisfy(\.isFinite) else {
        writeError(PhononBridgeError("Phonon received non-finite audio."), to: error)
        return nil
    }
    do {
        return try autoreleasepool { try encode(session.transcribe(audio)) }
    } catch let failure {
        writeError(failure, to: error)
        return nil
    }
}

@_cdecl("hex_phonon_release")
public func hexPhononRelease(_ opaque: UnsafeMutableRawPointer?) {
    guard let opaque else { return }
    Unmanaged<PhononSession>.fromOpaque(opaque).release()
}

@_cdecl("hex_phonon_default_single_shot_max_samples")
public func hexPhononDefaultSingleShotMaxSamples() -> Int {
    hexPhononSingleShotMaxSamples
}

@_cdecl("hex_phonon_free_string")
public func hexPhononFreeString(_ string: UnsafeMutablePointer<CChar>?) {
    free(string)
}
