//! The native Phonon-2 Core ML backend: one warm Swift `Transcriber` behind
//! a narrow C ABI (`src/phonon.swift`), owned and serialised by Rust.

use std::ffi::{CStr, CString, c_char, c_float, c_void};
use std::path::Path;
use std::ptr::{self, NonNull};
use std::time::Instant;

use color_eyre::eyre::{Result, WrapErr, bail, eyre};
use serde::Deserialize;

use crate::transcription::{Transcript, TranscriptSegment};
use crate::transcription_models::{TranscriptionSelection, model_path, validate};

unsafe extern "C" {
    fn hex_phonon_create(
        directory: *const c_char,
        single_shot_max_samples: isize,
        eager_all_functions: i32,
        report: *mut *mut c_char,
        error: *mut *mut c_char,
    ) -> *mut c_void;
    fn hex_phonon_transcribe(
        session: *mut c_void,
        samples: *const c_float,
        count: isize,
        error: *mut *mut c_char,
    ) -> *mut c_char;
    fn hex_phonon_release(session: *mut c_void);
    fn hex_phonon_default_single_shot_max_samples() -> isize;
    fn hex_phonon_free_string(string: *mut c_char);
}

/// The bridge's wire shape: the canonical transcript plus word timings.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhononTranscript {
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
    pub words: Vec<TranscriptSegment>,
}

/// How a session prepares the package. Hex uses the defaults; tests vary them.
#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    /// Overrides Hex's one-window limit (tests reproduce the upstream default).
    pub single_shot_max_samples: Option<usize>,
    /// Load every encoder function during preparation instead of on first use.
    pub eager_all_functions: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            single_shot_max_samples: None,
            eager_all_functions: EAGER_ALL_FUNCTIONS,
        }
    }
}

const EAGER_ALL_FUNCTIONS: bool = true;

pub struct Phonon {
    session: NonNull<c_void>,
    selection: Option<TranscriptionSelection>,
}

// The Swift session is a retained class instance with its own lock. Rust
// gives it exactly one owner and requires `&mut self` for inference, so it
// may move to the dedicated inference thread.
unsafe impl Send for Phonon {}

impl Phonon {
    /// Loads the verified bundle for `selection`. Never downloads: an
    /// unverified bundle is an error that leaves the caller's model in place.
    pub fn load_selection(selection: &TranscriptionSelection) -> Result<Self> {
        let definition = validate(selection)?;
        if !crate::transcription_models::is_verified(definition) {
            if !crate::transcription_models::is_installed(definition, &selection.language) {
                bail!(
                    "{} is not installed. Open Settings to download it.",
                    definition.name
                );
            }
            // A complete bundle whose receipt is missing or stale (for
            // example after a restore changed modification times) is hashed
            // again here; it never downloads or replaces files.
            crate::transcription_models::verify_installed(
                definition,
                &std::sync::atomic::AtomicBool::new(false),
            )?;
        }
        let mut phonon = Self::load_from(&model_path(definition)?, LoadOptions::default())?;
        phonon.selection = Some(selection.clone());
        Ok(phonon)
    }

    /// Loads a bundle directory directly.
    pub fn load_from(directory: &Path, options: LoadOptions) -> Result<Self> {
        let directory = CString::new(directory.as_os_str().as_encoded_bytes())
            .wrap_err("Phonon model path contains a NUL byte")?;
        let limit = match options.single_shot_max_samples {
            Some(limit) => isize::try_from(limit).wrap_err("Phonon window limit is too large")?,
            None => unsafe { hex_phonon_default_single_shot_max_samples() },
        };
        let started = Instant::now();
        let mut report = ptr::null_mut();
        let mut error = ptr::null_mut();
        let session = unsafe {
            hex_phonon_create(
                directory.as_ptr(),
                limit,
                i32::from(options.eager_all_functions),
                &mut report,
                &mut error,
            )
        };
        let report = take_string(report);
        let Some(session) = NonNull::new(session) else {
            return Err(eyre!(
                "could not load Phonon-2 Core ML: {}",
                take_string(error).unwrap_or_else(|| "the bridge reported no error".into())
            ));
        };
        tracing::info!(
            load_ms = started.elapsed().as_millis(),
            report = report.as_deref().unwrap_or("{}"),
            "loaded and prewarmed Phonon-2 Core ML"
        );
        Ok(Self {
            session,
            selection: None,
        })
    }

    pub fn matches_selection(&self, selection: &TranscriptionSelection) -> bool {
        self.selection.as_ref() == Some(selection)
    }

    /// Transcribes exactly the supplied normalised mono 16 kHz samples.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<PhononTranscript> {
        validate_samples(samples)?;
        let mut error = ptr::null_mut();
        let json = unsafe {
            hex_phonon_transcribe(
                self.session.as_ptr(),
                samples.as_ptr(),
                samples.len() as isize,
                &mut error,
            )
        };
        if json.is_null() {
            return Err(eyre!(
                "Phonon transcription failed: {}",
                take_string(error).unwrap_or_else(|| "the bridge reported no error".into())
            ));
        }
        let json = take_string(json).unwrap_or_default();
        decode(&json)
    }

    pub fn transcribe_text(&mut self, samples: &[f32]) -> Result<String> {
        self.transcribe(samples).map(|result| result.text)
    }

    pub fn transcribe_segments(&mut self, samples: &[f32]) -> Result<Transcript> {
        self.transcribe(samples).map(|result| Transcript {
            text: result.text,
            segments: result.segments,
        })
    }
}

impl Drop for Phonon {
    fn drop(&mut self) {
        // Inference is synchronous behind `&mut self`, so no call can be in
        // flight when the owner drops the session.
        unsafe { hex_phonon_release(self.session.as_ptr()) };
    }
}

fn validate_samples(samples: &[f32]) -> Result<()> {
    if let Some(index) = samples.iter().position(|sample| !sample.is_finite()) {
        bail!("Phonon received a non-finite audio sample at index {index}");
    }
    Ok(())
}

fn decode(json: &str) -> Result<PhononTranscript> {
    let transcript: PhononTranscript =
        serde_json::from_str(json).wrap_err("Phonon returned an invalid transcript")?;
    for span in transcript.words.iter().chain(&transcript.segments) {
        if span.start_ms < 0 || span.end_ms < span.start_ms {
            bail!("Phonon returned an invalid timestamp");
        }
    }
    Ok(transcript)
}

fn take_string(string: *mut c_char) -> Option<String> {
    if string.is_null() {
        return None;
    }
    let value = unsafe { CStr::from_ptr(string) }
        .to_string_lossy()
        .into_owned();
    unsafe { hex_phonon_free_string(string) };
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_json_keeps_full_text_and_millisecond_word_timings() {
        let transcript = decode(
            r#"{"text":"Hello, world.","segments":[{"startMs":0,"endMs":900,"text":"Hello, world."}],
                "words":[{"startMs":80,"endMs":400,"text":"Hello,"},{"startMs":480,"endMs":880,"text":"world."}]}"#,
        )
        .unwrap();
        assert_eq!(transcript.text, "Hello, world.");
        assert_eq!(transcript.words.len(), 2);
        assert_eq!(transcript.words[1].start_ms, 480);
        assert_eq!(transcript.segments[0].end_ms, 900);
    }

    #[test]
    fn invalid_bridge_output_is_rejected() {
        for json in [
            r#"{"text":"x","segments":[],"words":[{"startMs":-1,"endMs":2,"text":"x"}]}"#,
            r#"{"text":"x","segments":[],"words":[{"startMs":5,"endMs":2,"text":"x"}]}"#,
            r#"{"text":"x","segments":[]}"#,
            "not json",
        ] {
            assert!(decode(json).is_err(), "{json}");
        }
    }

    #[test]
    fn non_finite_audio_is_rejected_before_the_bridge() {
        assert!(validate_samples(&[0.0, 0.5, -0.5]).is_ok());
        assert!(validate_samples(&[]).is_ok());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = validate_samples(&[0.0, bad]).unwrap_err().to_string();
            assert!(error.contains("index 1"), "{error}");
        }
    }
}

/// Real Core ML inference through the compiled bridge. These need an Apple
/// silicon Mac, a verified bundle in `HEX_PHONON_BUNDLE`, and the fixtures
/// written by `scripts/phonon-fixtures.py` in `HEX_PHONON_FIXTURES`:
///
/// ```sh
/// cargo test --bin voice-control phonon::native -- --ignored --nocapture --test-threads=1
/// ```
#[cfg(test)]
mod native {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::time::Instant;

    use super::*;
    use crate::transcription_benchmark::word_errors;

    /// The single-window limit PhononCoreML 1.1.2 uses by default (35 s).
    const UPSTREAM_SINGLE_SHOT_MAX_SAMPLES: usize = 560_000;
    const SAMPLE_RATE: usize = 16_000;

    fn bundle() -> PathBuf {
        PathBuf::from(std::env::var_os("HEX_PHONON_BUNDLE").expect("set HEX_PHONON_BUNDLE"))
    }

    fn fixtures() -> PathBuf {
        PathBuf::from(std::env::var_os("HEX_PHONON_FIXTURES").expect("set HEX_PHONON_FIXTURES"))
    }

    fn load() -> Phonon {
        Phonon::load_from(&bundle(), LoadOptions::default()).unwrap()
    }

    /// This process's current resident set size, in megabytes.
    fn resident_mb() -> f64 {
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .unwrap();
        String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<f64>()
            .unwrap()
            / 1_024.0
    }

    fn read_wav(path: &Path) -> Vec<f32> {
        let mut reader = hound::WavReader::open(path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate as usize, SAMPLE_RATE, "{}", path.display());
        assert_eq!(spec.channels, 1, "{}", path.display());
        assert_eq!(spec.bits_per_sample, 16, "{}", path.display());
        reader
            .samples::<i16>()
            .map(|sample| f32::from(sample.unwrap()) / 32_768.0)
            .collect()
    }

    /// The pinned LibriSpeech test-clean clips published with the model,
    /// with their reference texts.
    fn clips() -> Vec<(String, Vec<f32>, String)> {
        let expected: BTreeMap<String, String> =
            serde_json::from_slice(&std::fs::read(fixtures().join("expected.json")).unwrap())
                .unwrap();
        expected
            .into_iter()
            .map(|(id, text)| {
                let samples = read_wav(&fixtures().join(format!("{id}.wav")));
                (id, samples, text)
            })
            .collect()
    }

    /// Natural speech made of whole clips separated by `gap_ms` of silence,
    /// repeated until it reaches `minimum_samples`.
    fn speech(minimum_samples: usize, gap_ms: usize) -> (Vec<f32>, String) {
        let clips = clips();
        let mut samples = Vec::new();
        let mut reference = Vec::new();
        'outer: loop {
            for (_, clip, text) in &clips {
                samples.extend_from_slice(clip);
                samples.resize(samples.len() + gap_ms * SAMPLE_RATE / 1_000, 0.0);
                reference.push(text.as_str());
                if samples.len() >= minimum_samples {
                    break 'outer;
                }
            }
        }
        (samples, reference.join(" "))
    }

    fn exactly(samples: usize) -> Vec<f32> {
        let (mut audio, _) = speech(samples, 300);
        audio.truncate(samples);
        audio
    }

    fn wer(reference: &str, text: &str) -> f64 {
        let (errors, words) = word_errors(reference, text);
        errors as f64 / words as f64
    }

    fn assert_well_formed(transcript: &PhononTranscript, samples: usize) {
        let duration_ms = (samples * 1_000).div_ceil(SAMPLE_RATE) as i64;
        let mut previous = 0;
        for word in &transcript.words {
            assert!(word.start_ms >= previous - 250, "{word:?} is out of order");
            assert!(
                word.end_ms <= duration_ms,
                "{word:?} exceeds {duration_ms} ms"
            );
            previous = word.start_ms;
        }
        let joined = transcript
            .words
            .iter()
            .map(|word| word.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(joined, transcript.text);
    }

    #[test]
    #[ignore = "requires Apple silicon, HEX_PHONON_BUNDLE and HEX_PHONON_FIXTURES"]
    fn reference_clips_transcribe_exactly() {
        let mut phonon = load();
        for (id, samples, expected) in clips() {
            let transcript = phonon.transcribe(&samples).unwrap();
            println!(
                "{}",
                serde_json::json!({"clip": id, "samples": samples.len(), "text": transcript.text,
                    "expected": expected, "wer": wer(&expected, &transcript.text)})
            );
            assert_well_formed(&transcript, samples.len());
            assert_eq!(transcript.text, expected, "{id}");
        }
    }

    #[test]
    #[ignore = "requires Apple silicon, HEX_PHONON_BUNDLE and HEX_PHONON_FIXTURES"]
    fn upstream_35_second_limit_fails_and_hex_limit_covers_every_sample() {
        let mut upstream = Phonon::load_from(
            &bundle(),
            LoadOptions {
                single_shot_max_samples: Some(UPSTREAM_SINGLE_SHOT_MAX_SAMPLES),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let mut hex = load();
        let mut upstream_failures = Vec::new();
        for samples in [559_520, 559_680, 559_840, 560_000, 560_001] {
            let audio = exactly(samples);
            let upstream_result = upstream.transcribe(&audio);
            let hex_result = hex.transcribe(&audio).unwrap();
            println!(
                "{}",
                serde_json::json!({"samples": samples, "seconds": samples as f64 / 16_000.0,
                    "upstream_error": upstream_result.as_ref().err().map(ToString::to_string),
                    "hex_words": hex_result.words.len(), "hex_segments": hex_result.segments.len(),
                    "hex_text": hex_result.text})
            );
            if upstream_result.is_err() {
                upstream_failures.push(samples);
            }
            assert_well_formed(&hex_result, samples);
            let words_per_second = hex_result.words.len() as f64 / (samples as f64 / 16_000.0);
            assert!((1.5..5.0).contains(&words_per_second), "{words_per_second}");
        }
        // The native package reproduces the reference runner's defect: one
        // window through 35.0 s, but no encoder function for its last frames.
        assert_eq!(upstream_failures, [559_680, 559_840, 560_000]);
    }

    #[test]
    #[ignore = "requires Apple silicon, HEX_PHONON_BUNDLE and HEX_PHONON_FIXTURES"]
    fn encoder_bucket_boundaries_transcribe() {
        let mut phonon = load();
        for seconds in [5, 10, 15] {
            for samples in [
                seconds * SAMPLE_RATE - 160,
                seconds * SAMPLE_RATE,
                seconds * SAMPLE_RATE + 160,
            ] {
                let audio = exactly(samples);
                let transcript = phonon.transcribe(&audio).unwrap();
                println!(
                    "{}",
                    serde_json::json!({"samples": samples, "words": transcript.words.len(), "text": transcript.text})
                );
                assert_well_formed(&transcript, samples);
                assert!(!transcript.words.is_empty());
            }
        }
    }

    #[test]
    #[ignore = "requires Apple silicon, HEX_PHONON_BUNDLE and HEX_PHONON_FIXTURES"]
    fn long_natural_speech_keeps_every_window() {
        let mut phonon = load();
        for (minimum, gap_ms) in [
            (60 * SAMPLE_RATE, 300),
            (120 * SAMPLE_RATE, 300),
            (60 * SAMPLE_RATE, 0),
            (120 * SAMPLE_RATE, 0),
        ] {
            let (audio, reference) = speech(minimum, gap_ms);
            let started = Instant::now();
            let transcript = phonon.transcribe(&audio).unwrap();
            let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let reference_words = reference.split_whitespace().count();
            let error_rate = wer(&reference, &transcript.text);
            println!(
                "{}",
                serde_json::json!({"seconds": audio.len() as f64 / 16_000.0, "gap_ms": gap_ms,
                    "segments": transcript.segments.len(), "words": transcript.words.len(),
                    "reference_words": reference_words, "wer": error_rate, "inference_ms": elapsed_ms,
                    "text": transcript.text})
            );
            assert_well_formed(&transcript, audio.len());
            assert!(transcript.segments.len() > 1, "long audio must use windows");
            assert!(error_rate < 0.08, "WER {error_rate}");
            let ratio = transcript.words.len() as f64 / reference_words as f64;
            assert!(
                (0.95..1.05).contains(&ratio),
                "word ratio {ratio}: seam loss or duplication"
            );
        }
    }

    #[test]
    #[ignore = "requires Apple silicon, HEX_PHONON_BUNDLE and HEX_PHONON_FIXTURES"]
    fn edge_inputs_return_empty_text_or_explicit_errors() {
        let mut phonon = load();
        let noise = |samples: usize, amplitude: f32| {
            let mut state = 7_u32;
            (0..samples)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((state >> 8) as f32 / (1 << 24) as f32 - 0.5) * amplitude
                })
                .collect::<Vec<_>>()
        };
        for (name, audio) in [
            ("empty", Vec::new()),
            ("one sample", vec![0.5]),
            ("10 ms", noise(160, 0.5)),
            ("19.9 ms", noise(319, 0.5)),
            ("20 ms", noise(320, 0.5)),
            ("silence", vec![0.0; 2 * SAMPLE_RATE]),
            ("noise", noise(3 * SAMPLE_RATE, 0.2)),
        ] {
            let transcript = phonon.transcribe(&audio).unwrap();
            println!(
                "{}",
                serde_json::json!({"input": name, "text": transcript.text})
            );
            assert_well_formed(&transcript, audio.len());
            if matches!(
                name,
                "empty" | "one sample" | "10 ms" | "19.9 ms" | "silence"
            ) {
                assert!(transcript.text.is_empty(), "{name}: {}", transcript.text);
            }
        }
        let mut invalid = vec![0.0; SAMPLE_RATE];
        invalid[100] = f32::NAN;
        assert!(phonon.transcribe(&invalid).is_err());
        // The session stays usable after rejected input.
        let (_, samples, expected) = clips().remove(0);
        assert_eq!(phonon.transcribe(&samples).unwrap().text, expected);
    }

    /// Voice-delimited dictation on `say` speech: the full transcript ends
    /// with the control phrase, which the worker strips as text. Synthetic
    /// speech checks control handling, not recognition accuracy.
    #[test]
    #[ignore = "requires Apple silicon, HEX_PHONON_BUNDLE and HEX_PHONON_FIXTURES"]
    fn voice_delimited_dictation_drops_control_phrases() {
        let protocol = crate::dictation::DictationProtocol::try_new(
            vec!["say".into()],
            vec!["say paste".into(), "say stop".into()],
            vec!["say send".into()],
            vec!["say cancel".into(), "never mind".into()],
        )
        .unwrap();
        let mut phonon = load();
        for (name, expected) in [
            ("period-stop", "I don't understand this."),
            ("question-stop", "Is this working?"),
            ("exclamation-send", "Ship it!"),
            ("comma-stop", "Meet me at five."),
        ] {
            let samples = read_wav(&fixtures().join(format!("protocol/{name}.wav")));
            let full = phonon.transcribe(&samples).unwrap();
            let prepared = protocol.strip(&full.text);
            println!(
                "{}",
                serde_json::json!({"clip": name, "full": full.text,
                    "prepared": prepared, "expected": expected})
            );
            assert!(
                protocol.control_suffix(&full.text).is_some(),
                "{name}: no control phrase heard in {:?}",
                full.text
            );
            assert_eq!(wer(expected, &prepared), 0.0, "{name}: {prepared:?}");
            assert!(
                protocol.control_suffix(&prepared).is_none(),
                "{name}: {prepared:?}"
            );
        }
    }

    /// Lazy versus eager encoder functions: resident memory and the first
    /// call that needs each bucket. Lazy runs first because freed memory is
    /// not always returned to the system.
    #[test]
    #[ignore = "requires Apple silicon, HEX_PHONON_BUNDLE and HEX_PHONON_FIXTURES"]
    fn function_loading_memory_and_first_bucket_latency() {
        let buckets = [3, 8, 13, 30].map(|seconds| exactly(seconds * SAMPLE_RATE));
        for eager_all_functions in [false, true] {
            let before_mb = resident_mb();
            let started = Instant::now();
            let mut phonon = Phonon::load_from(
                &bundle(),
                LoadOptions {
                    eager_all_functions,
                    ..LoadOptions::default()
                },
            )
            .unwrap();
            let load_ms = started.elapsed().as_millis();
            let loaded_mb = resident_mb();
            let first_ms = buckets
                .iter()
                .map(|samples| {
                    let started = Instant::now();
                    assert!(!phonon.transcribe(samples).unwrap().words.is_empty());
                    started.elapsed().as_millis()
                })
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::json!({"eager_all_functions": eager_all_functions, "load_ms": load_ms,
                    "rss_before_mb": before_mb, "rss_loaded_mb": loaded_mb,
                    "rss_after_all_buckets_mb": resident_mb(),
                    "first_call_ms_3_8_13_30_s": first_ms})
            );
        }
    }

    #[test]
    #[ignore = "requires Apple silicon, HEX_PHONON_BUNDLE and HEX_PHONON_FIXTURES"]
    fn sessions_load_and_release_repeatedly_with_warm_latency() {
        let (_, samples, expected) = clips().remove(1);
        for cycle in 0..3 {
            let started = Instant::now();
            let mut phonon = load();
            let load_ms = started.elapsed().as_millis();
            let mut timings = (0..20)
                .map(|_| {
                    let started = Instant::now();
                    assert_eq!(phonon.transcribe(&samples).unwrap().text, expected);
                    started.elapsed().as_secs_f64() * 1_000.0
                })
                .collect::<Vec<_>>();
            let first_ms = timings[0];
            timings.sort_by(f64::total_cmp);
            println!(
                "{}",
                serde_json::json!({"cycle": cycle, "load_ms": load_ms, "audio_ms": samples.len() / 16,
                    "first_ms": first_ms, "median_ms": timings[10], "p95_ms": timings[18]})
            );
        }
    }
}
