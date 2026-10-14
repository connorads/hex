use color_eyre::eyre::Result;
use serde::Deserialize;

use crate::apple_speech::AppleSpeech;
use crate::dictation::DictationProtocol;
use crate::parakeet::Parakeet;
#[cfg(target_arch = "aarch64")]
use crate::phonon::Phonon;
use crate::transcription_models::{
    ModelRuntime, TranscriptionModelId, TranscriptionSelection, validate,
};

const UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES: usize = 3_200;

pub enum Transcriber {
    Gguf(Box<Parakeet>),
    AppleSpeech(AppleSpeech),
    #[cfg(target_arch = "aarch64")]
    Phonon(Phonon),
}

#[derive(Default)]
pub struct WarmTranscriber {
    active: Option<Transcriber>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Transcript {
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
}

impl Transcriber {
    pub fn load() -> Result<Self> {
        let (_, selection) = crate::app_settings::transcription_selection();
        Self::load_selection(&selection)
    }

    pub fn load_selection(selection: &TranscriptionSelection) -> Result<Self> {
        match validate(selection)?.runtime {
            ModelRuntime::Gguf(_) => Parakeet::load_selection(selection)
                .map(Box::new)
                .map(Self::Gguf),
            ModelRuntime::AppleSpeech => AppleSpeech::load(selection).map(Self::AppleSpeech),
            #[cfg(target_arch = "aarch64")]
            ModelRuntime::PhononCoreMl(_) => Phonon::load_selection(selection).map(Self::Phonon),
            // `validate` rejects Phonon on Intel Macs before this point.
            #[cfg(not(target_arch = "aarch64"))]
            ModelRuntime::PhononCoreMl(_) => {
                color_eyre::eyre::bail!("Phonon-2 Core ML requires a Mac with Apple silicon")
            }
        }
    }

    pub fn matches_selection(&self, selection: &TranscriptionSelection) -> bool {
        match self {
            Self::Gguf(model) => model.matches_selection(selection),
            Self::AppleSpeech(model) => model.matches_selection(selection),
            #[cfg(target_arch = "aarch64")]
            Self::Phonon(model) => model.matches_selection(selection),
        }
    }

    /// Transcribe normalized 16 kHz audio. Backend padding is applied once here,
    /// before GGUF chunking or any voice-control retranscription.
    pub fn transcribe(&mut self, samples: Vec<f32>) -> Result<String> {
        self.transcribe_with_protocol(samples, None)
    }

    pub fn transcribe_voice(
        &mut self,
        samples: Vec<f32>,
        protocol: &DictationProtocol,
    ) -> Result<String> {
        self.transcribe_with_protocol(samples, Some(protocol))
    }

    fn transcribe_with_protocol(
        &mut self,
        mut samples: Vec<f32>,
        protocol: Option<&DictationProtocol>,
    ) -> Result<String> {
        match self {
            Self::Gguf(model) => {
                prepare_gguf_samples(&mut samples, model.model_id());
                match protocol {
                    Some(protocol) => model.transcribe_voice(&samples, protocol),
                    None => model.transcribe(&samples),
                }
            }
            Self::AppleSpeech(model) => model.transcribe(&samples).map(|result| result.text),
            // Phonon receives the captured audio exactly: no GGUF padding.
            #[cfg(target_arch = "aarch64")]
            Self::Phonon(model) => match protocol {
                Some(protocol) => model.transcribe_voice(&samples, protocol),
                None => model.transcribe_text(&samples),
            },
        }
    }

    /// Transcribe one normalized segment with the same input preparation as
    /// plain dictation. The caller retains ownership of segment offsets.
    pub fn transcribe_segments(&mut self, mut samples: Vec<f32>) -> Result<Transcript> {
        match self {
            Self::Gguf(model) => {
                prepare_gguf_samples(&mut samples, model.model_id());
                model
                    .transcribe_segments(&samples)
                    .map(|result| Transcript {
                        text: result.text,
                        segments: result
                            .segments
                            .into_iter()
                            .map(|segment| TranscriptSegment {
                                start_ms: segment.t0_ms,
                                end_ms: segment.t1_ms,
                                text: segment.text,
                            })
                            .collect(),
                    })
            }
            Self::AppleSpeech(model) => model.transcribe(&samples),
            #[cfg(target_arch = "aarch64")]
            Self::Phonon(model) => model.transcribe_segments(&samples),
        }
    }
}

fn prepare_gguf_samples(samples: &mut Vec<f32>, model: Option<TranscriptionModelId>) {
    crate::dictation::pad_for_parakeet(samples);
    if model == Some(TranscriptionModelId::ParakeetUnifiedEnglish) {
        samples.resize(
            samples.len() + UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES,
            0.0,
        );
    }
}

impl WarmTranscriber {
    pub fn load() -> Result<Self> {
        Ok(Self {
            active: Some(Transcriber::load()?),
        })
    }

    pub fn activate(&mut self, selection: &TranscriptionSelection) -> Result<&mut Transcriber> {
        if !self
            .active
            .as_ref()
            .is_some_and(|model| model.matches_selection(selection))
        {
            let candidate = Transcriber::load_selection(selection)?;
            self.active = Some(candidate);
        }
        Ok(self
            .active
            .as_mut()
            .expect("activated transcriber must be available"))
    }
}

#[cfg(test)]
mod tests {
    use super::{Transcript, UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES, prepare_gguf_samples};
    use crate::transcription_models::TranscriptionModelId;

    #[test]
    fn apple_bridge_transcript_uses_the_canonical_wire_shape() {
        let transcript: Transcript = serde_json::from_str(
            r#"{"text":"hello","segments":[{"startMs":10,"endMs":40,"text":"hello"}]}"#,
        )
        .unwrap();

        assert_eq!(transcript.text, "hello");
        assert_eq!(transcript.segments.len(), 1);
        assert_eq!(transcript.segments[0].start_ms, 10);
        assert_eq!(transcript.segments[0].end_ms, 40);
    }

    /// Hex's catalogue, installer, activation and removal around real Phonon
    /// inference. Uses `HEX_APPLICATION_SUPPORT_DIR` with Phonon installed by
    /// `voice-control install-transcription-model phonon_2_coreml_en`, and
    /// the fixtures in `HEX_PHONON_FIXTURES`. Parakeet must NOT be installed
    /// there: it stands in for a failing candidate.
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "requires Apple silicon and an installed Phonon bundle"]
    fn phonon_activation_preserves_the_active_model_and_removal_unloads_cleanly() {
        use super::{Transcriber, WarmTranscriber};
        use crate::transcription_models::{
            TranscriptionSelection, definition, is_installed, is_verified, remove_installed,
        };

        let fixtures = std::path::PathBuf::from(std::env::var_os("HEX_PHONON_FIXTURES").unwrap());
        let mut reader = hound::WavReader::open(fixtures.join("1089-134686-0002.wav")).unwrap();
        let samples = reader
            .samples::<i16>()
            .map(|sample| f32::from(sample.unwrap()) / 32_768.0)
            .collect::<Vec<_>>();
        let expected = "After early nightfall the yellow lamps would light up here and there \
                        the squalid quarter of the brothels.";
        let phonon = TranscriptionSelection {
            model: TranscriptionModelId::Phonon2CoreMlEnglish,
            ..TranscriptionSelection::default()
        };
        let missing = TranscriptionSelection::default();
        let phonon_model = definition(phonon.model);
        assert!(is_verified(phonon_model));
        assert!(!is_installed(definition(missing.model), "en"));

        let mut warm = WarmTranscriber::default();
        assert_eq!(
            warm.activate(&phonon)
                .unwrap()
                .transcribe(samples.clone())
                .unwrap(),
            expected
        );
        // A failed candidate never replaces the working model.
        assert!(warm.activate(&missing).is_err());
        let active = warm.active.as_mut().unwrap();
        assert!(active.matches_selection(&phonon));
        assert_eq!(active.transcribe(samples.clone()).unwrap(), expected);
        let segments = active.transcribe_segments(samples.clone()).unwrap();
        assert_eq!(segments.text, expected);
        let duration_ms = samples.len().div_ceil(16) as i64;
        assert!(
            segments
                .segments
                .iter()
                .all(|segment| segment.end_ms <= duration_ms)
        );

        // Removal deletes the bundle and its compiled model; the engine that
        // still owns mapped files is dropped afterwards, then cannot reload.
        remove_installed(phonon_model).unwrap();
        assert!(!is_installed(phonon_model, "en"));
        assert!(!is_verified(phonon_model));
        drop(warm);
        let error = Transcriber::load_selection(&phonon)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("not installed"), "{error}");
    }

    #[test]
    fn unified_english_gets_trailing_silence_for_final_token_context() {
        let mut samples = vec![0.5; 32_000];

        prepare_gguf_samples(
            &mut samples,
            Some(TranscriptionModelId::ParakeetUnifiedEnglish),
        );

        assert_eq!(
            samples.len(),
            32_000 + UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES
        );
        assert!(samples[..32_000].iter().all(|sample| *sample == 0.5));
        assert!(samples[32_000..].iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn other_gguf_models_do_not_get_trailing_silence() {
        let mut samples = vec![0.5; 32_000];

        prepare_gguf_samples(&mut samples, Some(TranscriptionModelId::ParakeetV2));

        assert_eq!(samples.len(), 32_000);
    }

    #[test]
    fn short_unified_input_is_minimum_padded_before_trailing_context() {
        let mut samples = vec![0.5; 1_600];
        prepare_gguf_samples(
            &mut samples,
            Some(TranscriptionModelId::ParakeetUnifiedEnglish),
        );

        // The 200 ms context is additional to the 1.5 s minimum, not part of it.
        assert_eq!(
            samples.len(),
            24_000 + UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES
        );
        assert!(samples[..1_600].iter().all(|sample| *sample == 0.5));
        assert!(samples[1_600..].iter().all(|sample| *sample == 0.0));
    }
}
