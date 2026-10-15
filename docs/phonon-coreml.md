# Phonon-2 Core ML backend

Phonon-2 Core ML is an opt-in English dictation model for Apple silicon Macs.
It runs Fermion Research's Core ML encoder (Neural Engine where available) and
C greedy-TDT decoder in-process, through the same queue, cancellation,
processing, history and paste path as the GGUF models. The default model is
unchanged.

## Architecture

```ts
Settings / menu bar / local API -> select phonon_2_coreml_en
  -> transcription_models: ModelRuntime::PhononCoreMl(bundle)
     -> model_bundle::install           // staged, per-file SHA-256, atomic publish
  -> transcription_preparation -> Transcriber::prepare_selection
     -> phonon::Phonon::prepare         // requires a verified receipt; never downloads
        -> hex_phonon_create            // src/phonon.swift
           -> compile <bundle>/Phonon-2.mlmodelc once (atomic rename)
           -> load every encoder function once, prewarm, release
  -> worker: Phonon::load_selection     // warm session, lazy functions
  -> dictation job (selection snapshotted per job)
     -> Transcriber::transcribe         // captured 16 kHz audio, no GGUF padding
        -> hex_phonon_transcribe -> PhononCoreML.Transcriber -> canonical JSON
  -> existing cancellation fence, processing, history and paste
```

- `src/model_bundle.rs`: pinned multi-file bundle install, verification,
  receipt and removal. Downloads go to `<models>/<bundle>.partial/`, resume with
  `curl --continue-at -`, and are hashed once each. Only a complete, verified
  tree is renamed into place; an invalid published bundle is replaced only after
  its replacement verifies. Symlinks and path traversal are refused.
- `src/phonon.swift`: the C ABI (`hex_phonon_create`, `_transcribe`, `_release`,
  `_free_string`). It is compiled in the same Swift module as the pinned
  PhononCoreML sources, so no separate Swift package or dynamic library ships.
- `src/phonon.rs`: the Rust owner. One session per `Phonon`, `&mut self` for
  inference, released on drop. Non-finite samples are rejected before the bridge.
  Voice-delimited dictation keeps Phonon's full transcript and lets the worker
  strip the trailing control phrase as text. Unlike the GGUF path it does not
  cut the audio at the phrase's word timing and retranscribe: on real Phonon
  output that cut kept the onset of "say" and produced an extra word
  ("I don't understand this, you").
- `build.rs`: on `aarch64-apple-darwin` only, fetches the pinned commit (or uses
  `HEX_PHONON_COREML_SOURCE`), checks the decoder library's SHA-256, compiles
  `src/phonon.swift` plus `Sources/PhononCoreML/*.swift` into one object, and
  links the universal static `libPhononTDT.a` plus CoreML, Accelerate and
  AVFoundation.

Intel Macs and Linux compile none of the bridge. The catalogue entry exists on
every platform for persisted-selection compatibility, but `available()` is false
there: it is hidden from pickers, the local API catalogue and preparation, and
`validate` rejects it. On macOS a persisted unavailable selection is reset to
the default by `migrate_disabled_transcription_model`; Linux settings refuse to
load it, as they already do for `apple_speech`.

## Pins

| Item | Pin |
| --- | --- |
| Swift package | `fermionresearch/phonon-coreml` 1.1.2, commit `bf59e574639b060e30ee1a3f9c0e003fed5744c8` |
| Decoder library | `libPhononTDT.a` SHA-256 `22fe6f7b…62930e5` (universal arm64/x86_64, libSystem only) |
| Model | `FermionResearch/Phonon-2-CoreML` revision `1143812fd4236522232a8c33e3115f0577e3cd7d` |
| Bundle | 8 files, 345,023,465 bytes, listed with SHA-256 in `PHONON_2_COREML_BUNDLE` |

The GitHub tag and the `phonon-coreml-src.tgz` published in the model repository
at the pinned revision have identical `Sources/` and `Binaries/`.

### Updating

1. Pick a new model revision. Take each file's size and LFS SHA-256 from
   `https://huggingface.co/api/models/FermionResearch/Phonon-2-CoreML/revision/<rev>?blobs=true`
   and cross-check `SHA256SUMS`. Update `PHONON_2_COREML_BUNDLE`, including its
   `directory` suffix so the new bundle and its compiled cache never reuse the
   old directory.
2. For a new package release, update `PHONON_COREML_COMMIT` and
   `PHONON_TDT_SHA256` in `build.rs`, then diff `Transcriber.swift` and
   `Windower.swift` for planning or encoder-selection changes. Re-run the native
   boundary test: if upstream fixes the selection rule, its expected failure list
   changes and `hexPhononSingleShotMaxSamples` can return to the package default.
3. Run the `Check macOS Phonon` workflow and compare transcripts and benchmarks.

## The 35-second boundary

PhononCoreML 1.1.2 (`Transcriber.plan`) keeps audio of up to
`Int(35 * 16000) = 560,000` samples as one window. `encodeFeatures` then asks for
an encoder function of at least `frames * 0.01 + 0.02` seconds, with
`frames = samples / 160`. For 559,680 to 560,000 samples that is 35.00000000000001
to 35.02 s, larger than the biggest (35 s) function, so the call throws
"window of … s exceeds the largest encoder function". 560,001 samples take the
long-audio path and succeed. This is the reference Python runner's defect,
present natively too.

Hex passes `singleShotMaxSeconds = 559,520 / 16,000` (34.97 s), the largest
hop-aligned length whose selection still fits. Longer audio uses the package's
own pause-cut 15 s windows with overlap stitching, so no samples are dropped and
recording length is not capped. `upstream_35_second_limit_fails_and_hex_limit_covers_every_sample`
creates one session with the package default and one with Hex's limit and
checks both behaviours on real inference.

## Inputs and outputs

- Exactly the normalised mono 16 kHz capture: no resampling, minimum-length or
  trailing-silence padding.
- Fewer than 320 samples (two feature frames) returns an empty transcript
  without calling the engine. Below 160 samples the package's front end produces
  no frames at all.
- Silence (peak below 1e-4) is skipped by the package and returns empty text.
- NaN or infinity is an error; the session remains usable.
- Text is the package's full text. Words and windows become millisecond spans,
  clamped to the clip and checked to be ordered.

## Lifecycle

- Preparation (`Transcriber::prepare_selection`, used by Settings, the menu bar,
  the local API's prepare route and `install-transcription-model`) compiles the `.mlpackage` once
  into `<bundle>/Phonon-2.mlmodelc` (a stable path inside Hex's models
  directory, deleted with the model), loads every encoder function once and
  prewarms, then releases that session. The first load of each function on a
  Mac compiles it for the Neural Engine; doing it here keeps that out of the
  first dictation of each length.
- The warm dictation session uses the package's lazy default: its 15 s
  function is resident and the others load on first use, reusing the
  system's compiled programs. This keeps resident memory lower (see below).
- `WarmTranscriber::activate` loads the candidate before replacing the active
  model; a failed or cancelled preparation leaves the previous model and saved
  selection untouched.
- A synchronous Core ML call cannot be interrupted. Cancelling a dictation marks
  the job; the worker drops the result after inference, so it is never pasted,
  saved to history or made the last result.
- Removal takes the download lock, deletes the bundle, compiled model, receipt
  and any staging. The engine is never freed while inference runs because the
  worker owns it exclusively.

## Building and testing

Requirements: Apple silicon, macOS 15+, Xcode with the macOS 26 SDK (already
required by the Apple Speech bridge), git and network access on first build (or
`HEX_PHONON_COREML_SOURCE=/path/to/phonon-coreml` at the pinned commit).

```sh
cargo build --release
python3 scripts/phonon-fixtures.py /tmp/phonon-fixtures
export HEX_APPLICATION_SUPPORT_DIR=/tmp/hex-phonon       # optional isolated data
target/release/voice-control install-transcription-model phonon_2_coreml_en
HEX_PHONON_BUNDLE=$HEX_APPLICATION_SUPPORT_DIR/models/phonon-2-coreml-1143812f \
HEX_PHONON_FIXTURES=/tmp/phonon-fixtures \
  cargo test --release --bin voice-control -- phonon::native transcription::tests::phonon \
  --ignored --nocapture --test-threads=1
target/release/voice-control install-transcription-model phonon_2_coreml_en  # reinstall
target/release/voice-control benchmark-transcription /tmp/phonon-fixtures/manifest.json \
  --backend selection --model-id phonon_2_coreml_en
target/release/voice-control benchmark-transcription /tmp/phonon-fixtures/manifest.json \
  --backend selection --model-id parakeet_unified_en
```

`transcription::tests::phonon…` deletes the installed bundle at the end, so
reinstall before benchmarking. It also needs the default Parakeet model to be
absent from that support directory, because it uses it as a failing candidate.

The `selection` benchmark backend runs any installed catalogue model through
Hex's dictation `Transcriber`, including each backend's own input preparation,
and reports load, first inference, median/p95 per clip, WER and peak RSS. It does
not include capture, processing or paste.

The fixtures are three LibriSpeech test-clean clips (CC-BY-4.0) published with
the model, whisper.cpp's public-domain JFK sample, and 60/120-second
concatenations of the LibriSpeech clips. Nothing is committed.

`.github/workflows/check-macos-phonon.yml` runs all of this on a hosted
`macos-26` arm64 runner. Hosted runners have no Neural Engine, so Core ML runs
the encoder on the CPU/GPU there: texts are meaningful, timings are not
representative of a real Mac.

## Observed on a hosted runner (15 October 2026)

`Check macOS Phonon` run 38660457347 on GitHub's `macos-26` arm64 runner (Apple M1
virtual machine, 7 GB, macOS 26.6.2, Xcode 26.6, no Neural Engine). Real Core ML,
CPU path; timings are not representative of a physical Mac.

- Build, static linkage audit (no Phonon or build-local dynamic libraries; only
  system frameworks and `/usr/lib/swift`), Hex install, 9 real-inference tests,
  511 unit tests and strict Clippy all passed.
- Install through Hex: 9.8 s download and verify; first load (Core ML compile plus
  all functions plus prewarm) 313 s on this VM; a later process loads in 5.5 s.
  On disk: 646 MB (345 MB bundle plus 317 MB compiled model).
- Exact reference text for the three LibriSpeech clips; 0 word errors on 60 s and
  120 s natural speech with and without pauses (5-11 windows), and no lost or
  duplicated words at seams. One cosmetic seam effect: a window can start with a
  capital letter mid-sentence ("office, His place").
- The 35 s defect reproduces natively: with the package default, 559,680, 559,840
  and 560,000 samples fail with "window of … s exceeds the largest encoder
  function"; 560,001 succeeds. With Hex's limit all five lengths transcribe all
  99 words.
- Same fixtures (6 clips, 223 s, rewriting off) through Hex's `Transcriber`:

  | | Phonon-2 Core ML | Parakeet Unified English (default) |
  | --- | --- | --- |
  | Load | 5.0 s | 1.6 s |
  | Median corpus pass | 19.1 s (11.7x realtime) | 45.9 s (4.9x realtime) |
  | Median / p95 clip | 1.9 s / 10.6 s | 2.8 s / 26.0 s |
  | WER | 0 / 617 words | 0 / 617 words |
  | Peak RSS | 5.6 GB | 1.25 GB |

  An earlier run of the same job measured 17.4x and 8.1x: VM timings vary.
- Memory and first calls: on this CPU path, keeping every function resident
  added about 3.2 GB against about 1.3-1.6 GB for the lazy default, but only
  saved about 1 s on the first call of each length: first calls after a load
  took 6-17 s either way, against about 0.6 s warm. Prewarming every function
  at load time tripled load time without helping. Hex therefore prepares all
  functions once and keeps the warm session lazy. On a real Mac the questions
  are resident memory on the Neural Engine path and how slow the first call
  of each length is after preparation.

- The 6.6 s LibriSpeech clip, the only fixture that uses the 10 s encoder
  function, was consistently 2-10x slower per second of audio than longer clips
  on the 15 s function, in every run. Whether the smaller functions are slower
  on the Neural Engine too, or only on this CPU fallback, needs a real Mac.
- The local API smoke test transcribes the JFK fixture. macOS `say` on the
  runner occasionally wrote a near-empty file (5 ms of audio), for which Phonon
  correctly returned no text; the scripts now reject such output.

## Known limitations

- English only; no language detection, translation, recognition hints or
  streaming partials.
- The prebuilt decoder library is a binary from the package release; its source
  is not published.
- Core ML inference cannot be pre-empted mid-call.
- The end-to-end hotkey -> paste path, real Neural Engine timings, memory
  under repeated switching, and signed/notarised packaging need a physical Mac.
