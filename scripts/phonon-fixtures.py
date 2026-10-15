#!/usr/bin/env python3
"""Prepare redistribution-safe speech fixtures for Phonon tests and benchmarks.

Downloads the three LibriSpeech test-clean clips (CC-BY-4.0) published with
FermionResearch/Phonon-2-CoreML at the pinned revision, plus whisper.cpp's
public-domain JFK sample, verifies every SHA-256, converts the clips to 16 kHz
mono PCM16 WAV with macOS `afconvert`, and writes:

  OUT/expected.json         clip id -> reference text (native tests)
  OUT/<id>.wav              the clips
  OUT/long-60s.wav, long-120s.wav   whole clips joined by 300 ms pauses
  OUT/manifest.json         a `benchmark-transcription` corpus of all of them
  OUT/protocol/*.wav        `say` speech with voice-delimited control phrases
                            (synthetic: checks control handling, not accuracy)

Nothing here is committed; run it into a scratch directory.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import urllib.request
import wave

REVISION = "1143812fd4236522232a8c33e3115f0577e3cd7d"
BASE = f"https://huggingface.co/FermionResearch/Phonon-2-CoreML/resolve/{REVISION}"
CLIPS = {
    "1089-134686-0000": "30885601173f96b0d8ddd020dc959b055c6c1582b85a33e3fcab8c4b08ed94c2",
    "1089-134686-0002": "66277a3fa3df407261dc2a3ce685a7ceef19999ab0c10531bee5257cb64cb59d",
    "1089-134686-0011": "c65e16be3ef92c11a31a9c2c4c570542eb2ed66153150a7c4cca00c01b30a4e2",
}
EXPECTED_SHA256 = "695c112e2bbf6aa04b67e95a631e269d9643f7bb840f67313ab471cf7d8b242c"
JFK_URL = (
    "https://raw.githubusercontent.com/ggml-org/whisper.cpp/"
    "b0a11594aec50892a02cd8d129eee2dfe93a8bb8/samples/jfk.wav"
)
JFK_SHA256 = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"
JFK_TEXT = (
    "And so my fellow Americans, ask not what your country can do for you, "
    "ask what you can do for your country."
)
PROTOCOL = {
    "period-stop": "Say, I don't understand this. Say stop.",
    "question-stop": "Say, is this working? Say stop.",
    "exclamation-send": "Say, ship it! Say send.",
    "comma-stop": "Say, meet me at five, say stop.",
}
RATE = 16_000
GAP = b"\x00\x00" * (RATE * 300 // 1000)


def fetch(url: str, sha256: str, path: pathlib.Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        sys.exit(f"checksum mismatch for {url}: {digest}")
    path.write_bytes(data)


def frames(path: pathlib.Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (RATE, 1, 2):
            sys.exit(f"{path} is not 16 kHz mono PCM16")
        return source.readframes(source.getnframes())


def write(path: pathlib.Path, data: bytes) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(RATE)
        output.writeframes(data)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: phonon-fixtures.py OUTPUT_DIRECTORY")
    output = pathlib.Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        temporary = pathlib.Path(temporary)
        expected_path = temporary / "expected_texts.json"
        fetch(f"{BASE}/ci/expected_texts.json", EXPECTED_SHA256, expected_path)
        expected = json.loads(expected_path.read_text())
        for clip, sha256 in CLIPS.items():
            flac = temporary / f"{clip}.flac"
            fetch(f"{BASE}/ci/clips/{clip}.flac", sha256, flac)
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                 str(flac), str(output / f"{clip}.wav")],
                check=True,
            )
        fetch(JFK_URL, JFK_SHA256, output / "jfk.wav")

    (output / "expected.json").write_text(json.dumps({clip: expected[clip] for clip in CLIPS}, indent=1))
    corpus = [{"id": clip, "audio": f"{clip}.wav", "expected": expected[clip]} for clip in CLIPS]
    corpus.append({"id": "jfk", "audio": "jfk.wav", "expected": JFK_TEXT})
    for seconds in (60, 120):
        audio, words = b"", []
        while len(audio) < seconds * RATE * 2:
            for clip in CLIPS:
                audio += frames(output / f"{clip}.wav") + GAP
                words.append(expected[clip])
                if len(audio) >= seconds * RATE * 2:
                    break
        write(output / f"long-{seconds}s.wav", audio)
        corpus.append({"id": f"long-{seconds}s", "audio": f"long-{seconds}s.wav", "expected": " ".join(words)})
    (output / "manifest.json").write_text(json.dumps({"clips": corpus}, indent=1))
    protocol = output / "protocol"
    protocol.mkdir(exist_ok=True)
    for name, prompt in PROTOCOL.items():
        aiff = protocol / f"{name}.aiff"
        subprocess.run(["say", "-v", "Samantha", "-r", "165", "-o", str(aiff), prompt], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(protocol / f"{name}.wav")],
            check=True,
        )
        aiff.unlink()
    print(f"wrote {len(corpus)} fixtures to {output}")


if __name__ == "__main__":
    main()
