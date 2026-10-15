#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
support="${HEX_APPLICATION_SUPPORT_DIR:-$HOME/Library/Application Support/voice-control}"
discovery="$support/local-api.json"
model=${HEX_SMOKE_MODEL:-parakeet_unified_en}
generated="${TMPDIR:-/tmp}/hex-transcription-smoke.wav"
audio=${HEX_SMOKE_AUDIO:-$generated}
expect=${HEX_SMOKE_EXPECT:-}
events="${TMPDIR:-/tmp}/hex-transcription-prepare.ndjson"
service_pid=

cleanup() {
  if [ -n "$service_pid" ] && kill -0 "$service_pid" 2>/dev/null; then
    kill -INT "$service_pid" 2>/dev/null || true
    shutdown_attempts=100
    while kill -0 "$service_pid" 2>/dev/null && [ "$shutdown_attempts" -gt 0 ]; do
      shutdown_attempts=$((shutdown_attempts - 1))
      sleep 0.05
    done
    if kill -0 "$service_pid" 2>/dev/null; then
      kill -TERM "$service_pid" 2>/dev/null || true
      sleep 0.1
    fi
    if kill -0 "$service_pid" 2>/dev/null; then
      kill -KILL "$service_pid" 2>/dev/null || true
    fi
  fi
  if [ -n "$service_pid" ]; then
    wait "$service_pid" 2>/dev/null || true
  fi
  rm -f "$generated" "$generated.aiff" "$events"
}
trap cleanup EXIT INT TERM

if [ -e "$discovery" ]; then
  echo "Refusing to replace an existing local API discovery file: $discovery" >&2
  exit 1
fi

binary=${HEX_SMOKE_BINARY:-}
if [ -z "$binary" ]; then
  cargo build --manifest-path "$root/Cargo.toml"
  binary="$root/target/debug/voice-control"
fi
"$binary" service >/dev/null 2>&1 &
service_pid=$!

attempts=100
while [ ! -f "$discovery" ]; do
  attempts=$((attempts - 1))
  if [ "$attempts" -eq 0 ]; then
    echo "Timed out waiting for local API discovery." >&2
    exit 1
  fi
  sleep 0.05
done
port=$(jq -r .port "$discovery")
token=$(jq -r .token "$discovery")

curl --fail --silent --show-error --no-buffer \
  -X POST \
  -H "Authorization: Bearer $token" \
  "http://127.0.0.1:$port/models/$model/prepare" > "$events"
if ! grep -Fq '"type":"ok"' "$events"; then
  echo "Model preparation did not finish. Events:" >&2
  tail -n 20 "$events" >&2
  tail -n 40 "$support/logs/process.log" >&2 2>/dev/null || true
  exit 1
fi

if [ -z "${HEX_SMOKE_AUDIO:-}" ]; then
  /usr/bin/say "Hello from the local HEX transcription service." -o "$generated.aiff"
  /usr/bin/afconvert -f WAVE -d LEI16@16000 "$generated.aiff" "$generated"
  rm -f "$generated.aiff"
  # say occasionally writes a near-empty file on headless machines; that is
  # not a transcription result, so refuse it instead of reporting empty text.
  if [ "$(wc -c < "$generated")" -lt 16000 ]; then
    echo "say produced no usable speech; set HEX_SMOKE_AUDIO to a WAV file." >&2
    exit 1
  fi
fi

result=$(curl --fail --silent --show-error \
  -X POST \
  -H "Authorization: Bearer $token" \
  -H "Content-Type: audio/wav" \
  --data-binary "@$audio" \
  "http://127.0.0.1:$port/transcriptions?model=$model&language=en")
printf '%s\n' "$result" | jq .
if ! printf '%s\n' "$result" | jq -e '.transcript | length > 0' >/dev/null; then
  echo "Transcription returned no text. Service log:" >&2
  tail -n 40 "$support/logs/process.log" >&2 2>/dev/null || true
  exit 1
fi
if [ -n "$expect" ] && ! printf '%s\n' "$result" | jq -e --arg expect "$expect" \
  '.transcript | ascii_downcase | contains($expect | ascii_downcase)' >/dev/null; then
  echo "Transcription does not contain: $expect" >&2
  exit 1
fi
