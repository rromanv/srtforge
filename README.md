# srtforge

Generate, translate, and burn `.srt` subtitles locally on Apple Silicon.

`srtforge` extracts audio with ffmpeg, transcribes it with local MLX Whisper,
optionally re-cues the captions for readability, and can translate subtitles
with a local MLX language model. No cloud APIs or keys are required.

## Features

- Local Whisper transcription through `mlx-whisper`
- Sentence-aware subtitle cueing from word timestamps
- Standard SRT output
- Optional local LLM translation with timestamp preservation
- Optional hard-subtitle burn-in through ffmpeg/libass
- Reference transcript support to correct aligned word substitutions while keeping Whisper timing
- Installable Python CLI: `srtforge`

## Requirements

- macOS on Apple Silicon
- Python 3.10, 3.11, or 3.12
- `ffmpeg` on your `PATH`
- Enough disk/RAM for the models you choose

The default transcription model is `mlx-community/whisper-large-v3-turbo`.
The default translation model is `mlx-community/gemma-4-26b-a4b-it-4bit`.

First use downloads models from Hugging Face into the local cache. Later runs
can use the cached models.

By default, Hugging Face model files are stored outside this project in your
user cache, normally under:

```text
~/.cache/huggingface/hub
```

You can move that cache by setting Hugging Face cache environment variables such
as `HF_HOME` or `HUGGINGFACE_HUB_CACHE`. `srtforge` does not store downloaded
models in the repository or next to your videos. Temporary extracted WAV files
are created in the system temp directory and deleted after each run.

## Install

From PyPI:

```bash
pipx install --python python3.11 srtforge
```

`srtforge` currently supports Python 3.10-3.12. If your default Python is newer
than that, such as Python 3.14, pass a supported interpreter explicitly with
`--python`.

From GitHub:

```bash
pipx install --python python3.11 git+https://github.com/rromanv/srtforge.git
```

For local development:

```bash
git clone https://github.com/rromanv/srtforge.git
cd srtforge
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or into a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install srtforge
```

## Troubleshooting

### pipx uses Python 3.13 or newer

If installation fails because your default Python is outside the supported
range, install a supported Python and tell `pipx` to use it:

```bash
python3.11 --version
pipx install --python python3.11 git+https://github.com/rromanv/srtforge.git
```

If `python3.11` is not installed, install Python 3.11 or 3.12 first, then rerun
the `pipx install --python ...` command.

## Usage

Generate subtitles next to a video:

```bash
srtforge video.mp4
```

Write to a custom path:

```bash
srtforge video.mp4 -o captions.srt
```

Force the source language:

```bash
srtforge video.mp4 -l en
```

Use another Whisper model:

```bash
srtforge video.mp4 -m mlx-community/whisper-small
```

Disable sentence-aware re-cueing:

```bash
srtforge video.mp4 --no-resegment
```

Use a reference transcript to improve accuracy:

```bash
srtforge video.mp4 --transcript script.txt
srtforge video.mp4 --transcript notes.md
```

Timing still comes from the audio. The transcript is used as a short Whisper
prompt and for conservative post-correction of aligned word substitutions, such
as spelling/name fixes. It does not add missing words or remove extra words. If
the transcript barely matches the audio, correction is skipped automatically.

Tune subtitle readability:

```bash
srtforge video.mp4 --max-line-length 37 --max-lines 2 --reading-speed 15
```

Translate subtitles:

```bash
srtforge video.mp4 -t Spanish
srtforge video.mp4 -t "Brazilian Portuguese"
srtforge video.mp4 -t ja --translate-model mlx-community/Qwen3.5-9B-OptiQ-4bit
```

Burn subtitles into a video:

```bash
srtforge merge video.mp4 video.srt -o final.mp4
srtforge merge video.mp4 video.es.srt --crf 16 --font-size 26
```

Run:

```bash
srtforge --help
srtforge merge --help
```

## Readability

By default, `srtforge` asks Whisper for word-level timestamps and rebuilds cues
so they are easier to read:

- cues prefer sentence boundaries
- lines are wrapped to two lines of 42 characters by default
- long sentences are split across cue boundaries
- cues are paced around 17 characters per second
- cues are adjusted to avoid overlap

Translated subtitles are re-fitted after translation because translated text can
be longer or shorter than the source.

## ffmpeg Notes

Audio extraction requires `ffmpeg`.

Burn-in uses ffmpeg's `subtitles` filter, which requires libass. If your ffmpeg
does not include it, install a full build such as:

```bash
nb install ffmpeg-full
```

## Project Layout

```text
src/srtforge/
  audio.py       # ffmpeg audio extraction
  transcribe.py  # MLX Whisper transcription
  segment.py     # sentence-aware re-cueing, wrapping, pacing
  translate.py   # context-aware local-LLM translation
  transcript.py  # reference transcript reading + correction
  merge.py       # burn subtitles into video with ffmpeg/libass
  srt.py         # SRT rendering
  cli.py         # argparse CLI
tests/
  test_cli.py
  test_merge.py
  test_segment.py
  test_srt.py
  test_transcript.py
```

## Development

Run the test suite:

```bash
python -m unittest discover -s tests
```

Build release artifacts:

```bash
python -m build
python -m twine check dist/*
```

Publish to PyPI manually:

```bash
python -m twine upload dist/*
```

The repository also includes a GitHub Actions workflow for publishing to PyPI
when a release is created. It uses PyPI Trusted Publishing, so no PyPI API token
is needed in GitHub.

Before creating a release that should publish to PyPI, configure a pending
publisher in your PyPI account:

```text
PyPI project name: srtforge
Owner: rromanv
Repository name: srtforge
Workflow filename: publish.yml
Environment name: pypi
```

PyPI docs:

- Creating a new project with a pending publisher:
  https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/
- Publishing with a trusted publisher:
  https://docs.pypi.org/trusted-publishers/using-a-publisher/

## License

MIT
