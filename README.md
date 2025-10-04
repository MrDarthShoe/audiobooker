# Polish Audiobook Generator

Generate Polish audiobooks from text files using XTTS-v2 locally.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Default voice:**

```bash
python generate_audiobook.py input.txt output.mp3
```

**Choose a voice:**

```bash
python generate_audiobook.py input.txt output.mp3 --voice-name "Ana Florence"
```

**Clone a voice from WAV file:**

```bash
python generate_audiobook.py input.txt output.mp3 --voice-file voice_sample.wav
```

**List available voices:**

```bash
python generate_audiobook.py --list-speakers
```

## Notes

- First run downloads ~2GB XTTS-v2 model (cached in `~/.local/share/tts/`)
- GPU recommended but works on CPU (slower)
- Voice cloning needs 6+ second WAV file for best results
- Text files should be UTF-8 encoded

## Inputs

### text_samples

It contains `sp0jciec.txt` file which I am not author of. It is a funny ham-radio copypasta from [http://www.sp7pki.iq24.pl/default.asp?grupa=3538&temat=346125&nr_str=1](http://www.sp7pki.iq24.pl/default.asp?grupa=3538&temat=346125&nr_str=1). I am using it as a sample text for testing. All credits go to `SP5XMI` who is the original author.

## TODOs

- Make multilang (not needed for me right now)
