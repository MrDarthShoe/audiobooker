#!/usr/bin/env python3

import torch
import torch.serialization
from TTS.api import TTS
from pydub import AudioSegment
import sys
import os
import argparse

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    torch.serialization.add_safe_globals([XttsConfig])
except Exception:
    pass

FEMALE_SPEAKERS = [
    "Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema",
    "Alison Dietlinde", "Ana Florence", "Annmarie Nele", "Asya Anara",
    "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen",
    "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie"
]

MALE_SPEAKERS = [
    "Andrew Chipper", "Badr Odhiambo", "Dionisio Schuyler", "Royston Min",
    "Viktor Eka", "Abrahan Mack", "Adde Michal", "Baldur Sanjin",
    "Craig Gutsy", "Damien Black", "Gilberto Mathias", "Ilkin Urbano",
    "Kazuhiko Atallah", "Ludvig Milivoj", "Suad Qasim", "Torcull Diarmuid",
    "Viktor Menelaos", "Zacharie Aimilios"
]

AVAILABLE_SPEAKERS = FEMALE_SPEAKERS + MALE_SPEAKERS
DEFAULT_SPEAKER = "Damien Black"


def generate_audiobook(input_txt, output_mp3, voice_file=None, voice_name=None):
    if not os.path.exists(input_txt):
        raise FileNotFoundError(f"Input file not found: {input_txt}")

    with open(input_txt, 'r', encoding='utf-8') as f:
        text = f.read().strip()

    if not text:
        raise ValueError("Input file is empty")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("WARNING: Running on CPU. This will be slower than GPU.")

    print("Loading XTTS-v2 model... (first run will download ~2GB)")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    temp_wav = output_mp3.replace('.mp3', '_temp.wav')

    print(f"Generating Polish audiobook...")
    print(f"Input file: {input_txt}")
    print(f"Text length: {len(text)} characters")

    if voice_file:
        if not os.path.exists(voice_file):
            raise FileNotFoundError(f"Voice file not found: {voice_file}")

        print(f"Voice cloning from: {voice_file}")
        tts.tts_to_file(text=text, speaker_wav=voice_file, language="pl", file_path=temp_wav)
    else:
        if voice_name and voice_name not in AVAILABLE_SPEAKERS:
            print(f"Warning: Speaker '{voice_name}' not found. Using default: {DEFAULT_SPEAKER}")
            voice_name = DEFAULT_SPEAKER
        elif not voice_name:
            voice_name = DEFAULT_SPEAKER

        print(f"Using speaker: {voice_name}")
        tts.tts_to_file(text=text, speaker=voice_name, language="pl", file_path=temp_wav)

    print(f"Converting to MP3: {output_mp3}")

    audio = AudioSegment.from_wav(temp_wav)
    audio.export(output_mp3, format="mp3")
    os.remove(temp_wav)

    print(f"✓ Audiobook generated successfully: {output_mp3}")
    print(f"Duration: {len(audio) / 1000:.2f} seconds")


def list_speakers():
    print("Available speakers:")
    print("\nFemale voices:")
    for voice in FEMALE_SPEAKERS:
        print(f"  - {voice}")

    print("\nMale voices:")
    for voice in MALE_SPEAKERS:
        print(f"  - {voice}")


def main():
    parser = argparse.ArgumentParser(
        description="Polish Audiobook Generator using XTTS-v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_audiobook.py input.txt output.mp3
  python generate_audiobook.py input.txt output.mp3 --voice-name "Ana Florence"
  python generate_audiobook.py input.txt output.mp3 --voice-file speaker.wav
  python generate_audiobook.py --list-speakers
        """
    )

    parser.add_argument('input_txt', nargs='?', help='Input text file (.txt)')
    parser.add_argument('output_mp3', nargs='?', help='Output audio file (.mp3)')
    parser.add_argument('--voice-file', help='WAV file for voice cloning (6+ seconds recommended)')
    parser.add_argument('--voice-name', help='Predefined speaker name')
    parser.add_argument('--list-speakers', action='store_true', help='List all available predefined voices')

    args = parser.parse_args()

    if args.list_speakers:
        list_speakers()
        sys.exit(0)

    if not args.input_txt or not args.output_mp3:
        parser.print_help()
        sys.exit(1)

    if args.voice_file and args.voice_name:
        print("Error: Cannot use both --voice-file and --voice-name. Choose one.")
        sys.exit(1)

    generate_audiobook(args.input_txt, args.output_mp3, voice_file=args.voice_file, voice_name=args.voice_name)


if __name__ == "__main__":
    main()
