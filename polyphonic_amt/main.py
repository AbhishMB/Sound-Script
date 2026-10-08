import argparse
import os

from amt.pipeline import TranscriptionPipeline


def main():
    ap = argparse.ArgumentParser(
        description="Polyphonic transcription: Demucs separation + harmonic-gated torchcrepe pitch tracking"
    )
    ap.add_argument("--input", "-i", required=True, help="audio file (wav/flac/mp3)")
    ap.add_argument("--output", "-o", default="output.mid", help="multi-track MIDI file to write")
    args = ap.parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)
    pm, results = TranscriptionPipeline().transcribe(args.input, args.output)
    for inst in pm.instruments:
        print(f"{inst.name:<8} {len(inst.notes):>4} notes")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
