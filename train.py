import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output_dir", default="checkpoints")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "ready",
        "config": args.config,
        "message": "Training entry point prepared for the DAViT pneumonia detection pipeline.",
    }
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
