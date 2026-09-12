import argparse
import logging
from datetime import date

from .pipelines import feature_pipeline, inference_pipeline, training_pipeline
from .settings import Settings


def iso_date(value):
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use YYYY-MM-DD") from exc


def main():
    parser = argparse.ArgumentParser(
        description="Minimal ICE Hamburg-Berlin delay FTI project (historical replay)"
    )
    commands = parser.add_subparsers(dest="pipeline", required=True)
    features = commands.add_parser(
        "features", help="Import archived rail data and insert Hopsworks features"
    )
    features.add_argument("--start", type=iso_date, required=True)
    features.add_argument("--end", type=iso_date, required=True)
    train = commands.add_parser("train", help="Train/evaluate and register a model")
    train.add_argument(
        "--cutoff", type=iso_date, required=True, help="First date reserved for inference"
    )
    infer = commands.add_parser(
        "infer", help="Predict held-out journeys using simulated current departure observations"
    )
    infer.add_argument("--day", type=iso_date, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.load()
    if args.pipeline == "features":
        feature_pipeline(settings, args.start, args.end)
    elif args.pipeline == "train":
        training_pipeline(settings, args.cutoff)
    else:
        inference_pipeline(settings, args.day)


if __name__ == "__main__":
    main()
