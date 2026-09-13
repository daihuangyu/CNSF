#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from track_mt3.config_merge import load_merged_config
from track_mt3.training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Track-MT3")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--resume")
    arguments = parser.parse_args()
    trainer = Trainer(load_merged_config(arguments.config, *arguments.overlay))
    if arguments.resume:
        trainer.restore(arguments.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
