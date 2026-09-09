"""Patch the official CycleGRPO trainer to honor val_freq=-1 after training."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = """        if self.val_reward_fn is not None:\n            if (\n                val_metrics is None\n                or self.config.trainer.val_freq <= 0\n                or self.global_step % self.config.trainer.val_freq != 0\n            ):\n                val_metrics = self._validate()\n                self.logger.log(data=val_metrics, step=self.global_step)\n\n            print(f\"Final validation metrics:\\n{convert_dict_to_str(unflatten_dict(val_metrics))}\")\n"""
NEW = """        if self.val_reward_fn is not None and self.config.trainer.val_freq > 0:\n            if (\n                val_metrics is None\n                or self.global_step % self.config.trainer.val_freq != 0\n            ):\n                val_metrics = self._validate()\n                self.logger.log(data=val_metrics, step=self.global_step)\n\n            print(f\"Final validation metrics:\\n{convert_dict_to_str(unflatten_dict(val_metrics))}\")\n"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "official_repo",
        type=Path,
        help="Path to the unmodified official CycleGRPO repository.",
    )
    args = parser.parse_args()
    trainer_path = args.official_repo / "verl/trainer/ray_trainer.py"
    if not trainer_path.is_file():
        raise FileNotFoundError(f"Official trainer not found: {trainer_path}")

    source = trainer_path.read_text(encoding="utf-8")
    if NEW in source:
        print(f"Already patched: {trainer_path}")
        return
    if OLD not in source:
        raise RuntimeError(
            "Expected official final-validation block was not found; "
            "refusing to edit an unknown trainer version."
        )
    trainer_path.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"Patched final validation guard: {trainer_path}")


if __name__ == "__main__":
    main()
