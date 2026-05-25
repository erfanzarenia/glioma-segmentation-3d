"""CLI entry point: preprocess the raw BraTS 2021 dataset.

Example
-------
    python scripts/preprocess.py \
        data.raw_dir=data/raw/BraTS2021 \
        data.processed_dir=data/processed
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from glioma_seg.data import Preprocessor
from glioma_seg.utils.logging import get_logger
from glioma_seg.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log = get_logger()
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

    pre = Preprocessor(
        raw_dir=cfg.data.raw_dir,
        processed_dir=cfg.data.processed_dir,
        split_ratio=tuple(cfg.data.split_ratio),
    )

    log.info("Computing metadata (crop bounds + intensity stats)…")
    crop_bounds, stats = pre.compute_metadata()
    log.info("Crop bounds: %s", crop_bounds)
    log.info("Intensity stats: %s", stats)

    log.info("Running preprocessing…")
    pre.run(crop_bounds, stats)
    log.info("Done.")


if __name__ == "__main__":
    main()
