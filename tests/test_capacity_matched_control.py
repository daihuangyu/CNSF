from pathlib import Path

from track_mt3.config_merge import load_merged_config
from track_mt3.models import TrackMT3


ROOT = Path(__file__).resolve().parents[1]


def test_capacity_matched_control_is_exact_and_parameter_matched():
    config = load_merged_config(
        ROOT / "configs/paper.yaml",
        ROOT / "configs/training/track_mt3.yaml",
        ROOT / "configs/training/track_mt3_cm_exact12k.yaml",
    )
    assert config.model.encoder_layers == 3
    assert config.model.decoder_layers == 3
    assert config.model.feedforward_dim == 1472
    assert config.model.window_size == 20
    assert config.training.seed == 1919
    assert config.training.updates == 12_000
    assert config.training.checkpoint_interval == 2_000
    assert config.training.checkpoint_keep >= 6
    assert not config.training.evaluate_on_checkpoint
    assert config.training.early_stopping_patience == 0
    assert sum(parameter.numel() for parameter in TrackMT3(config).parameters()) == 8_484_460
