from pathlib import Path

from track_mt3.config_merge import load_merged_config
from track_mt3.models import TrackMT3


ROOT = Path(__file__).resolve().parents[1]


def test_capacity_matched_control_is_parameter_matched():
    config = load_merged_config(
        ROOT / "configs/paper.yaml",
        ROOT / "configs/training/track_mt3.yaml",
        ROOT / "configs/training/track_mt3_cm.yaml",
    )
    assert config.model.encoder_layers == 3
    assert config.model.decoder_layers == 3
    assert config.model.feedforward_dim == 1472
    assert config.model.window_size == 20
    assert sum(parameter.numel() for parameter in TrackMT3(config).parameters()) == 8_484_460
