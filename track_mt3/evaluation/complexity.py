from __future__ import annotations

from dataclasses import dataclass

from track_mt3.config import ModelConfig


@dataclass(frozen=True)
class ComplexityEstimate:
    encoder_decoder: int
    prediction_heads: int
    cta: int
    qtm: int
    cal: int

    @property
    def total(self) -> int:
        return self.encoder_decoder + self.prediction_heads + self.cta + self.qtm + self.cal


def paper_complexity(
    measurements: int,
    model: ModelConfig,
    *,
    adjacent_max_targets: int,
    frame_max_targets: int,
) -> ComplexityEstimate:
    """Operation-order estimate corresponding to paper equations (57)--(59)."""
    t = int(measurements)
    d = model.hidden_dim
    df = model.feedforward_dim
    encoder_decoder = model.encoder_layers * (t * t * d + t * d * df)
    encoder_decoder += model.decoder_layers * (2 * t * t * d + t * d * df)
    prediction_heads = t * d * model.prediction_hidden_dim * model.output_dim + t * d
    cta = t * adjacent_max_targets**3
    qtm = t * d**2
    cal = t * frame_max_targets**3
    return ComplexityEstimate(encoder_decoder, prediction_heads, cta, qtm, cal)

