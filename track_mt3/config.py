from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, TypeVar

import yaml


@dataclass
class SimulationConfig:
    dt: float = 0.1
    field_of_view: Tuple[float, float, float, float] = (-10.0, 10.0, -10.0, 10.0)
    initial_targets: int = 4
    max_targets: int = 16
    birth_rate: float = 0.01
    survival_probability: float = 0.95
    detection_probability: float = 0.9
    process_noise: float = 0.5
    measurement_noise: float = 0.1
    clutter_rate: float = 10.0
    initial_position_mean: Tuple[float, float] = (0.0, 0.0)
    initial_position_covariance: Tuple[Tuple[float, float], Tuple[float, float]] = ((3.0, 0.0), (0.0, 3.0))
    initial_velocity_mean: Tuple[float, float] = (0.0, 0.0)
    initial_velocity_covariance: Tuple[Tuple[float, float], Tuple[float, float]] = ((3.0, 0.0), (0.0, 3.0))
    process_noise_is_std: bool = False
    measurement_noise_is_std: bool = False


@dataclass
class ModelConfig:
    window_size: int = 20
    measurement_dim: int = 2
    output_dim: int = 2
    hidden_dim: int = 256
    num_heads: int = 8
    encoder_layers: int = 6
    decoder_layers: int = 6
    feedforward_dim: int = 2048
    dropout: float = 0.1
    num_detection_queries: int = 16
    prediction_hidden_dim: int = 128
    prediction_layers: int = 3
    iterative_refinement: bool = True
    auxiliary_loss: bool = True
    temporal_encoding: str = "learned"
    # Re-add the positional signature to queries and keys in every encoder
    # layer. With a single injection at the input, six post-norm layers drive
    # the same-target margin from +0.0865 down to +0.0001 and cross-attention
    # degenerates to a uniform average.
    encoder_position_every_layer: bool = True
    # Pairwise association head over the encoder memory. It is the only term
    # that directly penalises memory collapse.
    contrastive_classifier: bool = True
    # Two independent knobs so the memory-key encoding and the decoder-query
    # conditioning can be ablated separately rather than as one bundle.
    spatial_encoding: str = "sinusoidal"
    # How the reference point conditions the decoder query positional embedding.
    #   "per_layer" recomputes it from the refined reference in every layer,
    #   "once"      computes it from the initial reference and holds it fixed,
    #   "none"      leaves the query positional embedding out entirely.
    # "once" is what official MT3 does (query_pos is passed unchanged into all six
    # decoder layers, and reference_points only feed the offset chain, never
    # attention) and what MOTR does for track queries (the positional half of
    # query_pos is frozen for the whole tracklet; the box enters only through
    # ref_pts, which matters there because deformable attention samples at it).
    # "per_layer" was the default through v6 and measurably does not work: the
    # reference is detached between layers, so the gradient path
    # delta -> reference -> query_pos -> attention is cut and the "a query that
    # moves also moves where it attends" effect it was written for is unlearnable,
    # while the per-layer input shift it causes is real. Probing v6 at 16k gave
    # per-decoder-layer matched error 0.900, 0.655, 0.659, 0.662, 0.658, 0.664:
    # layer 1 does all the work and layers 2-6 are inert.
    # Booleans are accepted for backwards compatibility (True -> "per_layer").
    reference_query_embedding: str = "once"
    spatial_encoding_temperature: float = 20.0
    # Cosine scores of the contrastive head are divided by this temperature before
    # the row-wise log_softmax. Without it (official MT3, and this repo through v6)
    # the scores live in [-1, 1], so over the ~200 measurements of a window the
    # objective is floor-limited to about 1 - log(e + 199/e) = 3.3 against a
    # uniform baseline of log(200) = 5.3. The whole dynamic range is 2 nats and the
    # measured value sat at 4.32-4.36 for 14000 steps without moving while the
    # encoder margin decayed from +0.83 to +0.11, i.e. the term as written cannot
    # constrain absolute cosine and therefore cannot prevent the collapse it was
    # added for.
    contrastive_temperature: float = 0.1
    # Let the network sharpen or soften that temperature itself, as in CLIP.
    contrastive_learnable_temperature: bool = True
    # Keep the layer-to-layer reference update out of the autograd graph, as
    # official MT3 does. Turning it off lets a layer learn how its own offset
    # helps the next layer, which is the only mechanism that can make iterative
    # refinement improve monotonically.
    detach_reference_between_layers: bool = True
    # Detection anchor spread. "prior" scales a grid to the initial target
    # covariance and matches "center" convergence while spreading the anchors
    # 14x wider. "grid" covers the whole field of view and measurably slows
    # optimization because the simulator rarely populates the corners.
    # "center" reproduces the previous behaviour where every anchor starts at
    # the field-of-view centre.
    detection_reference_init: str = "prior"
    # Two-stage proposal, as in official MT3 (mt3.py:170-195) and Deformable DETR.
    # The encoder scores every measurement, the top num_detection_queries become the
    # detection queries, and each one's reference point is that measurement's own
    # (refined) coordinate. It replaces the learnable detection queries entirely, so
    # it is a deliberate deviation from Track-MT3's "fixed learnable detection
    # queries". Two symptoms motivate it: localization is 1.32 even on scenario1
    # where the measurements *are* the ground truth, and the learnable anchors all
    # sit near the field-of-view centre regardless of where targets are.
    two_stage: bool = False
    # Restrict the two-stage top-k to the last ``proposal_pool_frames`` frames of the
    # window. 0 keeps the whole window, which is what official MT3 does.
    #
    # The window spans ``window_size`` frames but only the last frame's targets are
    # predicted, so a measurement from the window's start is (window_size-1)*dt old.
    # At dt=0.1 and window_size=20 that is 2.0 s, and a target at ~3 units/s has
    # drifted ~6 units, far past the GOSPA cutoff of 2.0. Probing an untrained score
    # head over a 323-measurement window gave selected indices
    # [63, 66, 68, ..., 272, 321, 322]: only 3 of 24 came from the last two frames,
    # so 21 detection queries started from a badly stale reference point. Official
    # MT3 relies on the proposal loss to teach recency implicitly -- its Hungarian
    # step matches against the current frame, so a stale measurement pays a high L1
    # cost and its score is pushed down -- and restricting the pool supplies that
    # prior directly instead of hoping it is learned.
    #
    # Several frames rather than only the last one keeps the recovery path for a
    # target missed in the current frame: its measurement from one or two frames back
    # is still a far better starting point than a learnable anchor.
    #
    # The eligible pool is often smaller than num_detection_queries -- during
    # curriculum warmup it holds 0-9 measurements -- so selection is per-row
    # variable-length and padded slots are never turned into queries.
    proposal_pool_frames: int = 0
    qtm_enabled: bool = True
    qtm_source: str = "prediction"
    detection_threshold: float = 0.75
    tracking_threshold: float = 0.5
    max_tracks: int = 16
    propagate_tracks: bool = True
    teacher_force_matched_queries: bool = False
    # Per-track Mamba in QTM: replace the single-step TFN temporal mixing with
    # a selective SSM state carried across frames.
    track_mamba_enabled: bool = False
    track_mamba_state_dim: int = 16
    # Track-before-detect: a newborn track is "tentative" until it survives
    # confirm_frames consecutive frames at the strict tracking_threshold; only
    # then does it become "confirmed" and may coast at the lax
    # confirmed_threshold. This filters single-frame clutter that a naive
    # detection query would otherwise turn into a false track.
    # Disable this for checkpoints trained before track-before-detect was added:
    # those models used tracking_threshold for every propagated track.
    track_confirmation_enabled: bool = True
    confirm_frames: int = 3
    confirmed_threshold: float = 0.5
    # DFSMN-style historical-measurement fusion: the encoder sees only the
    # current frame, but its output is fused with the previous
    # memory_frames frames' measurement encodings (memory_tokens_per_frame
    # tokens each) via cross-attention.
    memory_frames: int = 0
    memory_tokens_per_frame: int = 8
    # Encode every frame exactly once, then retain the encoded measurements in a
    # frame-structured ring buffer.  The decoder queries this buffer; historical
    # measurements never attend to one another again and are never re-encoded.
    streaming_cache_frames: int = 0
    streaming_cache_tokens_per_frame: int = 32

    def __post_init__(self) -> None:
        # Configs and checkpoints written before the three-way switch existed used
        # a boolean here. Normalising once keeps every consumer on the string form.
        if isinstance(self.reference_query_embedding, bool):
            self.reference_query_embedding = (
                "per_layer" if self.reference_query_embedding else "none"
            )
        allowed = ("per_layer", "once", "none")
        if self.reference_query_embedding not in allowed:
            raise ValueError(
                "reference_query_embedding must be one of "
                f"{allowed}, got {self.reference_query_embedding!r}"
            )
        if self.contrastive_temperature <= 0.0:
            raise ValueError("contrastive_temperature must be positive")
        if self.memory_frames > 0 and self.streaming_cache_frames > 0:
            raise ValueError("memory fusion and streaming measurement cache are mutually exclusive")
        if self.streaming_cache_frames < 0 or self.streaming_cache_tokens_per_frame < 1:
            raise ValueError("streaming cache dimensions must be positive")


@dataclass
class LossConfig:
    localization_weight: float = 1.0
    confidence_weight: float = 1.0
    auxiliary_weight: float = 1.0
    matching_state_weight: float = 1.0
    matching_class_weight: float = 1.0
    empty_target_normalizer: float = 1.0
    confidence_positive_weight: float = 1.0
    auxiliary_matching: str = "shared"
    # "uniform" reproduces official MT3 (every decoder layer weighted equally).
    # "progressive" ramps the weight linearly across auxiliary layers so that later
    # layers dominate, which rewards refining the previous layer instead of
    # re-predicting independently.
    auxiliary_layer_weighting: str = "uniform"
    # Official MT3 sums every decoder layer with equal weight and scales the
    # contrastive term by 4. With contrastive_temperature in place the term is no
    # longer floor-limited, so this weight is not comparable to the pre-temperature
    # value and needs its own ablation.
    contrastive_weight: float = 4.0
    # Weight of the encoder-proposal auxiliary loss used with model.two_stage. Without
    # it the top-k selection is unsupervised and there is nothing teaching the encoder
    # which measurements are targets. Official MT3 adds the same term at weight 1.0.
    proposal_weight: float = 1.0
    # How localization and confidence are normalised inside one frame.
    #   "sum_per_target" divides both sums by max(num_targets, empty_normalizer),
    #   "mean"           averages L1 over matched pairs and coordinates and BCE over
    #                    all queries, which is exactly what official MT3 does
    #                    (F.l1_loss and BCEWithLogitsLoss both at reduction="mean",
    #                    summed with implicit weights 1 and 1).
    # "sum_per_target" was the default through v6 and silently skews the balance:
    # with 2 coordinates it inflates localization by 2x over a mean, but with ~20
    # queries per 4 targets it inflates confidence by ~5x, so at equal weights
    # localization is relatively 2.5x weaker than in the paper. Tuning the scalar
    # weight to compensate (v6 ramped it to 5.0) fixes the ratio but leaves the
    # absolute magnitude incomparable to both the paper and to earlier runs.
    reduction: str = "mean"
    # Weight multiplier for newborn target pairs in localization and confidence
    # loss. A newborn target (matched by Hungarian, not inherited) has only one
    # frame of information, so lowering its weight lets the model learn "something
    # is here" before demanding precise localisation. 1.0 = no distinction.
    newborn_weight: float = 1.0
    # Per-target weight ramp based on target age (frames since birth). The weight
    # for a target of age a is: max(newborn_weight, 1 - age_ramp * (1 - newborn_weight) / a).
    # With newborn_weight=0.3 and age_ramp=5, age=1→0.3, age=3→0.58, age=5→0.82,
    # age≥5→1.0. Set 0 to disable and use a flat newborn_weight for all newborns.
    age_ramp: int = 0

    def __post_init__(self) -> None:
        allowed = ("sum_per_target", "mean")
        if self.reduction not in allowed:
            raise ValueError(f"loss.reduction must be one of {allowed}, got {self.reduction!r}")


@dataclass
class TrainingConfig:
    seed: int = 0
    device: str = "auto"
    batch_size: int = 32
    clip_windows: int = 4
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 1.0
    teacher_forcing_decay_steps: int = 0
    updates: int = 50_000
    learning_rate: float = 2e-4
    # ReduceLROnPlateau is driven by the validation loop, so the patience unit
    # is validation rounds rather than optimizer steps.
    lr_patience: int = 10
    lr_factor: float = 0.5
    weight_decay: float = 0.0
    ddp_find_unused_parameters: bool = False
    gradient_clip_norm: float = 1.0
    # Optional linear schedules for loss weights, expressed as
    # {"localization_weight": [start, end, steps], ...}. The weight moves linearly
    # from start to end over the given number of updates and is then held. Use this
    # when the bottleneck migrates during training, e.g. representation collapse
    # first and localization accuracy later.
    loss_weight_schedule: Dict[str, List[float]] = field(default_factory=dict)
    # Curriculum learning: simulation difficulty ramps linearly from easy to full
    # over this many optimizer steps. 0 disables the curriculum (the simulation
    # starts at its configured difficulty immediately). A positive value means the
    # first ``curriculum_warmup_steps`` steps train on progressively harder scenes:
    #   clutter_rate        0      → config value   (learn detection before clutter)
    #   detection_prob      1.0    → config value   (perfect detections first)
    #   process_noise      0.1    → config value   (near-constant velocity first)
    #   measurement_noise   0.02   → config value   (precise measurements first)
    #   initial_targets    1      → config value   (single target first)
    #   max_targets        4      → config value   (fewer slots first)
    #   birth_rate         0      → config value   (stable scene first)
    #   survival_prob      1.0    → config value   (no disappearances first)
    # The terminal values are read from ``config.simulation`` so that a YAML
    # override of any simulation parameter is automatically respected.
    curriculum_warmup_steps: int = 0
    # Fraction of training clips that start at the beginning of a sequence, where the
    # sliding window is not yet full. Required for the tracker to produce estimates on
    # frames 1..window_size-1 at deployment; neither MT3 nor Track-MT3 covers that regime.
    cold_start_clip_probability: float = 0.0
    checkpoint_interval: int = 1_000
    checkpoint_keep: int = 2
    log_interval: int = 50
    validation_interval: int = 500
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_metric: str = "mean_pro_gospa"
    wandb_enabled: bool = False
    wandb_project: str = "track-mt3"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_log_interval: int = 1
    evaluate_on_checkpoint: bool = False
    evaluation_dataset_dir: str = "datasets/track_mt3_paper/evaluation"
    # Trajectory replayed and stored at every checkpoint evaluation, so training
    # progress can be inspected as tracking pictures rather than only as metrics.
    demo_scene: str = "scenario3"
    demo_run: int = 0
    demo_plot: bool = True
    demo_plot_panels: int = 8
    output_dir: str = "outputs/paper"


@dataclass
class EvaluationConfig:
    gospa_cutoff: float = 2.0
    gospa_order: int = 1
    gospa_alpha: float = 2.0
    existence_threshold: float = 0.75
    monte_carlo_runs: int = 50
    trajectory_steps: int = 100
    warmup_windows: int = 1
    batch_size: int = 10


@dataclass
class ExperimentConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _construct(cls: Type[T], values: Dict[str, Any]) -> T:
    valid = {item.name: item for item in fields(cls)}
    unknown = set(values) - set(valid)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: Dict[str, Any] = {}
    for name, value in values.items():
        field_info = valid[name]
        default = getattr(cls(), name)
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _construct(type(default), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    return _construct(ExperimentConfig, values)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
