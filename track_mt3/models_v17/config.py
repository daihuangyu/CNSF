from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class V17AConfig:
    """Configuration for the oracle recursive-filter gate.

    v17-A deliberately keeps this surface small.  Learned association,
    Sinkhorn, Mamba and learned lifecycle belong to later independently gated
    versions and are not hidden behind switches here.
    """

    hidden_dim: int = 128
    num_heads: int = 4
    encoder_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.0
    max_tracks: int = 24
    field_scale: float = 10.0
    min_variance: float = 1.0e-4
    max_variance: float = 100.0
    gain_residual_scale: float = 0.05
    initial_position_std: float = 0.25
    initial_velocity_std: float = 2.0
    existence_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if self.encoder_layers < 0:
            raise ValueError("encoder_layers must be non-negative")
        if self.max_tracks < 1:
            raise ValueError("max_tracks must be positive")
        if self.field_scale <= 0.0:
            raise ValueError("field_scale must be positive")
        if not 0.0 < self.min_variance <= self.max_variance:
            raise ValueError("variance bounds are invalid")


@dataclass(frozen=True)
class V17BConfig(V17AConfig):
    """Configuration for the G2 joint-association gate."""

    association_sinkhorn_iterations: int = 20
    association_temperature: float = 1.0
    association_measurement_std: float = 0.1
    association_residual_scale: float = 2.0
    joint_lifecycle_transport: bool = False
    # Optional v18-A neural posterior decoder.  A zero layer count preserves
    # every legacy v17 checkpoint and its shallow pair scorer exactly.  When
    # enabled, track priors reason jointly through self-attention and attend to
    # the current frame before the capacity-constrained transport is formed.
    association_decoder_layers: int = 0
    association_shared_observation_noise: bool = False
    association_learned_physical_gate: bool = False
    # Optional v18-B per-track Bernoulli evidence recurrence.  This is kept
    # disabled by default so every v17/v18-A checkpoint retains its exact
    # architecture and lifecycle semantics.
    track_evidence_recurrence: bool = False
    track_evidence_residual_scale: float = 4.0
    track_evidence_confirmation_threshold: float = 0.5
    # Once a recurrent track has accumulated enough evidence to be admitted,
    # keep that admission until the lifecycle model explicitly terminates the
    # track.  This separates reporting confirmation from the frame-to-frame
    # Bernoulli posterior and prevents a single ambiguous frame from silently
    # turning a mature track back into a tentative one.
    track_evidence_persistent_confirmation: bool = False
    # Optional v18-C Bernoulli log-odds filter.  Unlike v18-B's convex blend
    # of two absolute logits, this first predicts a causal survival prior and
    # then adds a learned current-frame log-likelihood ratio.
    track_evidence_log_odds_update: bool = False
    track_survival_probability_init: float = 0.985
    track_survival_logit_residual_scale: float = 3.0
    track_evidence_likelihood_residual_scale: float = 3.0
    track_evidence_pair_anchor_scale: float = 2.0
    track_evidence_pair_anchor_center: float = 0.15
    # v18-D options.  Both default to False so historical v17/v18 checkpoints
    # retain their exact numerical path.
    track_evidence_objectness_aware: bool = False
    track_evidence_exact_bernoulli_prediction: bool = False
    # v19-A: inject the learned Bernoulli existence odds as an explicit prior
    # on PAIR edges.  The legacy learned association prior remains in place;
    # this anchor prevents low-existence tentative tracks from competing with
    # mature tracks on equal structural footing in dense scenes.
    association_bernoulli_gate: bool = False
    association_bernoulli_logit_scale: float = 0.5
    # A Bernoulli filter associates with the predicted current existence
    # r^-_t=r_{t-1}p_S, not the previous posterior r_{t-1}.  The optional
    # probability floor prevents an early calibration error from completely
    # removing a geometrically plausible PAIR edge.  Both are opt-in so old
    # checkpoints retain their exact path.
    association_bernoulli_use_predicted_existence: bool = False
    association_bernoulli_probability_floor: float = 0.0
    # v19-B: reduce physical/query state writes when the exclusive association
    # posterior is ambiguous.  Association evidence itself is left untouched.
    ambiguity_aware_update: bool = False
    ambiguity_update_min_gate: float = 0.1
    # v21: PMBM-inspired measurement-born Bernoulli odds. The objectness head
    # represents local target-vs-clutter odds; the probability that the
    # measurement remains unclaimed scales the target intensity in odds space.
    # Disabled by default so v20-B and all historical checkpoints are exact.
    birth_pmbm_odds_fusion: bool = False
    birth_unclaimed_log_scale: float = 1.0
    # v21-B: a bounded neural PMBM approximation.  A learned recurrent PPP
    # summarizes undetected mass for measurement-born Bernoulli odds.  The
    # association posterior may retain two mutually exclusive global Sinkhorn
    # hypotheses instead of immediately collapsing every ambiguity.
    birth_neural_ppp: bool = False
    birth_ppp_initial_mean: float = 1.0
    birth_ppp_survival_probability: float = 0.95
    birth_ppp_detection_probability: float = 0.80
    birth_ppp_new_mean: float = 0.10
    association_mbm_hypotheses: int = 1
    association_mbm_repulsion: float = 4.0
    association_mbm_entropy_threshold: float = 0.35

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.association_sinkhorn_iterations < 1:
            raise ValueError("association_sinkhorn_iterations must be positive")
        if self.association_temperature <= 0:
            raise ValueError("association_temperature must be positive")
        if self.association_measurement_std <= 0:
            raise ValueError("association_measurement_std must be positive")
        if self.association_residual_scale <= 0:
            raise ValueError("association_residual_scale must be positive")
        if self.association_decoder_layers < 0:
            raise ValueError("association_decoder_layers must be non-negative")
        if (
            self.association_learned_physical_gate
            and not self.association_decoder_layers
        ):
            raise ValueError(
                "learned physical gate requires association_decoder_layers > 0"
            )
        if self.track_evidence_residual_scale <= 0.0:
            raise ValueError("track_evidence_residual_scale must be positive")
        if not 0.0 < self.track_evidence_confirmation_threshold < 1.0:
            raise ValueError(
                "track_evidence_confirmation_threshold must lie strictly in (0,1)"
            )
        if self.track_evidence_log_odds_update and not self.track_evidence_recurrence:
            raise ValueError(
                "log-odds evidence update requires track_evidence_recurrence"
            )
        if not 0.0 < self.track_survival_probability_init < 1.0:
            raise ValueError("track_survival_probability_init must lie in (0,1)")
        if self.track_survival_logit_residual_scale <= 0.0:
            raise ValueError("track_survival_logit_residual_scale must be positive")
        if self.track_evidence_likelihood_residual_scale <= 0.0:
            raise ValueError(
                "track_evidence_likelihood_residual_scale must be positive"
            )
        if self.track_evidence_pair_anchor_scale < 0.0:
            raise ValueError("track_evidence_pair_anchor_scale must be non-negative")
        if not 0.0 <= self.track_evidence_pair_anchor_center <= 1.0:
            raise ValueError("track_evidence_pair_anchor_center must be in [0,1]")
        if self.association_bernoulli_logit_scale < 0.0:
            raise ValueError("association_bernoulli_logit_scale must be non-negative")
        if self.association_bernoulli_gate and not self.track_evidence_recurrence:
            raise ValueError(
                "Bernoulli-gated association requires track_evidence_recurrence"
            )
        if (
            self.association_bernoulli_use_predicted_existence
            and not self.track_evidence_log_odds_update
        ):
            raise ValueError(
                "predicted-existence association requires log-odds evidence update"
            )
        if not 0.0 <= self.association_bernoulli_probability_floor < 1.0:
            raise ValueError(
                "association_bernoulli_probability_floor must lie in [0,1)"
            )
        if not 0.0 <= self.ambiguity_update_min_gate < 1.0:
            raise ValueError("ambiguity_update_min_gate must lie in [0,1)")
        if self.birth_unclaimed_log_scale < 0.0:
            raise ValueError("birth_unclaimed_log_scale must be non-negative")
        if self.birth_ppp_initial_mean <= 0.0 or self.birth_ppp_new_mean < 0.0:
            raise ValueError("PPP means must be positive/non-negative")
        if not 0.0 < self.birth_ppp_survival_probability < 1.0:
            raise ValueError("PPP survival probability must lie in (0,1)")
        if not 0.0 < self.birth_ppp_detection_probability < 1.0:
            raise ValueError("PPP detection probability must lie in (0,1)")
        if self.association_mbm_hypotheses not in (1, 2):
            raise ValueError("association_mbm_hypotheses must be 1 or 2")
        if self.association_mbm_repulsion < 0.0:
            raise ValueError("association_mbm_repulsion must be non-negative")
        if self.association_mbm_entropy_threshold < 0.0:
            raise ValueError("association_mbm_entropy_threshold must be non-negative")
