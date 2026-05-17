# Experiment Development Plan: Does the Model Plan Before It Renders?

> This document is a concrete engineering and research plan for the planning circuit study. It covers code organisation, function contracts, logging strategy, and how mechanistic interpretability tools plug into each experiment stage. The goal is a codebase where every experiment is a configuration, every measurement is reproducible, and every mechanistic claim is backed by a logged artefact.

---

## 0. The Window of Questions This Plan Covers

All experiments below are windows into the same central question from different angles:

| Window | Core question | Primary tool |
|---|---|---|
| W1 — Probe timing | Does the model compute the size ratio before generation begins? | Layer-wise linear probe |
| W2 — Curriculum effect | Which data ordering creates the planning circuit fastest? | Verifier timeline × curriculum |
| W3 — Weight transition | What exact weight update corresponds to the circuit forming? | Gradient tracking + SVD at transition |
| W4 — Thinking token | Does the reasoning trace carry causal geometric state? | Activation patching + KV probe |
| W5 — Invariance | Is the circuit shape-general, scale-general, color-general? | Targeted OOD evaluation |
| W6 — Threshold | How many tight examples are needed for the circuit to form? | Binary search + verifier |
| W7 — Error taxonomy | Which error types fall together and which independently? | Verifier over all checkpoints |

---

## 1. Repository Layout

```
planning_circuit/
│
├── data/
│   ├── scene.py               # Scene dataclass and renderer
│   ├── generator.py           # Parametric scene generation
│   ├── prompt.py              # Prompt builder (all explicitness levels)
│   ├── axes.py                # Axis enum definitions and samplers
│   ├── splits.py              # All train/test split strategies
│   ├── curriculum.py          # Curriculum schedulers
│   └── dataset.py             # torch Dataset wrappers
│
├── model/
│   ├── transformer.py         # Small transformer, trained from scratch
│   ├── tokenizer.py           # Joint image+text tokenizer
│   ├── trainer.py             # Training loop with hook support
│   └── checkpoint.py          # Save/load with full experiment state
│
├── verifier/
│   ├── checks.py              # Individual check functions
│   ├── taxonomy.py            # ErrorVector and ErrorType definitions
│   └── pipeline.py            # Full verifier pipeline + batch eval
│
├── mech_interp/
│   ├── hooks.py               # Activation and gradient hook manager
│   ├── probes.py              # Linear probe training and evaluation
│   ├── attention.py           # Attention pattern extraction and metrics
│   ├── gradients.py           # Per-weight gradient tracking
│   ├── weights.py             # Weight analysis: norm, rank, SVD
│   ├── patching.py            # Activation patching for causal tests
│   └── kv_cache.py            # KV cache extraction and probing
│
├── experiments/
│   ├── w1_probe_timing.py     # W1: layer-wise probe experiment
│   ├── w2_curriculum.py       # W2: curriculum comparison
│   ├── w3_weight_transition.py# W3: weight update at transition
│   ├── w4_thinking.py         # W4: thinking token causality
│   ├── w5_invariance.py       # W5: invariance tests
│   ├── w6_threshold.py        # W6: minimum data threshold
│   └── w7_taxonomy.py         # W7: error taxonomy timeline
│
├── logging/
│   ├── logger.py              # Central logger (wraps wandb + local)
│   ├── metrics.py             # Metric definitions and aggregators
│   └── artefacts.py           # Attention maps, weight snapshots, probes
│
├── analysis/
│   ├── timeline.py            # Circuit formation timeline from logs
│   ├── transition.py          # Transition checkpoint detection
│   └── compare.py             # Cross-curriculum and cross-split comparison
│
├── configs/
│   ├── base.yaml              # Shared hyperparameters
│   ├── model/                 # Model size configs
│   ├── curricula/             # One yaml per curriculum design
│   └── experiments/           # One yaml per experiment window
│
└── run.py                     # Entry point: loads config, runs experiment
```

---

## 2. Data Layer

### 2.1 Scene and axes

```python
# data/axes.py

from enum import IntEnum
from dataclasses import dataclass
import numpy as np

class ExplicitnessLevel(IntEnum):
    """
    How much geometric information the prompt reveals.
    Level 0 = no constraint stated.
    Level 5 = explicit geometric proof embedded in prompt.
    """
    NONE        = 0   # "a circle and a square"
    RELATION    = 1   # "a circle inside a square"
    QUALITATIVE = 2   # "a small circle inside a large square"
    RELATIVE    = 3   # "a circle half the width of the square, inside it"
    ABSOLUTE    = 4   # "circle radius 20, square side 60, circle inside square"
    PROOF       = 5   # full geometric proof embedded in prompt

class Tightness(IntEnum):
    """Margin between inner object boundary and outer object boundary."""
    INVALID = -1   # geometrically impossible; labeled as INVALID in prompt
    EXACT   = 0    # inner object touches all sides of outer
    TIGHT   = 1    # 1-5% margin
    MEDIUM  = 2    # 5-20% margin
    LOOSE   = 3    # >20% margin

class ComplexityLevel(IntEnum):
    """Number of objects and relations in the scene."""
    SINGLE        = 0   # one object, no relation
    TWO_FREE      = 1   # two objects, no spatial constraint
    TWO_POSITIONAL= 2   # two objects, positional relation (left of, above)
    TWO_GEOMETRIC = 3   # two objects, geometric constraint (inside, touching)
    THREE_CHAIN   = 4   # three objects, chain of relations
    N_HIERARCHICAL= 5   # N objects, nested/hierarchical constraints

class FreedomLevel(IntEnum):
    """How many parameters are left unspecified in the prompt."""
    FULLY_SPECIFIED   = 0  # all parameters given
    RANGE_SPECIFIED   = 1  # parameters given as intervals
    RELATION_ONLY     = 2  # only relation stated, sizes free
    FULLY_FREE        = 3  # nothing specified


@dataclass
class SceneParams:
    """Complete ground-truth parameters for a scene. The verifier reads this."""
    objects: list          # list of ObjectParams
    relations: list        # list of RelationParams
    canvas_size: tuple     # (W, H)
    explicitness: ExplicitnessLevel
    tightness: Tightness
    complexity: ComplexityLevel
    freedom: FreedomLevel
    is_valid: bool         # False for INVALID tightness examples

@dataclass
class ObjectParams:
    shape: str             # "circle", "square", "triangle", "text"
    color: tuple           # (R, G, B)
    position: tuple        # (cx, cy) normalised 0-1
    size: float            # radius for circle, side for square, normalised
    font: str | None       # for text objects
    content: str | None    # for text objects

@dataclass
class RelationParams:
    relation_type: str     # "inside", "left_of", "above", "touching"
    subject_idx: int       # index into SceneParams.objects
    object_idx: int        # index into SceneParams.objects
    margin: float          # computed margin (negative if invalid)
```

```python
# data/generator.py

class SceneGenerator:
    """
    Generates SceneParams objects according to axis settings.
    All randomness is seeded and logged so scenes are reproducible.
    """

    def __init__(self, rng_seed: int, canvas_size=(256, 256)):
        self.rng = np.random.default_rng(rng_seed)
        self.canvas_size = canvas_size

    def generate(
        self,
        complexity: ComplexityLevel,
        tightness: Tightness,
        freedom: FreedomLevel,
        explicitness: ExplicitnessLevel,
    ) -> SceneParams:
        """
        Main entry point. Generates a valid (or deliberately invalid)
        scene according to axis settings. Returns full ground-truth params.
        All geometric constraints are verified before returning.
        """
        ...

    def _sample_containment_pair(self, tightness: Tightness) -> tuple[ObjectParams, ObjectParams]:
        """
        Sample inner (circle) and outer (square) parameters such that
        the tightness constraint is satisfied exactly.
        Returns (inner, outer).
        Raises ValueError if tightness=INVALID and no invalid config is found
        within max_tries — should not happen with correct parameter sampling.
        """
        ...

    def _compute_margin(self, inner: ObjectParams, outer: ObjectParams) -> float:
        """
        Returns signed margin in pixels.
        Positive = inner fits inside outer with this clearance.
        Negative = inner overflows outer by this amount.
        """
        ...
```

```python
# data/prompt.py

class PromptBuilder:
    """
    Converts a SceneParams into a natural language prompt at the
    requested explicitness level. The same scene can be described
    at any level; the image is identical, only the prompt changes.
    """

    def build(self, scene: SceneParams, level: ExplicitnessLevel) -> str:
        """
        Returns the NLP prompt for this scene at the given explicitness level.
        Level 5 (PROOF) embeds a step-by-step geometric verification.
        """
        ...

    def build_proof(self, scene: SceneParams) -> str:
        """
        Builds a Level-5 proof string.
        Example output:
          'Circle radius 20, square side 60.
           Containment check: 20 < 60/2 = 30 ✓
           Center (30, 30), distance to nearest edge = 30, clearance = 30 - 20 = 10 ✓
           Configuration is valid. Render.'
        This string is prepended to the image generation target.
        """
        ...
```

### 2.2 Splits

```python
# data/splits.py

class SplitStrategy:
    """Base class for all train/test split strategies."""

    def split(self, dataset: list[SceneParams]) -> tuple[list, list]:
        """Returns (train_scenes, test_scenes)."""
        raise NotImplementedError


class StructuredOODSplit(SplitStrategy):
    """
    Train on axis_value_A, test on axis_value_B.
    Guarantees zero overlap between train and test on the split axis.

    Example: train on LOOSE tightness, test on TIGHT tightness.
    Answers: does the planning circuit generalise to tighter constraints?
    """
    def __init__(self, axis: str, train_values: list, test_values: list): ...


class SubgroupSplit(SplitStrategy):
    """
    For arithmetic/group datasets.
    Train on elements of a subgroup, test on elements outside.
    Guarantees the test set follows the same rule but uses unseen elements.
    """
    def __init__(self, group_size: int, subgroup_indices: list[int]): ...


class AxisCrossingSplit(SplitStrategy):
    """
    Train on (axis_A=v1, axis_B=v1). Test on (axis_A=v1, axis_B=v2)
    and (axis_A=v2, axis_B=v1). Tests whether generalisation is
    axis-specific or compositional across axes.
    """
    def __init__(self, axis_a: str, axis_b: str,
                 train_combo: tuple, test_combos: list[tuple]): ...


class DisjointVocabSplit(SplitStrategy):
    """
    Train on shape_type_A and shape_type_B independently.
    Test on compositions involving both.
    Answers: are rendering circuits compositional or entangled?
    """
    def __init__(self, train_shapes: list[str], test_compositions: list[tuple]): ...


class ThresholdSplit(SplitStrategy):
    """
    Include exactly N examples of a specific type in the training set.
    All other examples are from the base distribution.
    Used for binary-search minimum-data experiments.
    """
    def __init__(self, special_type: dict, n_special: int, n_base: int): ...
```

### 2.3 Curriculum

```python
# data/curriculum.py

class CurriculumScheduler:
    """
    Controls which examples the model sees at each training step.
    The scheduler is a deterministic function of step number and config,
    so the exact training sequence is always reproducible from the config.
    """

    def get_batch(self, step: int, batch_size: int) -> list[SceneParams]:
        """Returns the batch for training step `step`."""
        raise NotImplementedError

    def describe(self) -> str:
        """Human-readable description of the curriculum for logging."""
        raise NotImplementedError


class PureImplicit(CurriculumScheduler):
    """Curriculum I: Level-1 prompts only. All tightness levels uniform."""

class PureExplicit(CurriculumScheduler):
    """Curriculum II: Level-4 prompts only."""

class ProofToImplicit(CurriculumScheduler):
    """
    Curriculum III: Level-5 for first third, Level-4 for second third,
    Level-1 for final third. Each stage introduces the previous stage's
    examples at low proportion to prevent forgetting.
    """

class TightFirst(CurriculumScheduler):
    """Curriculum IV: TIGHT tightness examples dominate early; LOOSE added later."""

class InvalidInterleaved(CurriculumScheduler):
    """Curriculum V: Adds INVALID examples at ratio r alongside valid ones."""
    def __init__(self, base_curriculum: CurriculumScheduler, invalid_ratio: float): ...

class FreedomToConstraint(CurriculumScheduler):
    """Curriculum VII: Starts FULLY_FREE, tightens freedom over time."""

class CrossRelationTransfer(CurriculumScheduler):
    """
    Curriculum VIII: Trains containment to grokking, then introduces
    ordering while using a lower LR on containment-circuit weights.
    Requires identifying containment-circuit weights first (from W3).
    """
    def __init__(self, containment_circuit_params: list[str], lr_scale: float): ...
```

---

## 3. Model Layer

```python
# model/transformer.py

class PlanningTransformer(nn.Module):
    """
    Small transformer trained from scratch.
    Designed for full observability: every weight matrix is named
    consistently so mech_interp tools can address them by name.

    Architecture:
      - Joint token vocabulary: image patch tokens + text tokens
      - Standard causal transformer with pre-norm
      - All weight matrices follow naming convention:
          layers.{L}.attn.W_Q, W_K, W_V, W_O
          layers.{L}.mlp.W_in, W_out
          embed.token, embed.position
    """

    def __init__(self, cfg: ModelConfig): ...

    def forward(
        self,
        tokens: Tensor,           # (B, T)
        return_activations: bool = False,
        return_attention: bool = False,
    ) -> ModelOutput:
        """
        If return_activations=True, output includes residual stream at every layer.
        If return_attention=True, output includes attention weights at every layer.
        These flags add compute cost; enable only during eval/interp passes.
        """
        ...

    def get_layer_names(self) -> list[str]:
        """Returns all addressable weight matrix names for mech_interp tools."""
        ...
```

```python
# model/trainer.py

class Trainer:
    """
    Training loop. Owns the optimizer, scheduler, and checkpoint manager.
    Calls the eval pipeline and mech_interp tools at configured intervals.
    """

    def __init__(
        self,
        model: PlanningTransformer,
        curriculum: CurriculumScheduler,
        verifier_pipeline: VerifierPipeline,
        mech_interp_suite: MechInterpSuite,
        logger: ExperimentLogger,
        cfg: TrainConfig,
    ): ...

    def train(self, n_steps: int):
        """
        Main training loop. At every step:
          - Get batch from curriculum
          - Forward + backward
          - Log step-level metrics (loss, grad norms)
          - At eval_every steps: run verifier, run mech_interp suite
          - At checkpoint_every steps: save full checkpoint
          - Detect transition if verifier signals it
          - On transition: trigger W3 (weight transition analysis)
        """
        ...

    def _detect_transition(self, verifier_history: list[TaxonomyReport]) -> int | None:
        """
        Scans verifier history for the first checkpoint where
        geometric_calculation_error rate drops below threshold.
        Returns the step number of the transition, or None if not yet detected.
        This is the trigger for W3.
        """
        ...
```

---

## 4. Verifier Layer

```python
# verifier/checks.py

def check_shape_quality(image: np.ndarray, obj: ObjectParams, tolerance: float = 0.05) -> bool:
    """
    Checks whether the rendered object looks like the intended shape.
    Uses simple geometric CV (Hough circles, contour fitting) not a neural model.
    Tolerance is fraction of canvas size.
    """
    ...

def check_attribute_binding(
    image: np.ndarray,
    objects: list[ObjectParams],
) -> dict[int, bool]:
    """
    For each object, checks whether the rendered color matches the specified color.
    Returns {object_idx: passed}.
    Failure here = attribute binding error, not geometric error.
    """
    ...

def check_containment(
    image: np.ndarray,
    inner: ObjectParams,
    outer: ObjectParams,
    scene_params: SceneParams,
) -> ContainmentResult:
    """
    Checks whether every point on the inner object's boundary is inside
    the outer object's boundary.
    Returns:
      - passed: bool
      - margin_px: float (signed, negative = overflow)
      - margin_fraction: float (relative to outer size)
    """
    ...

def check_ordering(
    image: np.ndarray,
    objects: list[ObjectParams],
    ordering_axis: str,   # "left_to_right", "top_to_bottom"
    ordering_property: str,   # "size", "color_hue", "position"
) -> bool:
    """Checks whether objects satisfy the stated ordering constraint."""
    ...

def check_cardinality(image: np.ndarray, expected_count: int, shape: str) -> tuple[bool, int]:
    """Returns (passed, actual_count_detected)."""
    ...
```

```python
# verifier/taxonomy.py

from dataclasses import dataclass, field

@dataclass
class ErrorVector:
    """
    One ErrorVector per generated image. Each field is True if that
    check passed, False if it failed. None if the check was not applicable.
    Stored in full for every eval example at every checkpoint.
    """
    shape_quality:          bool | None = None
    attribute_binding:      bool | None = None
    containment_valid:      bool | None = None
    scale_consistent:       bool | None = None  # inner < outer * valid_ratio
    geometric_calculation:  bool | None = None  # config is achievable at all
    cardinality:            bool | None = None
    constraint_ignored:     bool | None = None  # constraint stated but fully ignored
    preference_calibration: float | None = None # margin fraction when free

    @property
    def planning_circuit_signature(self) -> bool:
        """
        True if shape and attribute are OK but containment fails.
        This is the specific signature of an absent planning circuit:
        the model can draw objects but cannot compose them.
        """
        return (
            self.shape_quality is True
            and self.attribute_binding is True
            and self.containment_valid is False
        )


@dataclass
class TaxonomyReport:
    """Aggregated error rates across the eval set at one checkpoint."""
    step: int
    n_examples: int
    error_rates: dict[str, float]          # field_name -> fraction failed
    planning_signature_rate: float         # fraction showing planning_circuit_signature
    preference_distribution: list[float]   # margin fractions for free examples
    raw_vectors: list[ErrorVector] = field(default_factory=list)  # kept for reanalysis
```

```python
# verifier/pipeline.py

class VerifierPipeline:
    """
    Runs the full verifier on the model's outputs over an eval set.
    Called at every eval checkpoint during training.
    """

    def __init__(self, eval_dataset: list[tuple[SceneParams, str]], batch_size: int = 32):
        self.eval_dataset = eval_dataset

    def evaluate(self, model: PlanningTransformer, step: int) -> TaxonomyReport:
        """
        Generates images for every eval example, runs all checks,
        returns a TaxonomyReport. The report is logged immediately.
        """
        ...

    def evaluate_split(
        self,
        model: PlanningTransformer,
        split: SplitStrategy,
        step: int,
    ) -> dict[str, TaxonomyReport]:
        """
        Runs the verifier separately on the train split and test split.
        Returns {'train': TaxonomyReport, 'test': TaxonomyReport}.
        The gap between train and test planning_signature_rate is the
        generalisation gap for the planning circuit.
        """
        ...
```

---

## 5. Mechanistic Interpretability Layer

```python
# mech_interp/hooks.py

class HookManager:
    """
    Registers forward hooks on all layers of the model.
    Hooks are cheap by default (no-op) and activated selectively.
    All captured activations are stored in a flat dict keyed by layer name.
    """

    def __init__(self, model: PlanningTransformer):
        self.model = model
        self._hooks = {}
        self._store = {}

    def register_all(self):
        """Registers hooks on every named module. No-op cost until activated."""
        ...

    def activate(self, layer_names: list[str]):
        """
        Activates capture for specified layers.
        Only activated layers store tensors; others pass through.
        Call this before a forward pass, deactivate after.
        """
        ...

    def deactivate(self):
        """Stops capture, clears store."""
        ...

    def get(self, layer_name: str) -> Tensor:
        """Returns captured activation for a layer. Raises if not captured."""
        ...

    @contextmanager
    def capturing(self, layer_names: list[str]):
        """Context manager: activate, yield store, deactivate."""
        self.activate(layer_names)
        try:
            yield self._store
        finally:
            self.deactivate()
```

```python
# mech_interp/probes.py

class LinearProbe(nn.Module):
    """
    A single linear layer trained to decode a property from residual stream
    activations at a specific layer and token position.

    The key probe for W1: predict size_ratio (float) from the residual stream
    at layer L, at the token position just before the first image token.
    If this probe is accurate at L < first_image_token_layer, the model
    computed the ratio internally before generation began.
    """

    def __init__(self, d_model: int, output_dim: int, probe_type: str = "regression"):
        """probe_type: 'regression' for continuous targets, 'classification' for discrete."""
        ...

    def forward(self, activation: Tensor) -> Tensor: ...


class ProbeTrainer:
    """Trains a LinearProbe on captured activations."""

    def train_probe(
        self,
        model: PlanningTransformer,
        hook_manager: HookManager,
        dataset: list[tuple[str, SceneParams]],  # (prompt, ground_truth_params)
        layer_name: str,
        token_position: int | str,   # int or "pre_generation" / "post_generation"
        target_property: str,        # "size_ratio", "center_x", "margin_fraction"
        n_epochs: int = 100,
    ) -> tuple[LinearProbe, ProbeResult]:
        """
        Captures activations at layer_name for all dataset examples,
        trains a probe to predict target_property, returns the probe
        and its accuracy/R² on a held-out probe-test set.

        Critical: the probe-test set must be DIFFERENT from the model's
        training data. The probe is measuring what the model computed,
        not what it was trained to compute.
        """
        ...

    def probe_all_layers(
        self,
        model: PlanningTransformer,
        hook_manager: HookManager,
        dataset: list,
        target_property: str,
        token_position: str,
    ) -> dict[str, ProbeResult]:
        """
        Runs train_probe for every layer in the model.
        Returns {layer_name: ProbeResult}.
        The profile of probe accuracy across layers is the
        'where in the network' answer for the target property.
        """
        ...


@dataclass
class ProbeResult:
    layer_name: str
    token_position: str
    target_property: str
    train_r2: float
    test_r2: float       # key metric: high R² = property is decodable here
    train_mse: float
    test_mse: float
    is_pre_generation: bool   # True if this layer fires before image tokens begin
```

```python
# mech_interp/attention.py

class AttentionAnalyzer:
    """
    Extracts and analyses attention patterns.
    Focuses on cross-object attention: which heads attend from
    the inner object region to the outer object region (and vice versa),
    as this is the signature of the containment circuit checking itself.
    """

    def get_all_patterns(
        self,
        model: PlanningTransformer,
        prompt: str,
        scene: SceneParams,
    ) -> dict[str, Tensor]:
        """
        Returns {layer_name: attention_matrix} for all heads at all layers.
        Shape of each matrix: (n_heads, T, T).
        """
        ...

    def compute_cross_object_entropy(
        self,
        attention: dict[str, Tensor],
        inner_token_range: tuple[int, int],
        outer_token_range: tuple[int, int],
    ) -> dict[str, float]:
        """
        For each head in each layer, computes how much attention flows
        from inner-object tokens to outer-object tokens.
        High cross-object attention = head is doing cross-object reasoning.
        Returns {layer.head: cross_object_fraction}.
        """
        ...

    def compare_to_theoretical_optimum(
        self,
        model_attention: Tensor,
        theoretical_attention: Tensor,
    ) -> float:
        """
        Computes cosine similarity between model's attention pattern
        and the theoretically optimal pattern (derived from the task structure).
        For containment: the optimal pattern has the inner-object token
        attending maximally to the outer-object size token.
        Returns similarity in [0, 1].
        """
        ...

    def track_attention_evolution(
        self,
        checkpoints: list[Path],
        prompt: str,
        scene: SceneParams,
        layer: str,
        head: int,
    ) -> list[Tensor]:
        """
        Returns attention patterns at each checkpoint for a fixed input.
        Plots how a specific head's attention evolves during training.
        The transition checkpoint should show a structural change in the pattern.
        """
        ...
```

```python
# mech_interp/gradients.py

class GradientTracker:
    """
    Tracks which weight matrices receive the largest gradient updates
    during specific types of training examples.
    The core tool for W3: finding the weight update that forms the planning circuit.
    """

    def compute_per_weight_gradient_norms(
        self,
        model: PlanningTransformer,
        batch: list[tuple[str, SceneParams]],
        loss_fn: callable,
        filter_example_type: str | None = None,
    ) -> dict[str, float]:
        """
        Runs a forward+backward pass on the batch.
        Returns {weight_name: gradient_norm} for every named parameter.
        If filter_example_type is set (e.g. 'containment_fail'),
        only examples of that type contribute to the gradient.
        """
        ...

    def compare_gradient_profiles(
        self,
        profile_A: dict[str, float],   # gradients on composition examples
        profile_B: dict[str, float],   # gradients on single-object examples
    ) -> dict[str, float]:
        """
        Returns {weight_name: ratio} = profile_A[w] / profile_B[w].
        Weights with ratio >> 1 receive disproportionate gradient from
        composition examples: candidate planning circuit components.
        """
        ...

    def track_gradient_at_transition(
        self,
        checkpoints: list[Path],
        transition_step: int,
        window: int = 5,
    ) -> dict[str, list[float]]:
        """
        For each weight, plots its gradient norm over the window of
        checkpoints around the transition step.
        The weight with the largest gradient spike at the transition
        is the most likely planning circuit formation site.
        """
        ...
```

```python
# mech_interp/weights.py

class WeightAnalyzer:
    """
    Analyses weight matrix geometry: norm, effective rank, SVD trajectory.
    """

    def effective_rank(self, W: Tensor, threshold: float = 0.01) -> int:
        """
        Number of singular values above threshold * max_singular_value.
        Low effective rank = solution is compressed / generalised.
        High effective rank = solution is high-dimensional / memorised.
        """
        ...

    def svd_trajectory(
        self,
        checkpoints: list[Path],
        weight_name: str,
    ) -> list[SVDSnapshot]:
        """
        Computes SVD of the named weight matrix at each checkpoint.
        Returns the trajectory of singular values and top singular vectors.
        A sudden change in the singular value distribution at the transition
        checkpoint is mechanistic evidence of circuit formation.
        """
        ...

    def cosine_similarity_across_checkpoints(
        self,
        checkpoints: list[Path],
        weight_name: str,
    ) -> np.ndarray:
        """
        (n_checkpoints × n_checkpoints) matrix of cosine similarities
        between the flattened weight matrices at each pair of checkpoints.
        A block structure (high similarity within a block) indicates
        stable training phases separated by transition events.
        """
        ...

    def compare_memorised_vs_generalised(
        self,
        memorised_checkpoint: Path,
        generalised_checkpoint: Path,
        layer_names: list[str],
    ) -> dict[str, dict]:
        """
        For each layer, compares norm, rank, and top singular vectors
        between the memorising and generalising checkpoint.
        Returns the geometric difference that corresponds to grokking.
        """
        ...


@dataclass
class SVDSnapshot:
    step: int
    weight_name: str
    singular_values: np.ndarray
    left_vectors: np.ndarray    # U
    right_vectors: np.ndarray   # V^T
    effective_rank: int
    frobenius_norm: float
```

```python
# mech_interp/patching.py

class ActivationPatcher:
    """
    Activation patching for causal verification.
    The question: if we replace the activations at layer L with those
    from a 'planning-successful' run, does a 'planning-failed' run
    suddenly succeed?
    This verifies that layer L is causally necessary for planning.
    """

    def patch_and_run(
        self,
        model: PlanningTransformer,
        source_prompt: str,      # prompt from a run where planning succeeded
        target_prompt: str,      # prompt from a run where planning failed
        layer_name: str,
        token_position: int,
        hook_manager: HookManager,
    ) -> ModelOutput:
        """
        Captures activations from source_prompt at layer_name, token_position.
        Runs target_prompt but replaces its activations at that point with source's.
        Returns the patched output.
        If the patched output now satisfies the geometric constraint, layer_name
        at token_position is causally sufficient for planning.
        """
        ...

    def sweep_patching(
        self,
        model: PlanningTransformer,
        source_prompt: str,
        target_prompt: str,
        verifier: VerifierPipeline,
    ) -> dict[str, dict[int, bool]]:
        """
        Runs patch_and_run for every (layer, token_position) combination.
        Returns {layer: {token_pos: planning_succeeded_after_patch}}.
        The (layer, token_pos) that flips planning from fail to pass
        is the causal location of the planning computation.
        """
        ...
```

```python
# mech_interp/kv_cache.py

class KVCacheAnalyzer:
    """
    For the thinking-token variant (W4).
    Extracts and probes the KV cache at the end of the thinking phase,
    before the first image generation token.
    """

    def extract_thinking_kv(
        self,
        model: PlanningTransformer,
        prompt_with_thinking: str,
        thinking_end_token: str,
    ) -> dict[str, Tensor]:
        """
        Runs the model on a prompt that includes thinking tokens.
        Stops at the thinking_end_token and returns the KV cache at that point.
        Returns {layer_name: (K_matrix, V_matrix)}.
        """
        ...

    def probe_kv_for_geometry(
        self,
        kv_cache: dict[str, Tensor],
        probe_trainer: ProbeTrainer,
        target_property: str,
        scene: SceneParams,
    ) -> dict[str, ProbeResult]:
        """
        Trains a linear probe on the KV cache values at each layer
        to predict the geometric parameter (size_ratio, margin_fraction).
        High R² in the KV cache = thinking phase encoded the geometric plan.
        """
        ...

    def ablate_thinking_tokens(
        self,
        model: PlanningTransformer,
        full_prompt: str,
        token_indices_to_ablate: list[int],
        verifier: VerifierPipeline,
        scene: SceneParams,
    ) -> dict[str, ErrorVector]:
        """
        Replaces specified thinking tokens with a null token.
        Runs generation and verifies the output.
        If geometric_calculation error rate increases after ablating
        specific thinking tokens, those tokens were causally carrying
        the geometric plan.
        Returns error vectors for the ablated vs. original run.
        """
        ...
```

---

## 6. Experiment Windows

### W1 — Probe Timing

```python
# experiments/w1_probe_timing.py

def run_w1(
    cfg: W1Config,
    model: PlanningTransformer,
    hook_manager: HookManager,
    probe_trainer: ProbeTrainer,
    logger: ExperimentLogger,
):
    """
    For each checkpoint in the training run:
      1. Run probe_all_layers for target_property='size_ratio'
         at token_position='pre_generation' and 'post_generation'.
      2. Log ProbeResult for every layer at both positions.
      3. Compare pre_generation accuracy vs post_generation accuracy.

    Key finding threshold:
      If any layer achieves test_r2 > 0.7 at pre_generation before
      the verifier detects the transition → planning circuit exists.
      If pre_generation accuracy only rises after the transition → circuit
      forms at the same time as the behavioral change.
      If pre_generation accuracy never exceeds baseline → no planning circuit;
      model does layout on the fly.

    Logged per checkpoint:
      - ProbeResult for every layer × position combination
      - The layer with highest pre_generation R² (the 'planning layer')
      - Δ R² between pre and post generation (planning specificity)
    """
    checkpoints = load_all_checkpoints(cfg.checkpoint_dir)

    for ckpt in checkpoints:
        model.load_state_dict(ckpt.state_dict)

        pre_gen_results  = probe_trainer.probe_all_layers(
            model, hook_manager, cfg.probe_dataset,
            target_property='size_ratio',
            token_position='pre_generation',
        )
        post_gen_results = probe_trainer.probe_all_layers(
            model, hook_manager, cfg.probe_dataset,
            target_property='size_ratio',
            token_position='post_generation',
        )

        logger.log_probe_sweep(ckpt.step, 'pre_gen',  pre_gen_results)
        logger.log_probe_sweep(ckpt.step, 'post_gen', post_gen_results)

        planning_layer = max(pre_gen_results, key=lambda l: pre_gen_results[l].test_r2)
        logger.log_scalar('w1/best_pre_gen_r2',  pre_gen_results[planning_layer].test_r2,  ckpt.step)
        logger.log_scalar('w1/best_post_gen_r2', post_gen_results[planning_layer].test_r2, ckpt.step)
```

### W2 — Curriculum Comparison

```python
# experiments/w2_curriculum.py

def run_w2(
    cfg: W2Config,
    curricula: dict[str, CurriculumScheduler],
    logger: ExperimentLogger,
):
    """
    Trains one model per curriculum, all else equal.
    At each eval step, runs the verifier and logs the full TaxonomyReport.
    At convergence, runs W1 probe sweep and logs probe profile for each curriculum.

    Key comparisons logged:
      - Time to planning circuit transition (steps until planning_signature_rate < 0.05)
      - Final OOD generalisation error (planning_signature_rate on test split)
      - Loss landscape sharpness at convergence (perturbation test)
      - Whether solutions are mode-connected (linear interpolation test)
    """
    results = {}

    for name, curriculum in curricula.items():
        model  = PlanningTransformer(cfg.model_cfg)
        trainer = Trainer(model, curriculum, cfg)
        trainer.train(cfg.n_steps)
        results[name] = trainer.history

        logger.log_curriculum_summary(name, results[name])

    # mode connectivity check between all curriculum pairs
    for (name_a, hist_a), (name_b, hist_b) in combinations(results.items(), 2):
        connectivity = test_mode_connectivity(
            hist_a.final_checkpoint, hist_b.final_checkpoint,
            steps=20, eval_fn=cfg.eval_fn,
        )
        logger.log_mode_connectivity(name_a, name_b, connectivity)
```

### W3 — Weight Transition Analysis

```python
# experiments/w3_weight_transition.py

def run_w3(
    cfg: W3Config,
    checkpoint_dir: Path,
    transition_step: int,       # from W7 / Trainer._detect_transition
    gradient_tracker: GradientTracker,
    weight_analyzer: WeightAnalyzer,
    attention_analyzer: AttentionAnalyzer,
    logger: ExperimentLogger,
):
    """
    Called automatically when the transition is detected during training.
    Analyses the checkpoint window around transition_step.

    Steps:
      1. Compute gradient profiles before and after transition.
         Find weights with largest gradient spike at transition.
      2. SVD trajectory for top-5 candidate weights through transition window.
      3. Attention pattern evolution for top cross-object heads through window.
      4. Compare effective rank and Frobenius norm before/after transition.

    This is the mechanistic cause of the planning circuit formation.
    """
    window_checkpoints = load_checkpoints_around(checkpoint_dir, transition_step, window=cfg.window)

    # Step 1: gradient spike detection
    grad_profiles = {}
    for ckpt in window_checkpoints:
        model.load_state_dict(ckpt.state_dict)
        grad_profiles[ckpt.step] = gradient_tracker.compute_per_weight_gradient_norms(
            model, cfg.composition_batch, loss_fn=cfg.loss_fn,
            filter_example_type='containment',
        )

    spike_weights = detect_gradient_spike(grad_profiles, transition_step, top_k=10)
    logger.log_spike_weights(spike_weights)

    # Step 2: SVD trajectory for candidate weights
    for weight_name in spike_weights[:5]:
        svd_traj = weight_analyzer.svd_trajectory(window_checkpoints, weight_name)
        logger.log_svd_trajectory(weight_name, svd_traj)

    # Step 3: attention pattern evolution
    cross_object_heads = identify_cross_object_heads(
        window_checkpoints[-1], attention_analyzer, cfg.sample_scene
    )
    for layer, head in cross_object_heads[:5]:
        attn_evolution = attention_analyzer.track_attention_evolution(
            window_checkpoints, cfg.sample_prompt, cfg.sample_scene, layer, head
        )
        logger.log_attention_evolution(layer, head, attn_evolution)

    # Step 4: rank and norm comparison
    pre_ckpt  = window_checkpoints[0]
    post_ckpt = window_checkpoints[-1]
    comparison = weight_analyzer.compare_memorised_vs_generalised(
        pre_ckpt, post_ckpt, model.get_layer_names()
    )
    logger.log_weight_comparison(comparison)
```

### W4 — Thinking Token Causality

```python
# experiments/w4_thinking.py

def run_w4(
    cfg: W4Config,
    thinking_model: PlanningTransformer,   # trained with thinking tokens enabled
    baseline_model: PlanningTransformer,   # same training without thinking tokens
    kv_analyzer: KVCacheAnalyzer,
    probe_trainer: ProbeTrainer,
    verifier: VerifierPipeline,
    logger: ExperimentLogger,
):
    """
    Three sub-experiments:

    W4a — KV probe: does the KV cache after thinking encode the size ratio?
      Run probe_kv_for_geometry on thinking_model's KV cache.
      Compare probe R² against baseline_model's KV cache at same layer.
      Higher R² in thinking model = thinking phase encoded the plan.

    W4b — Ablation: are specific thinking tokens causally necessary?
      For each thinking token position, ablate and re-verify.
      Tokens whose ablation specifically degrades geometric_calculation_error
      (but not shape_quality or attribute_binding) are causally carrying
      the geometric plan.

    W4c — Comparison: does thinking reduce specific error types?
      Compare TaxonomyReport of thinking_model vs baseline_model.
      Prediction: thinking reduces geometric_calculation errors specifically,
      not shape_quality errors (those are handled by the rendering circuit).
    """
    eval_set = cfg.eval_dataset

    # W4a
    for scene, prompt in eval_set[:cfg.n_probe_examples]:
        kv = kv_analyzer.extract_thinking_kv(thinking_model, prompt, cfg.thinking_end_token)
        probe_results = kv_analyzer.probe_kv_for_geometry(
            kv, probe_trainer, 'size_ratio', scene
        )
        logger.log_kv_probe_results(probe_results)

    # W4b
    ablation_results = {}
    for i, (scene, prompt) in enumerate(eval_set[:cfg.n_ablation_examples]):
        thinking_tokens = extract_thinking_token_positions(thinking_model, prompt)
        for tok_idx in thinking_tokens:
            ablated_errors = kv_analyzer.ablate_thinking_tokens(
                thinking_model, prompt, [tok_idx], verifier, scene
            )
            ablation_results[(i, tok_idx)] = ablated_errors
    logger.log_ablation_results(ablation_results)

    # W4c
    thinking_report = verifier.evaluate(thinking_model, step=0)
    baseline_report = verifier.evaluate(baseline_model, step=0)
    logger.log_model_comparison('thinking_vs_baseline', thinking_report, baseline_report)
```

### W5 — Invariance Tests

```python
# experiments/w5_invariance.py

INVARIANCE_AXES = {
    'shape_class':   dict(train_shapes=['circle', 'square'], test_shapes=['triangle', 'hexagon']),
    'color':         dict(train_colors=STANDARD_COLORS, test_colors=NOVEL_COLORS),
    'scale':         dict(train_size_range=(0.1, 0.4),    test_size_range=(0.5, 0.8)),
    'rotation':      dict(train_rotation=0,               test_rotation=[45, 90, 135]),
    'prompt_surface':dict(train_template='circle inside square',
                          test_templates=['square containing circle',
                                          'the circle is enclosed by the square',
                                          'a circle, enclosed within a square']),
    'text_transfer': dict(train_task='shape_containment', test_task='text_in_box'),
}

def run_w5(
    cfg: W5Config,
    model: PlanningTransformer,
    verifier: VerifierPipeline,
    attention_analyzer: AttentionAnalyzer,
    logger: ExperimentLogger,
):
    """
    For each invariance axis:
      1. Generate an eval set that varies only that axis.
      2. Run the verifier and log error rates (especially geometric_calculation).
      3. Capture attention patterns and check whether the same cross-object
         heads activate for the test axis setting as for the training setting.

    A circuit is invariant to an axis if:
      - Error rates are unchanged when the axis value changes.
      - The same set of cross-object heads activate.
    A circuit is NOT invariant if:
      - Error rates increase when the axis changes.
      - Different or fewer cross-object heads activate.

    Non-invariance is as informative as invariance:
    it tells you what the circuit's representation encodes that it shouldn't.
    """
    for axis_name, axis_cfg in INVARIANCE_AXES.items():
        train_eval = generate_eval_set(**axis_cfg['train_settings'])
        test_eval  = generate_eval_set(**axis_cfg['test_settings'])

        train_report = verifier.evaluate_on(model, train_eval, step=0)
        test_report  = verifier.evaluate_on(model, test_eval,  step=0)

        invariance_score = (
            1 - abs(train_report.error_rates['geometric_calculation']
                    - test_report.error_rates['geometric_calculation'])
        )
        logger.log_invariance(axis_name, invariance_score, train_report, test_report)

        # Attention pattern comparison
        train_attn = attention_analyzer.get_all_patterns(model, train_eval[0][1], train_eval[0][0])
        test_attn  = attention_analyzer.get_all_patterns(model, test_eval[0][1],  test_eval[0][0])
        attn_similarity = attention_analyzer.compare_to_theoretical_optimum(
            flatten_attention(train_attn), flatten_attention(test_attn)
        )
        logger.log_attention_invariance(axis_name, attn_similarity)
```

### W6 — Minimum Data Threshold

```python
# experiments/w6_threshold.py

def run_w6(
    cfg: W6Config,
    logger: ExperimentLogger,
):
    """
    Binary search for the minimum number of tight-containment examples
    needed for the planning circuit to form.

    Outer loop: binary search on N (number of tight examples in training set).
    Inner loop: train a model with N tight examples + M base examples.
    Signal: does the verifier detect the planning circuit transition?

    Also varies:
      - Tightness level (TIGHT vs EXACT) of the N special examples.
      - Whether invalid examples are included.
      - Explicitness level of the N special examples.

    This gives a 3D threshold surface:
      N_needed(tightness, has_invalid, explicitness)
    """
    lo, hi = 1, cfg.max_n_tight

    while lo < hi:
        mid = (lo + hi) // 2
        formed = train_and_detect_circuit(
            n_tight=mid,
            tightness=cfg.tightness,
            has_invalid=cfg.has_invalid,
            explicitness=cfg.explicitness,
            n_base=cfg.n_base,
            n_steps=cfg.n_steps,
            cfg=cfg,
        )
        logger.log_threshold_trial(mid, formed)
        if formed:
            hi = mid
        else:
            lo = mid + 1

    logger.log_threshold_result(cfg.tightness, cfg.explicitness, cfg.has_invalid, lo)
```

### W7 — Error Taxonomy Timeline

```python
# experiments/w7_taxonomy.py

def run_w7(
    cfg: W7Config,
    verifier_history: list[TaxonomyReport],
    logger: ExperimentLogger,
):
    """
    Post-training analysis of the full verifier history.
    Produces the circuit formation timeline.

    For each error type:
      - Find the step where the error rate first drops below threshold
        and stays below for cfg.stability_window consecutive evals.
      - This is the 'formation step' for the corresponding circuit.

    Output: an ordered list of (circuit_name, formation_step) pairs.
    The ordering is the circuit formation timeline.

    Also computes:
      - Whether any two error types have formation steps within window_size
        steps of each other → these circuits may be part of the same formation event.
      - Whether planning_circuit_signature (shape OK, containment fail)
        is a necessary intermediate state, and for how many steps it persists.
    """
    ERROR_THRESHOLD = 0.05   # below this = circuit formed
    STABILITY_WINDOW = 3     # must stay below threshold for this many evals

    formation_steps = {}
    for error_type in TaxonomyReport.error_type_names():
        rates = [r.error_rates[error_type] for r in verifier_history]
        formation_step = find_stable_drop(rates, ERROR_THRESHOLD, STABILITY_WINDOW)
        formation_steps[error_type] = formation_step

    ordered_timeline = sorted(
        [(name, step) for name, step in formation_steps.items() if step is not None],
        key=lambda x: x[1]
    )

    logger.log_formation_timeline(ordered_timeline)

    # Planning circuit signature duration
    signature_rates = [r.planning_signature_rate for r in verifier_history]
    signature_duration = sum(1 for r in signature_rates if r > ERROR_THRESHOLD)
    logger.log_scalar('w7/planning_signature_duration_steps', signature_duration)

    return ordered_timeline
```

---

## 7. Logging Strategy

Every metric is tagged with: `experiment_window`, `curriculum_name`, `step`, `split` (train/test/ood). Nothing is logged without all four tags.

```python
# logging/logger.py

class ExperimentLogger:
    """
    Wraps wandb for remote logging and local jsonl for offline analysis.
    Every log call writes to both. The local log is the source of truth;
    wandb is for interactive inspection.
    """

    def __init__(self, cfg: LogConfig, run_name: str):
        self.run_name  = run_name
        self.local_log = open(cfg.log_dir / f'{run_name}.jsonl', 'w')
        wandb.init(project=cfg.wandb_project, name=run_name, config=cfg.to_dict())

    # ── Step-level (every training step) ───────────────────────────────
    def log_step(self, step: int, loss: float, grad_norm: float, curriculum_name: str):
        self._log({'type': 'step', 'step': step, 'loss': loss,
                   'grad_norm': grad_norm, 'curriculum': curriculum_name})

    # ── Eval-level (every eval_every steps) ────────────────────────────
    def log_taxonomy_report(self, report: TaxonomyReport, split: str):
        """Logs all error rates + planning signature rate."""
        payload = {'type': 'taxonomy', 'step': report.step, 'split': split,
                   **report.error_rates,
                   'planning_signature_rate': report.planning_signature_rate}
        self._log(payload)
        wandb.log({f'taxonomy/{split}/{k}': v for k, v in payload.items()},
                  step=report.step)

    # ── Probe results ───────────────────────────────────────────────────
    def log_probe_sweep(self, step: int, position: str, results: dict[str, ProbeResult]):
        for layer, result in results.items():
            self._log({'type': 'probe', 'step': step, 'position': position,
                       'layer': layer, 'test_r2': result.test_r2,
                       'is_pre_generation': result.is_pre_generation})
        # W&B: heatmap of R² across layers at this step
        wandb.log({f'probe/{position}/{layer}': r.test_r2 for layer, r in results.items()},
                  step=step)

    # ── Weight analysis ─────────────────────────────────────────────────
    def log_svd_trajectory(self, weight_name: str, trajectory: list[SVDSnapshot]):
        for snap in trajectory:
            self._log({'type': 'svd', 'step': snap.step, 'weight': weight_name,
                       'effective_rank': snap.effective_rank,
                       'frobenius_norm': snap.frobenius_norm,
                       'top_sv': snap.singular_values[:5].tolist()})

    # ── Transition event ────────────────────────────────────────────────
    def log_transition_detected(self, step: int, curriculum: str):
        self._log({'type': 'transition', 'step': step, 'curriculum': curriculum})
        wandb.alert(title='Planning circuit transition detected',
                    text=f'Curriculum: {curriculum}, step: {step}')

    # ── Spike weights ────────────────────────────────────────────────────
    def log_spike_weights(self, spike_weights: list[tuple[str, float]]):
        """Log the top-K weights by gradient spike magnitude at transition."""
        for rank, (name, magnitude) in enumerate(spike_weights):
            self._log({'type': 'spike_weight', 'rank': rank,
                       'weight_name': name, 'spike_magnitude': magnitude})

    # ── Formation timeline ───────────────────────────────────────────────
    def log_formation_timeline(self, timeline: list[tuple[str, int]]):
        for order, (circuit_name, step) in enumerate(timeline):
            self._log({'type': 'formation', 'circuit': circuit_name,
                       'formation_step': step, 'formation_order': order})

    # ── Invariance ───────────────────────────────────────────────────────
    def log_invariance(self, axis: str, score: float,
                       train_report: TaxonomyReport, test_report: TaxonomyReport):
        self._log({'type': 'invariance', 'axis': axis, 'invariance_score': score,
                   'train_geometric_error': train_report.error_rates['geometric_calculation'],
                   'test_geometric_error':  test_report.error_rates['geometric_calculation']})

    # ── Internal ─────────────────────────────────────────────────────────
    def _log(self, payload: dict):
        payload['run'] = self.run_name
        print(json.dumps(payload), file=self.local_log, flush=True)
```

### What is logged at each level

| Frequency | What |
|---|---|
| Every training step | loss, grad_norm per layer, curriculum_name, step |
| Every eval step | Full TaxonomyReport (all error rates), planning_signature_rate |
| Every eval step | Probe R² for all layers × positions |
| Every eval step | Top cross-object attention head activations |
| Every eval step | Weight norms and effective ranks for candidate layers |
| On transition detection | W3 full analysis: gradient spike, SVD trajectory, attention evolution |
| At convergence | Loss landscape sharpness, mode connectivity if W2 |
| W4 only | KV probe results, ablation results |
| W5 only | Invariance score per axis, attention pattern similarity |
| W6 only | Threshold trial result (N, formed: bool) |
| W7 post-hoc | Formation timeline, signature duration |

---

## 8. Configuration System

Every experiment is fully specified by a yaml config. Nothing is hardcoded.

```yaml
# configs/experiments/w1_probe_timing.yaml

experiment: w1_probe_timing
run_name: "w1_pure_implicit_seed42"

model:
  n_layers: 6
  d_model: 256
  n_heads: 8
  d_mlp: 1024

training:
  n_steps: 50000
  batch_size: 32
  lr: 3e-4
  weight_decay: 0.1
  eval_every: 500
  checkpoint_every: 1000

curriculum:
  type: PureImplicit
  tightness_distribution: {LOOSE: 0.4, MEDIUM: 0.3, TIGHT: 0.2, EXACT: 0.1}

data:
  explicitness_level: RELATION   # Level 1
  complexity: TWO_GEOMETRIC
  n_train: 50000
  n_probe: 5000     # separate probe dataset, never seen during training
  n_eval: 2000
  canvas_size: [256, 256]
  rng_seed: 42

split:
  type: StructuredOODSplit
  train_tightness: [LOOSE, MEDIUM]
  test_tightness: [TIGHT, EXACT]

mech_interp:
  probe_target: size_ratio
  probe_positions: [pre_generation, post_generation]
  probe_n_epochs: 200
  attention_cross_object: true
  weight_tracking: true
  candidate_weights_top_k: 10

logging:
  wandb_project: planning_circuit
  log_dir: ./logs/w1
  artefact_every: 5000    # save attention map images every N steps
```

---

## 9. Run Entry Point

```python
# run.py

def main(config_path: str):
    cfg = OmegaConf.load(config_path)

    # Reproducibility
    set_seed(cfg.data.rng_seed)
    log_git_hash(cfg.logging.log_dir)   # log exact code version

    # Build components
    generator   = SceneGenerator(cfg.data.rng_seed, cfg.data.canvas_size)
    prompt_bld  = PromptBuilder()
    curriculum  = build_curriculum(cfg.curriculum)
    split       = build_split(cfg.split)

    model       = PlanningTransformer(cfg.model)
    hook_mgr    = HookManager(model)
    hook_mgr.register_all()

    verifier    = VerifierPipeline(build_eval_set(generator, prompt_bld, cfg))
    mech_suite  = MechInterpSuite(cfg.mech_interp, model, hook_mgr)
    logger      = ExperimentLogger(cfg.logging, cfg.run_name)

    trainer = Trainer(model, curriculum, verifier, mech_suite, logger, cfg.training)

    # Dispatch to experiment window
    experiment_fn = {
        'w1_probe_timing':      run_w1,
        'w2_curriculum':        run_w2,
        'w3_weight_transition': run_w3,
        'w4_thinking':          run_w4,
        'w5_invariance':        run_w5,
        'w6_threshold':         run_w6,
        'w7_taxonomy':          run_w7,
    }[cfg.experiment]

    experiment_fn(cfg, trainer, logger)


if __name__ == '__main__':
    main(sys.argv[1])
```

---

## 10. Execution Order and Dependencies

```
W7  ←── runs during every training run (verifier timeline is always on)
 │
 ├── W2 (curriculum comparison) trains multiple runs in parallel, all feed W7
 │
 ├── W3 (weight transition) is triggered automatically when W7 detects transition
 │
 ├── W1 (probe timing) runs on checkpoints from any W2 curriculum run
 │
 ├── W6 (threshold) uses the W2 runs to determine which curriculum to use,
 │       then binary-searches N on that curriculum
 │
 ├── W5 (invariance) runs on the converged model from the best W2 curriculum
 │
 └── W4 (thinking) trains a parallel thinking-token variant and compares
         its W7 timeline and W1 probe profile against the best W2 model
```

Practically: start W2 (all curricula in parallel), W7 runs inside each. When the first transition is detected, W3 fires automatically. After W2 converges, run W1, W5, W6 on the best-performing curriculum's model. Run W4 as a separate training job in parallel with W2.

---

## 11. The Single Result This Plan Produces

After all seven windows are run and their logs are analysed, the output is one structured claim:

> "Under curriculum X, a planning circuit forms at step T. It is located at layers L1–L3, specifically in attention heads H4 and H7 at those layers. It is causally verified by activation patching at those positions. It encodes the size ratio with R²=0.82 pre-generation. It is shape-invariant (score 0.91) and scale-invariant (score 0.87) but not color-invariant (score 0.44). The minimum data to form it is N tight examples. The thinking token variant reduces geometric_calculation errors by 34% and the KV cache at the thinking-end token achieves R²=0.91 for size ratio at layer L2, confirmed causal by ablation of thinking tokens 14–17."

Every number in that claim is a logged artefact. Every artefact is tied to a specific config file. The experiment is fully reproducible.
