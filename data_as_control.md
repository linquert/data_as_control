# Data as Control: A Framework for Understanding How Transformers Learn

> The central claim of this document is simple: if you design the training data carefully enough, the data itself becomes a scientific instrument. By controlling what a model sees, in what order, in what proportions, and under what splits, you can run controlled experiments on the internals of a neural network the same way a scientist controls variables in a physical experiment. The model's weights, circuits, and representations become something you can probe, predict, and verify — not just observe after the fact.

---

## 1. The Core Idea

Standard machine learning research trains a model on a dataset and then evaluates what it learned. The dataset is treated as fixed background — a resource to be consumed, not a variable to be manipulated. Interpretability research then tries to reverse-engineer the trained model, asking what it learned and how.

This framework inverts that process.

Instead of training on existing data and asking what the model learned, we **design the data to test a specific hypothesis about how the model learns**. The training data is not background — it is the independent variable. The model's weights, circuits, attention patterns, and generalization behavior are the dependent variables. Every design choice about the dataset — its structure, its splits, its curriculum order, its mixture ratios, its coverage of edge cases — is a deliberate experimental manipulation with a predicted effect on what the model learns internally.

The key enabling property is **ground truth over the generation process**. When data is generated procedurally by an algorithm, you know exactly what rule produced each example. This means:

- The correct output is always known, so every generated image or output can be verified automatically.
- The internal algorithm the model *should* learn is derivable from first principles.
- Errors can be classified not just as "wrong" but as *specifically wrong in a way that reveals which capability is missing*.
- The training distribution can be parametrized exactly, so you can vary one dimension while holding all others fixed.

This is not possible with naturalistic data. When you train on scraped images or web text, you do not control what structures are present, how often they appear, in what combinations, or whether the data even contains enough signal for a specific capability to form. You can observe what the model learned, but you cannot run controlled experiments on *why* it learned that and not something else.

With procedural synthetic data, you can.

---

## 2. Why Verifiable Output Is the Foundation

The most powerful property of this framework is that **every output can be checked by the same algorithm that generated it**.

For language model experiments, if the dataset is arithmetic traces, the verifier is just an arithmetic evaluator. For sorting traces, the verifier checks whether each swap is valid and whether the final array is sorted. For a register machine execution trace, the verifier replays each instruction and checks whether the model's trace matches the correct execution.

For vision-language model experiments, if the dataset is procedurally generated geometric scenes, the verifier checks whether the rendered objects satisfy the stated spatial relations, whether their sizes are correct, whether their colors match the prompt, and whether the composition is geometrically valid.

This is a fundamentally different situation from standard image generation evaluation, where you compare generated images to human judgments or to a reference image using a perceptual metric. Those evaluations give you one number: how good is the output. A verifier over structured output gives you **a decomposed error signal**: which specific capability failed, and by how much.

### The error taxonomy idea

Because you know the ground truth completely, you can classify every error the model makes into a specific type. Each error type corresponds to a different internal capability, and therefore to a different circuit. Tracking the prevalence of each error type over training time gives you a circuit formation timeline without ever looking at weights.

For geometric scene generation, the error taxonomy looks like this:

| Error type | What it means | Circuit implicated |
|---|---|---|
| Shape quality failure | Object is not rendered correctly | Primitive rendering circuit |
| Attribute binding failure | Right shapes, wrong attributes on each | Color/attribute binding circuit |
| Compositional inversion | "Circle inside square" → square inside circle | Relational direction circuit |
| Scale inconsistency | Containment correct, size ratio wrong | Relative magnitude circuit |
| Geometric calculation failure | Model does not check if configuration is achievable | Planning circuit |
| Cardinality error | Wrong number of objects | Counting circuit |
| Constraint satisfaction failure | Constraint stated, entirely ignored | Constraint activation circuit |
| Preference error | Valid output, but tightness calibration wrong | Prior calibration |

For arithmetic trace generation, the error taxonomy looks like this:

| Error type | What it means | Circuit implicated |
|---|---|---|
| Single digit wrong, no carry | Per-digit addition rule not learned | Basic digit circuit |
| Carry propagation wrong | Carry circuit not formed | Carry circuit |
| Carry in digit N, not digit M | Position-specific, not general carry | Position-general carry circuit |
| Result correct, intermediate wrong | Backward-engineered rather than computed | Step-by-step algorithm circuit |
| Correct up to K digits, wrong after | Partial algorithm, not compositional | Multi-step composition circuit |

The behavioral delta vector is a sharper version of the error taxonomy timeline.
Rather than measuring error type prevalences at each checkpoint, measure the *change* in
each error type prevalence between two consecutive checkpoints — the vector:

    Δ[error_type_k] = prevalence_after_update - prevalence_before_update

This delta vector is the semantic fingerprint of a single training update. An update that
decreases all error types proportionally corresponds to general improvement. An update
that decreases only one error type — say, routing failures on rare neighborhoods — while
leaving others unchanged is a localized, pattern-specific update. An update that decreases
routing failures while increasing computation failures is a circuit-reorganization event:
the model improved its attention structure at the cost of temporary disruption to the
downstream computation.

The delta vector also captures interference: when improving one error type causes
regression in another. This is common early in training, when circuits compete for weight
capacity. If improving containment errors increases attribute binding errors, the two
circuits share weight structure. If they are independent, improvements do not interfere.
Tracking the cross-error-type correlation matrix of delta vectors over training reveals
whether circuits are developing modularly or entangled.

When you plot each error type as a separate curve over training time, different curves fall to zero at different times. Each fall corresponds to a circuit forming. This is a **circuit formation timeline** — a map of when each capability emerged, derived entirely from behavioral outputs, with no weight inspection required.

---

## 3. Dataset Families

### 3.1 Arithmetic and algebraic structures (language model)

Arithmetic is the most studied synthetic domain in mechanistic interpretability. Its power comes from the fact that it has clean mathematical structure that allows principled splits.

**Multi-digit addition with carry splits.** The key structural property is that some digit positions require a carry operation and some do not. This creates a natural OOD split:

- Train on examples where no digit pair requires a carry.
- Test on examples where at least one digit pair does.

*Question:* Did the model learn addition as a reusable algorithm, or only a no-carry digit rule?

Then vary the mixture:
- Train 95% no-carry, 5% carry. What is the minimum carry exposure for the carry circuit to form?
- Train carry only in the ones digit. Test carry in the tens digit. Is the carry circuit position-general or position-specific?

**Group theory splits.** The integers mod N under addition form a group. The group has subgroups — for example, the even integers under addition mod 10, or the multiples of 3. You can train on elements from one subgroup and test on elements outside it. If the model learned the group operation as a general rule, it should generalize. If it memorized the subgroup's behavior specifically, it will fail on elements outside the subgroup.

**Composed operations.** Train on single binary operations (a + b). Then introduce composed operations (a + b + c, or a + b × c). Does learning the composed operation improve performance on the single operation, or does it require the single operation to be already learned? The order of introduction reveals the dependency structure of the circuits.

### 3.2 Algorithm execution traces (language model)

Algorithm traces represent a step-by-step execution of a known algorithm. The model learns to produce the correct next step given the current state. The verifier checks each step independently, so you get per-step error classification.

**Sorting traces.** Input: an unsorted array. Output: a sequence of comparison-and-swap steps followed by the sorted array.

```
Input:  [3, 1, 4, 1, 5]
Step 1: compare 3,1 → swap → [1, 3, 4, 1, 5]
Step 2: compare 3,4 → no swap → [1, 3, 4, 1, 5]
Step 3: compare 4,1 → swap → [1, 3, 1, 4, 5]
...
Output: [1, 1, 3, 4, 5]
```

The verifier checks: is each swap decision correct given the current array state? This tells you whether the model failed at comparison, at swap execution, or at termination — three different circuits.

**Register machine traces.** A two-register machine with operations INC A, INC B, SWAP, ADD A B. The trace is an execution log.

```
State: A=2, B=3
INC A   → A=3, B=3
ADD A B → A=6, B=3
SWAP    → A=3, B=6
```

The verifier replays each instruction. If the model makes an error on INC but not on ADD, the circuits for those operations are separable — and you can test whether they share weight structure or use independent circuits.

**Graph reachability.** A small graph (4–6 nodes) with a BFS trace showing which nodes are discovered in which order.

```
Graph: A→B, A→C, B→D
BFS from A:
Queue: [A]
Visit A → Queue: [B, C]
Visit B → Queue: [C, D]
Visit C → Queue: [D]
Visit D → Queue: []
```

The verifier checks each queue state. Errors in queue management vs. errors in traversal order correspond to different algorithmic sub-circuits.

**Cellular automata traces.** A 1D cellular automaton (e.g., Rule 110) with N cells and
T time steps. The model learns to predict the next state from the current state.

    t=0: 0 1 1 0 1 0 1 1
    t=1: 0 1 1 1 1 1 1 0
    t=2: 0 1 0 0 0 0 1 0

What makes cellular automata particularly powerful as a dataset is that you can derive
exactly what the attention pattern *should* look like for a model that has learned the
rule: each output cell should attend to exactly the three input cells that determine it
(left, center, right). This gives you not just a behavioral ground truth but an internal
ground truth — a theoretically optimal circuit to compare against.

The verifier decomposes errors into a three-part taxonomy mirroring the computational
structure of the rule: routing failures (the model attends to the wrong cells), payload
failures (the model attends to the right cells but does not extract the cell state
cleanly), and computation failures (the model has correct local information but applies
the wrong Boolean function). Each failure type leaves a distinct signature: routing
failure shows up as low QK attention mass on the correct neighbor offsets; payload failure
shows up as poor linear probe accuracy on value vectors despite correct attention;
computation failure shows up as neighborhood-specific output errors with correct attention
and value representations.

For structured splits, CA is unusually productive:

- *Interior-only training, boundary-position test.* Train on cells at positions 2 through
  L-2, where every cell has three real neighbors. Test on positions 0 and L-1, where one
  neighbor is a padded value. If the interior rule generalizes to boundary handling, a
  single shared circuit exists. If not, boundary handling is a separate circuit that was
  never built.

- *Texture-invariant split.* Train on rows with at most 2 state transitions (smooth,
  mostly-constant regions). Test on rows with 6 or more transitions. Both sets contain the
  same 8 local neighborhoods, but their statistical character differs completely. Accuracy
  drop on the high-transition test set reveals that the learned circuit is entangled with
  row texture rather than computing the local rule independently.

- *Minimum rare-neighborhood threshold.* Train with 99% smooth rows (dominated by 000 and
  111 neighborhoods) and 1% of everything else. How many rare-neighborhood examples does
  the model need before it correctly predicts their outputs? Vary the proportion from 0.1%
  to 50% and plot the threshold curve. A model with an abstract rule should require far
  fewer rare examples than a frequency-weighted pattern cache. The curve shape is itself
  diagnostic: an abrupt transition at some threshold indicates rule formation; a smooth
  improvement indicates probabilistic interpolation.

- *Position-range split.* Train on cells at positions 4 through 12 of a 16-cell row. Test
  on positions 0 through 3 and 13 through 15. This directly tests whether the local-rule
  circuit is position-general or position-specific. A model that learned relative-offset
  attention should generalize; one that memorized absolute cell positions should not.
  
**Formal languages.** A language defined by a grammar with known rules (e.g., a Dyck language of balanced brackets, or a language with alternating symbols). The model learns to produce valid strings or to classify strings as valid or invalid.

```
Grammar: S → ε | (S) | SS
Valid:   (()(()))
Invalid: (()
```

Formal languages let you control exactly what syntactic properties the model must track. You can make the grammar as simple or as complex as you want, add or remove specific rules, and test whether the model has learned the grammar as a whole or just its most frequent patterns.

### 3.3 Geometric scene generation (vision-language model)

The VLM dataset is procedurally generated images of geometric scenes, each paired with an NLP prompt that describes the scene. The prompt is generated by the same algorithm that drew the image — so the prompt is always accurate and the verifier can check every property.

**Primitive scenes.** Single objects with specified attributes.

```
Prompt: "A red circle, radius 40, centered at (200, 150)"
Image:  a single red circle at the specified position
Verifier checks: is there exactly one circle? is it red? is its radius within tolerance of 40? is its center within tolerance of (200, 150)?
```

**Compositional scenes.** Multiple objects with spatial relations.

```
Prompt: "A blue circle inside a yellow square"
Image:  a yellow square with a blue circle contained within it
Verifier checks: shape quality, attribute binding, containment (every point on circle's boundary is inside square boundary), size consistency (circle radius < half of square side)
```

**Arrangement scenes.** Objects in structured arrangements.

```
Prompt: "Five circles arranged in a ring, equally spaced, alternating red and blue"
Image:  five circles at equal angular intervals, colors alternating
Verifier checks: count (exactly 5), arrangement (angular intervals within tolerance), color alternation, approximate equal spacing
```

**Text rendering scenes.** Text with specified font, size, alignment, and position.

```
Prompt: "The word HELLO in Arial 24pt, centered horizontally in a red rectangle"
Image:  a red rectangle with HELLO centered within it
Verifier checks: text content, font identity, approximate size, horizontal centering (text bounding box center-x ≈ rectangle center-x)
```

**Edit sequences.** A before-image, an instruction, and an after-image.

```
Before: "A red circle at top-left, a blue square at bottom-right"
Instruction: "Move the red circle to the center"
After:  "A red circle at center, a blue square at bottom-right"
Verifier checks: was the red circle moved? did its attributes change? did the blue square remain unchanged?
```

**Multi-image composition.** Two or more source images and a compositional instruction.

```
Image A: a red circle and a green triangle
Image B: objects arranged in a 2x2 grid
Instruction: "Arrange the objects from image A in the pattern of image B"
Output: red circle and green triangle in a 2x2 grid arrangement
Verifier checks: correct objects (from A), correct arrangement pattern (from B)
```

### 3.4 State machine and toggle datasets (language model)

A set of objects, each with binary state (on/off), and a set of rules for how toggling one object affects others. Starting from an initial configuration, the model must produce a sequence of toggle operations that reaches a target configuration.

```
Objects: A, B, C
Rules: toggling A also toggles B; toggling C is independent
Initial: A=on, B=on, C=off
Target:  A=off, B=off, C=on
Solution: toggle A (→ A=off, B=off), toggle C (→ C=on)
```

The verifier replays each toggle and checks whether the final state matches the target. Errors in rule application vs. errors in planning the sequence reveal whether the model learned the rules or the planning algorithm.

---

## 4. The Control Axes

The power of synthetic data comes from having multiple independent dimensions that you can vary. We call these **axes**. Each axis is a parametric dial. You can hold all axes fixed and vary one, or vary two to study their interaction.

### Axis 1: Constraint explicitness

How much information does the prompt give about the rule the output must satisfy?

| Level | Example prompt | What the model must do |
|---|---|---|
| 0 | "A circle and a square" | No constraint to satisfy |
| 1 | "A circle inside a square" | Satisfy containment, figure out how |
| 2 | "A small circle inside a large square" | Qualitative size constraint |
| 3 | "A circle half the width of the square, placed inside it" | Relative measurement |
| 4 | "A circle radius 20, square side 60, circle inside square" | Absolute measurement |
| 5 | "Circle r=20, square s=60. Since 20 < 30 and center (30,30) is 30px from each edge which exceeds r=20, containment holds. Render." | Explicit proof + render |

All six levels can produce the same image. The prompt changes; the target does not. This lets you train at one level and test at another, revealing exactly what the model extracts from each level of explicitness.

### Axis 2: Composition complexity

How many objects and relations does the scene involve?

| Level | Description | Example |
|---|---|---|
| 0 | Single object, no relation | One red circle |
| 1 | Two objects, no constraint | A circle and a square |
| 2 | Two objects, positional relation | A circle left of a square |
| 3 | Two objects, geometric constraint | A circle inside a square |
| 4 | Three objects, chain of relations | A circle inside a square, the square left of a triangle |
| 5 | N objects, hierarchical constraints | Three circles in decreasing radius, all inside a rectangle |

### Axis 3: Constraint tightness

How much margin does the valid configuration have?

| Level | Example | Margin |
|---|---|---|
| Loose | Circle r=10 inside square s=80 | 70px |
| Medium | Circle r=25 inside square s=60 | 5px |
| Tight | Circle r=29 inside square s=60 | 1px |
| Exact | Circle r=30 inside square s=60 | 0px, touches all sides |
| Invalid | Circle r=35 inside square s=60 | Impossible, labeled as invalid |

Tight examples are the equivalent of carry digits in addition — they require the model to compute the constraint precisely rather than approximately. A model that only learned from loose examples will fail on tight ones even if the relation type is the same.

### Axis 4: Freedom

How much choice does the model have over unspecified parameters?

| Level | Description | Example |
|---|---|---|
| Fully constrained | All parameters given | "Circle radius 20, square side 60, circle inside square" |
| Range constrained | Parameters given as intervals | "Square side between 40 and 80, circle inside square" |
| Partially free | Relation given, sizes free | "Circle inside square" |
| Fully free | No relation, no constraint | "A circle and a square" |

The freedom axis is particularly interesting because it lets you study the model's **implicit prior**: when given a valid range, what does the model prefer to generate? This prior is shaped by the training distribution and reveals what the model has internalized about typical configurations.

### Axis 5: Rendering/trace format (language model)

How is the output structured — final answer only, or step-by-step trace?

| Level | Description | Verifier capability |
|---|---|---|
| Final output only | Just the answer or image | Binary correct/incorrect |
| Partial trace | Key intermediate steps | Per-step error classification |
| Full trace | Every computational step | Error localization to exact step |
| Annotated trace | Trace with explicit reasoning at each step | Separates computation from planning |
| Proof + trace | Mathematical justification before execution | Tests whether justification affects execution |

---

## 5. Dataset Splits as Experiments

The split between training and test data is not just a holdout strategy — it is the primary experimental manipulation. Different splits test different hypotheses about what the model has learned.

### The carry-digit principle

The most powerful splits are **structured OOD splits** — the test set is not random examples from the same distribution but examples that require a capability not directly present in training. The test performance then answers a binary question: did the model learn the general rule, or only the specific instances it saw?

The carry-digit split in addition is the prototype. Train on all examples that require no carry. Test on examples that require carry. The model that learned "addition" should handle both; the model that learned "no-carry addition" handles only training. The gap between the two is the carry circuit — and you can measure exactly how much carry-example exposure closes that gap.

This principle extends to every dataset:

**In sorting:** Train on arrays of length ≤ 4. Test on length 5. Did the model learn a sorting algorithm (which is length-general) or a length-4 specific rule?

**In geometric scenes:** Train on containment (circle inside square). Test on a different relation (circle above square). Did the model learn spatial relations in general, or only containment?

**In graph traversal:** Train on graphs with 4 nodes. Test on graphs with 5 nodes. Did the model learn BFS as an algorithm, or a 4-node-specific procedure?

**In cellular automata:** Train on Rule 110. Test on Rule 30. Did the model learn "how to apply a local rule" in general, or the specific rule it trained on?

### The subgroup split

Algebraic structures have **subgroup splits** — you can partition the training data so that the test set contains only elements from a specific coset that was absent from training.

For addition mod 10, the even numbers form a subgroup. Train only on pairs where both operands are odd or both are even. Test on mixed parity pairs. If the model learned the group operation as a rule, it generalizes. If it learned parity-specific patterns, it fails.

This kind of split is extraordinarily clean because you can **prove** that the training set contains no examples of the test distribution, yet the correct answer for test examples is still determined by the same rule the model was trained on.

### The axis-crossing split

Choose two axes. Train on one setting of each. Test on combinations of axis values not seen during training.

**Example:** Train on (Level 1 explicitness, Loose tightness). Test on (Level 1 explicitness, Tight tightness). Did the model learn the containment rule well enough to apply it precisely, or only approximately?

**Example:** Train on (composition complexity 3, relation = containment). Test on (composition complexity 3, relation = ordering). Did the model learn "how to handle a geometric constraint between two objects" in general, or "how to handle containment" specifically?

**Example:** Train on (no carry, any digit position). Test on (carry, ones digit). Then separately test on (carry, tens digit). Is the carry circuit position-general or position-specific?

### The disjoint-vocabulary split

Train on object type A and object type B separately. Test on compositions involving both. For example:

- Train on circles only (many single-circle examples, many circle-arrangement examples)
- Train on squares only (many single-square examples, many square-arrangement examples)
- Test on circle-inside-square

Did the model learn circle-rendering and square-rendering as independent circuits that can be combined, or are the circuits entangled with the specific training compositions? If it succeeds on the test, the rendering circuits are compositional. If it fails, they are entangled.

### The data-quantity threshold split

Binary-search for the minimum number of examples of a specific type needed for a capability to emerge. 

- Train with N=1 tight-containment example alongside many loose examples. Does tight-containment generalize?
- Double N and repeat. Plot the grokking event timing vs. N.
- The resulting curve — how many examples of the hard type are needed, and how quickly does the capability emerge — is a characterization of how difficult it is to form the corresponding circuit.

This is the answer to: "How much carry evidence is needed for SGD to form a carry circuit?"

---

## 6. The Verifier as a Circuit Formation Tracker

Once you have error taxonomy categories and a verifier that classifies every output into them, you can produce a **circuit formation timeline** from pure behavioral data, with no weight inspection at all.

The procedure is simple:

1. Train the model, saving a checkpoint every K steps.
2. At each checkpoint, generate a fixed evaluation set and run the verifier.
3. Record the prevalence of each error type at each checkpoint.
4. Plot each error type as a curve over training time.

The prediction: different error types decrease at different times. Each time a curve falls to near-zero, a circuit has formed. The ordering of the falls is itself a finding — it tells you which circuits are prerequisites for others.

### Example: Geometric planning circuit timeline

A plausible timeline for a model learning geometric scene generation:

```
Steps 0-1k:   all error types high
Steps 1k-3k:  shape quality errors fall → primitive rendering circuit forms
Steps 3k-5k:  attribute binding errors fall → color/shape binding circuit forms
Steps 5k-12k: positional relation errors fall → spatial relation circuit forms
Steps 12k-20k: geometric calculation errors fall → planning circuit forms
Steps 20k-25k: cleanup of memorization errors → circuit consolidation
```

Each step in this timeline corresponds to a specific circuit. When you see the geometric calculation errors begin falling (steps 12k in the example), you know the planning circuit is forming at that moment. You then go to the weight checkpoints at steps 12k–20k and look at what changed. The weight update that caused the planning circuit's formation is the most important mechanistic finding in the experiment.

### Example: Arithmetic trace timeline

```
Steps 0-500:   all digit errors high, many carry errors
Steps 500-2k:  no-carry digit errors fall → per-digit circuit forms
Steps 2k-8k:   no improvement on carry errors → carry circuit not forming
Steps 8k:      grokking event → carry errors drop rapidly
Steps 8k-15k:  carry in ones digit errors fall before carry in tens digit errors
Steps 15k+:    position-general carry → tens-digit carry errors fall
```

The gap between no-carry grokking (step 2k) and carry grokking (step 8k) is the cost of forming the carry circuit. The gap between ones-digit carry and tens-digit carry (steps 8k to 15k) tells you whether the carry circuit is position-specific or needs to generalize across positions.

---

## 7. Curriculum as SGD Path Surgery

The order in which examples are presented is not a neutral scheduling choice — it determines which loss surface the optimizer traverses, which circuits are initialized when others begin forming, and whether circuits are available to scaffold each other.

### Easy-to-hard and its inversion

**Easy-to-hard:** Train on single objects, then simple compositions, then complex compositions. Each stage provides weight initialization for the next. The circuits built for simpler tasks are available when harder tasks are introduced.

**Hard-to-easy:** Train on complex compositions immediately. The gradient signal is noisy and the loss landscape is rough. But when simple tasks are introduced later, the model already has some compositional structure and may reach simpler-task circuits faster.

**Prediction:** Easy-to-hard reaches grokking faster for the easy tasks but may produce more fragile circuits for the hard tasks (because the model found shortcuts in the easy regime that are inconsistent with the hard regime). Hard-to-easy is slower but may produce more robust circuits. Testing both and comparing the error taxonomy at convergence tells you which prediction is correct.

### Interleaved curricula

Rather than presenting task types in sequence, interleave them. At each batch, mix examples from multiple levels of the explicitness axis, multiple levels of the complexity axis, and multiple tightness levels. 

Interleaving prevents the model from committing to a circuit optimized for one sub-distribution before seeing the broader distribution. The tradeoff is that the gradient signal is noisier and grokking may be delayed. Whether interleaving produces better final generalization than sequential curricula is an empirical question your setup can answer directly.

### Tight-examples-first

For the tightness axis, train on tight examples (small margin) before loose ones. Tight examples provide the strongest gradient signal for the planning circuit because they require precise geometric computation — approximate reasoning fails on tight examples in a way it does not fail on loose ones. Once the tight-example planning circuit is formed, loose examples are easy to generalize to. The reverse (loose-first, tight later) may result in a planning circuit that handles the average case but fails at the margin.

### The curriculum-as-proof strategy

Design the curriculum so that each stage provides evidence the model needs to reason about the next stage. For the planning circuit:

Stage 1: Explicit proof traces (geometric calculation shown in full)
Stage 2: Measurement-only prompts (model must apply learned calculation)
Stage 3: Qualitative prompts (model must infer measurements from qualitative descriptions)
Stage 4: Relation-only prompts (model must compute everything from the relation alone)

This is analogous to teaching a student: first show the derivation, then give them problems with the formula, then give them problems where they must derive the formula, then give them problems with no formula hints. Each stage scaffolds the next.

### Measuring curriculum effects

For every curriculum design, measure:

- **Time to grokking** for each error type in the taxonomy
- **Final error rates** on the OOD structural generalization test
- **Weight geometry at convergence** (effective rank, singular value distribution)
- **Sharpness of the loss minimum** (perturbation test: add noise to weights, measure loss increase)

The curriculum that produces the flattest minimum, the lowest OOD error, and the first grokking of each error type is the curriculum that most efficiently builds the circuits you want.

---

## 8. The Planning Circuit: A Worked Example

The planning circuit question is the most concrete illustration of this entire framework. We use it here as an anchor.

### The question

When a model generates "circle inside square" without any measurements given, does it compute a valid size ratio and placement before generating the first image token — or does it pattern-match to training examples and generate something that looks plausible without performing geometric reasoning?

This is asking: does the model have an internal algorithm that checks geometric feasibility, or does it have a lookup table dressed up as an algorithm?

### The experiment

**Dataset construction:** Generate scenes at all levels of the explicitness axis (Level 0 through Level 5) and all levels of the tightness axis (loose through invalid). All scenes at Level 1 through Level 5 that depict containment are geometrically valid. Invalid examples are labeled as such.

**Training conditions:** Run six parallel training runs, one for each curriculum design from Section 7. All other variables (model architecture, optimizer, batch size, random seed) are held constant.

**Verifier setup:** At every checkpoint, run the verifier on a fixed evaluation set of 1,000 scenes. Record: shape quality, attribute binding, geometric relation, tightness margin of valid outputs, and specifically whether any output violates containment while having correct individual object rendering (the "composition fail" signature of an absent planning circuit).

**Probe setup:** At every checkpoint, for every layer in the model, train a linear probe on the intermediate activations using prompts where containment is specified implicitly (Level 1). The probe predicts the size ratio (circle radius / square side) of the scene the model will generate. Probe accuracy at each layer tells you where in the network the geometric parameter is being computed.

**Prediction:** When the planning circuit is absent, the probe should have low accuracy at all layers. When the planning circuit forms, probe accuracy should jump at the layers that implement it — specifically at some layer before generation begins, not after. If the probe only becomes accurate during generation (in the image token stream), the model is doing layout on the fly, not planning.

**Weight tracking:** When the verifier detects the transition from "composition fail" to "composition pass" for a given curriculum, look at the weight updates in the checkpoints immediately before and after. The weights that received anomalously large updates at this transition are the planning circuit components.

### What the results can tell you

If the probe works pre-generation for Curriculum III (proof-to-implicit) but not for Curriculum I (pure implicit): the planning circuit forms only when explicitly scaffolded by proof traces, not from implicit visual statistics alone. Implication: implicit data is insufficient; explicit reasoning traces are necessary to bootstrap the planning circuit.

If the probe works pre-generation for both curricula but forms faster for Curriculum III: implicit data is sufficient but slow; explicit reasoning traces accelerate planning circuit formation. Implication: reasoning traces are a form of privileged data that provide direct gradient signal for the planning circuit.

If the probe works only post-generation for all curricula: no curriculum creates a pre-generation planning circuit. The model always does layout on the fly. Implication: the architecture does not support pre-generation planning, or the task does not require it because on-the-fly layout is always sufficient for the training data.

Each of these outcomes changes what you do next. The first motivates reasoning-trace augmentation of training data. The second motivates a minimum-data threshold experiment. The third motivates an architecture modification.

---

## 9. Exploiting Axis Interactions

The individual axes are interesting. The interactions between axes are where the most surprising findings live.

### Explicitness × Tightness

Train at Level 1 (implicit) and Level 4 (explicit measurements), separately, both with tight containment examples.

- Does explicit training at tight tightness produce a more precise planning circuit than implicit training at tight tightness?
- Does explicit training at loose tightness generalize to tight tightness better than implicit training at tight tightness?

If explicit-loose beats implicit-tight, then the information content of explicit prompts compensates for the lack of geometric pressure from tight examples. If implicit-tight beats explicit-loose, then the gradient signal from geometric pressure is more important than the information signal from explicit measurements.

### Update semantics × Curriculum order

The same curriculum can be analyzed at two levels: what the model eventually learns
(behavioral endpoint) and what each individual update did along the way (update
trajectory). Two curricula can produce identical endpoints through completely different
update trajectories — one via many small routing updates followed by computation updates,
another via large computation updates that also incidentally improve routing.

To distinguish these trajectories, compute for each update the gradient alignment with
the theoretically correct routing update direction: the gradient of the true-offset
attention margin with respect to the Q and K weight matrices. A high routing-alignment
score means the update is primarily correcting where the model looks. A low score means
the update is correcting downstream computation given wherever the model currently looks.

Plotting routing alignment over training for different curricula reveals when each
curriculum triggers routing correction versus computation correction. This turns curriculum
comparison from a behavioral question (which curriculum reaches lower final error) into a
mechanistic one (which curriculum produces a trajectory of update types consistent with
building circuits in the right order).

### Complexity × Freedom

Train at complexity Level 3 (two objects, geometric constraint) with range-constrained freedom. Test at complexity Level 5 (hierarchical constraints) with partially-free freedom. Did the model learn "how to satisfy a constraint between two objects" as a reusable subroutine that applies when there are more objects and more freedom?

### Tightness × Curriculum Order

Compare tight-first versus loose-first curricula for the same final distribution (both curricula see the same examples overall, just in different order). The tight-first curriculum provides strong gradient signal for the planning circuit early; the loose-first curriculum provides weak gradient signal early. 

**Prediction:** Tight-first produces a planning circuit that forms faster and is more robust (lower error rate on tight test examples at convergence). Loose-first produces a planning circuit that forms later but may be more tolerant of ambiguous cases. Testing this prediction directly gives you a quantitative account of how gradient signal early in training affects the final circuit quality.

### Explicitness × Complexity × Verifier error type

Cross all three. Train at (Level 1, Complexity 2). Measure each verifier error type. Train at (Level 4, Complexity 2). Measure each verifier error type. Train at (Level 1, Complexity 4). Measure each error type. Train at (Level 4, Complexity 4). Measure each error type.

The 2×2 table of error type prevalences at convergence tells you: which error types are sensitive to explicitness, which are sensitive to complexity, and which are sensitive to the interaction. The interaction-sensitive error types are the ones where the circuit cannot form from either explicitness or complexity alone — it needs both simultaneously.

---

## 10. Cross-Domain Transfer and Circuit Nudging

One of the most practically important questions this framework can answer is: **can you cause a model to reuse an existing circuit for a new capability, with far less data than training from scratch?**

### The low-resource language analogy

A low-resource language shares grammatical structure with high-resource languages already in the model's weights. If you can construct training data that makes this structural connection explicit — by presenting the same content in both languages side by side — the model can attach the new language to the existing grammatical circuits rather than building new ones.

The hypothesis is: the gradient from aligned examples propagates through the shared grammatical circuit, strengthening the connection between the new language's surface forms and the existing abstract grammatical representation. With random interleaving, this connection does not form because the model does not see the structural parallel explicitly.

**Experimental test:**
- Condition A: Fine-tune on low-resource language examples, randomly ordered.
- Condition B: Fine-tune on low-resource language examples paired with structurally identical high-resource examples, interleaved at the sentence level.
- Condition C: Condition B, but interleaved at the paragraph level.
- Condition D: Condition B, but with the high-resource example always preceding the low-resource example.

Measure: how many examples of the low-resource language are needed before the model generalizes to held-out low-resource test examples, under each condition?

The condition that achieves the fastest grokking with the fewest examples is the condition that most efficiently triggers circuit reuse. The position and granularity of the interleaving (sentence vs. paragraph vs. document level) tells you at what resolution the model computes the structural parallel.

### Visual domain transfer

The same principle applies in the VLM setting. After training a strong containment circuit for circles inside squares, introduce hexagons inside rectangles — but interleave hexagon-in-rectangle examples with structurally parallel circle-in-square examples:

```
Example A: "A circle radius 20 inside a square side 60"   [image]
Example B: "A hexagon of width 20 inside a rectangle of width 60"   [image]
Example A': "A circle radius 25 inside a square side 80"  [image]
Example B': "A hexagon of width 25 inside a rectangle of width 80"  [image]
```

The structural parallel (same size ratio, same relation type, different shapes) should cause the gradient from hexagon examples to strengthen the existing containment circuit rather than building a hexagon-specific circuit from scratch. Compare the number of hexagon examples needed for generalization under interleaved vs. non-interleaved conditions.

### Pretraining transfer

Pretrain a model on a language domain with strong spatial structure — for example, formal geometry proofs, or cellular automata traces, or sorting algorithm descriptions. Then fine-tune on visual scene generation.

The hypothesis: pretraining on structured reasoning tasks builds circuits that implement rule-following, constraint checking, and step-by-step computation. These circuits are reused by the visual planning circuit when the model encounters containment constraints. The model needs fewer visual examples to acquire the planning circuit because it already has the underlying computational substrate.

**Test:** Compare probe geometry of the reasoning circuit (from pretraining) against the planning circuit (after fine-tuning). High cosine similarity between the two circuits' weight directions is evidence for circuit reuse. Low similarity is evidence that the visual planning circuit was built from scratch despite pretraining.

---

## 11. Invariance Studies Under the Same Framework

Once a circuit is located — by its activation pattern, its probe accuracy, its weight update signature — you can test its invariance properties by varying the inputs in controlled ways.

### Shape invariance of containment

After training on circle-inside-square and locating the containment circuit, test:
- Does the circuit activate for triangle-inside-pentagon? (Shape class invariance)
- Does the circuit activate for circle-inside-rectangle? (Shape symmetry invariance)
- Does the circuit activate for circle-inside-irregular-polygon? (Strict convexity invariance)

Each test that fails tells you what the circuit's representation encodes that it should not — and what additional training data would be needed to make the circuit more general.

### Scale invariance

Does the planning circuit generalize to very large objects (radius 200 in a 500px canvas) if it was trained on medium objects (radius 20 in a 100px canvas)? Scale invariance failure would suggest the circuit is computing in absolute pixel coordinates rather than normalized ratios — a specific claim about the internal representation that is directly testable.

### Color invariance of geometric circuits

Add color variation to the containment training data — sometimes a red circle in a blue square, sometimes a green circle in a yellow square. Measure whether this degrades geometric accuracy. If it does, the color circuit and the planning circuit are entangled — the model is using color as a cue for the spatial relation rather than computing the relation geometrically. 

This is a shortcut detector: if geometric accuracy depends on color, the model learned a color-based shortcut, not a geometric rule.

### Rotation invariance

Train on axis-aligned containment (square sides parallel to image edges). Test on rotated containment (square rotated 45 degrees). Does the containment circuit generalize to rotated frames? Rotation invariance failure is evidence that the circuit is computing containment in image coordinates rather than object-relative coordinates.

### Prompt surface invariance

Train on "circle inside square." Test on "square containing circle," "the circle is enclosed by the square," "a circle, enclosed within a square." These are all semantically equivalent but syntactically different. Do they activate the same containment circuit (semantic invariance) or different circuits? Measuring attention pattern similarity across these prompts tells you whether the circuit is semantic or syntactic in its operation.

---

## 12. What This Approach Can Answer

To summarize what controlled synthetic data, verifiable outputs, structured splits, and deliberate curricula can tell you:

**About circuit formation:**
When does a specific circuit form (error taxonomy timeline), what data causes it to form (curriculum comparison), how much data is needed (threshold experiments), and what weight update corresponds to its formation (weight tracking at the transition point).

**About circuit composition:**
Which circuits are prerequisites for others (ordering of error type falls), whether circuits compose modularly or entangled (disjoint vocabulary splits), and whether a circuit for one task transfers to a related task (cross-relation splits).

**About update semantics:**
Whether a single training update strengthened routing (attention to correct positions),
payload extraction (correct information from attended positions), or local computation
(correct function applied to extracted information); whether two updates with the same
loss decrease had the same or different internal effects; and whether a given batch will
produce a generalizing, memorizing, or shortcut-forming update — predictable in advance
from batch composition and current internal state.


**About generalization:**
Whether a model learned a general rule or a specific pattern (structured OOD splits), what the minimum evidence for generalization is (threshold experiments), and whether the model's solution is in a flat or sharp loss minimum (curriculum comparison with sharpness measurement).

**About representations:**
Where in the network a specific feature is computed (layer-by-layer probing), how invariant a representation is to surface variation (invariance studies), and whether explicit or implicit training produces different geometric structures (curriculum comparison with probe geometry).

**About data efficiency:**
Whether circuit reuse can substitute for new data (low-resource transfer experiments), what data structure enables the most efficient circuit reuse (interleaving experiments), and whether pretraining on structured tasks transfers computational circuits to new domains (pretraining transfer experiments).

**About training dynamics:**
The full memorization → circuit formation → cleanup arc (error taxonomy + weight norm tracking), whether circuits form gradually or as phase transitions (checkpoint frequency experiments), and what internal markers precede the behavioral grokking event (multi-metric tracking).

---

## 13. The Underlying Principle

Every experiment in this framework is a variation of the same logical structure:

1. **Design a dataset split** such that the training distribution contains examples of type A but not type B.
2. **Predict** what a model would do on type-B examples if it had learned the general rule versus if it had learned only the type-A pattern.
3. **Train** the model on type-A examples.
4. **Test** on type-B examples.
5. **Classify the errors** using the verifier to determine not just whether the model failed but how it failed — which specific capability is missing.
6. **Locate the gap** in the weights by comparing the internal representations of the model that succeeded vs. the model that failed.

The data-generating rules are the independent variable. The weights, circuits, and error patterns are the dependent variables. The verifier is the measurement instrument.

The ground truth — the generator's rule — is always available. This means every hypothesis about what the model should have learned is testable, every claim about what it actually learned is verifiable, and every failure is classifiable.

This is what it means to use data as a control.

---

*The experiments described throughout this document are not independent projects — they are a single coordinated research program. Each experiment answers a question whose answer changes what the next experiment should be. The framework is designed to be self-directing: as you learn what circuits form under what conditions, you use that knowledge to design the next split, the next curriculum, the next axis interaction, and the next invariance test. The data is not consumed by training. It is used to ask questions.*
