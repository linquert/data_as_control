# Research Questions: Mechanistic Interpretability of Transformers and Generative Models

> Compiled from a working research discussion on using synthetic procedural datasets as a controlled experimental instrument to study how transformers learn, form circuits, represent knowledge, and generate images. The core thesis: when you control the data-generating rule, you control the ground truth against which every internal hypothesis can be verified.

---

## I. Image Generation: What the Synthetic Data Approach Lets Us Study

### Token-to-feature binding
When the prompt says "red circle", which part of the network decides *red*, which decides *circle*, and how do they compose? With controlled data you can generate prompts with deliberate incongruence — "red circle" where the circle was actually blue in similar training images — and watch where the conflict is resolved in the network. Does binding happen at a specific layer, or is it a distributed process across the whole forward pass?

### Spatial circuit formation and generalization
Train only on left/right relations. Test on above/below. Does a single spatial-relation circuit form, or do four separate circuits? Does training on "above" transfer to "below" faster than "left" transfers to "inside"? The topology of transfer reveals the circuit's internal representation — whether it encodes direction as a continuous variable or as a discrete vocabulary.

### Color binding versus shape binding
The classic feature binding problem. Make color and shape statistically independent in training data, then correlated, then anti-correlated. Under independence: does the model learn a universal binding mechanism or always carry both attributes together in a single fused representation? The answer determines whether binding is a general-purpose circuit or a per-attribute-pair lookup.

### Instruction grounding — how text tokens steer generation
Systematically vary which words in the prompt change against which images those prompts generate. This allows computing a prompt Jacobian: how much does each token independently shift the generated image? With controlled data you know the ground-truth answer (the generator's rule) and can compare it to the model's implicit one. Where they diverge is where a shortcut has formed.

### Curriculum as SGD path surgery
The order of training batches is not just a scheduling choice — it is a choice about which loss surface the optimizer traverses. Does "shapes first, then arrangements" produce more compositionally general circuits than "arrangements first, then shapes"? Does easy-to-hard ordering always reach grokking faster, or does early exposure to hard examples sometimes accelerate it by providing gradient signal that bootstraps the harder circuits?

### Circuit reuse versus forking
Does the circuit for "circle above square" *reuse* the circle-drawing circuit from single-object training, or does a new composed-object circuit fork from it? Test this by ablating the single-object circle circuit and measuring degradation on the composed task. If degradation is proportional to circuit sharing, reuse is occurring. If degradation is zero, the fork was complete and independent.

### Edit versus generation algorithm
Are image edits handled by a distinct edit circuit, or by running generation with context conditioning? Train generation and editing separately, then examine whether the intermediate representations share weight structure. The hypothesis is that a model with a genuine world-model representation reuses the same circuits for both; a model with a direct prompt-to-pixel mapping develops separate edit and generation circuits.

### Implicit versus explicit geometric constraint satisfaction
Does the model perform geometric inference before rendering, or pattern-match the visual outcome? When "circle inside square" is given without measurements, at which layer does a representation of the valid size ratio appear? Train a linear probe on intermediate activations to predict the ratio of circle radius to square side. If the probe works on a prompt where no sizes were stated, the model computed the ratio internally — a distinct planning circuit exists. If the probe only works after generation begins, the model is doing layout on the fly.

### When does violating a visual constraint become inevitable in the forward pass?
When the model generates an image that violates a stated constraint, at which layer and token position was the violation already determined? There is a point in the forward pass where the constraint information is irretrievably lost or overwritten. Finding this point, layer by layer, tells you whether the constraint circuit fails by never forming, by forming and then being overridden, or by forming correctly but failing during the rendering phase.

### Freedom and preference under range constraints
When given "circle inside square, any valid size", what size does the model choose, and does that preference shift with different curricula? A "perfect fit only" curriculum should bias the model toward near-maximum fit. A "loose and perfect" curriculum should produce a distribution. Does a single qualifier word like "snugly" shift the distribution, or does the training prior dominate? This exposes the relative strength of the prompt-following circuit versus the weight-stored prior.

### How does prompt structure position affect which circuit handles a constraint?
"A circle inside a square" versus "inside a square, place a circle" versus "the circle is contained by the square." Same constraint, different surface forms. Do different surface forms activate different circuits, producing different error profiles? Systematic inversion errors on one form but not another reveal that the containment circuit has a syntactic dependency — it is not a fully abstract relation circuit.

### Does patterned pretraining on non-visual structured data transfer geometric reasoning circuits into image generation?
Pretrain on formal geometric proofs, cellular automaton traces, or register machine execution — all requiring systematic rule-following on structured inputs. Then fine-tune on image generation. Does the model acquire spatial circuits faster, with less data, and with lower OOD error? Compare the probe geometry of the rule-following circuits before fine-tuning against the spatial circuits after. Overlap in geometry is evidence for circuit transfer.

### Thinking tokens: when does the reasoning trace causally carry geometric state versus being decorative?
A thinking-token model will produce intermediate text before generating an image. Ablate specific thinking tokens — those where the model appeared to calculate a size constraint — and measure whether the generated image violates that constraint more. If the effect is specific to the ablated constraint and not to other properties, the token was causally carrying that information. If ablating any tokens degrades all outputs equally, the scratchpad is not load-bearing working memory.

### What does the thinking token trace do to the KV cache, and does it create a spatial plan that conditions generation?
The KV cache at the end of the thinking phase becomes the conditioning context for image generation. Does it contain a higher-fidelity spatial representation than the KV cache of a non-thinking model given the same prompt? Train linear probes on both and compare prediction accuracy of geometric parameters against the ground truth. If the thinking model's KV cache encodes relative sizes and containment margins before the first image token is generated, the thinking phase is doing genuine geometric planning.

---

## II. Weights, Learning Dynamics, and Training Evolution

### What is actually stored in the weights, and where?
MLP layers appear to act as key-value memories where the first projection retrieves and the second projects into the residual stream. Attention layers route information between positions rather than store it. But the precise division of labor is unclear: how much factual knowledge lives in attention weights versus MLP weights? How much syntactic structure is in each? Is the functional specialization principled or an accident of gradient descent on this particular data distribution?

### Superposition and the geometry of representation
The superposition hypothesis says a model with N neurons can represent more than N features by encoding them as near-orthogonal directions in high-dimensional space, at the cost of interference. How does the degree of superposition change with model scale? Is there a phase transition in representational geometry — a point where the model finds a cleaner, less superposed solution? Does a more uniform training distribution produce less superposition because no features dominate?

### How do weights evolve during training, not just at convergence?
When does each weight matrix specialize? Do early layers stabilize first? Do MLP weights converge before attention weights? Is there a period of representational churn followed by consolidation? In natural language, there are presumably many simultaneous grokking events at different timescales — when does the model grok subject-verb agreement, long-range coreference, compositional semantics? Are these sequential or parallel?

### What is the minimum capacity needed for specific capabilities?
What is the minimum number of attention heads, layers, and MLP neurons needed to implement a specific algorithm — say, in-context learning, indirect object identification, or containment reasoning? Is there a threshold below which a capability is simply not representable, and above which it appears suddenly? This would be strong evidence for the circuit hypothesis — that capabilities are implemented by identifiable subgraphs with minimum size requirements.

### How does the weight matrix change under fine-tuning versus pretraining?
Does fine-tuning modify existing circuits or create new ones? LoRA evidence suggests fine-tuning operates in a low-rank subspace, implying it modifies the direction of existing features rather than creating new features. Does full fine-tuning do something richer? When a model forgets a capability after fine-tuning, is that capability's circuit overwritten, suppressed, or rerouted?

### What does weight norm, weight rank, and weight spectral geometry tell us about a model's knowledge?
The singular value distributions of trained weight matrices may be interpretable. The effective rank of a weight matrix may correlate with the number of circuits that matrix participates in. Changes in spectral structure during training may track circuit formation and consolidation. This is largely unexplored territory with clear tools available — SVD, effective rank, spectral norm — but no systematic application to training dynamics.

### What mechanically changes when model size or data increases and generalization improves?
We have behavioral scaling laws but no mechanistic account of what actually changes internally when a larger model suddenly handles compositional scenes correctly. Does a new circuit appear that smaller models physically cannot represent? Or does a weak partial circuit in the small model strengthen and stabilize? Your controlled data lets you track this precisely: train the same task at multiple scales and measure whether the containment circuit exists in embryonic form or not at all, whether it generalizes or memorizes, and whether its probe geometry changes continuously or jumps.

### Among models with the same training loss, what internal organization makes one transfer better?
Two models can achieve identical loss while implementing completely different internal algorithms — one memorizing specific configurations, one learning a reusable rule. The flatness of the loss minimum correlates with better transfer: a flat minimum means many weight perturbations leave the loss unchanged, meaning the solution is more distributed and less brittle. Mechanistically, what makes a minimum flat? The hypothesis: a circuit-based solution relies on relational structure that is robust to small weight perturbations, while a memorized solution relies on precise weight values.

### What is mechanistic mode connectivity — can two same-loss solutions be linearly interpolated, and what does that reveal?
Two models at the same loss may lie in the same loss basin or in disconnected basins with a high-loss barrier between them. If mode-connected, linear weight interpolation maintains low loss throughout — they use the same mechanism reached by different gradient paths. If not mode-connected, they use structurally different mechanisms. Deliberately training two models toward the same loss via different curricula and testing mode connectivity directly reveals whether the loss landscape has multiple mechanistically distinct valleys.

### What is SGD actually doing to the weights at the gradient level to produce learning?
The circuit picture describes what forms. The gradient-level question is more fundamental: when the model sees a containment example, which weights receive the largest gradient updates, in which direction, and does repeated exposure push them toward a low-rank generalizing structure or a high-rank memorizing structure? Your controlled data gives ground truth to check gradients against: you know which weights *should* be updating to implement the generator's rule.

### What exactly is the mechanism of memorization at the weight level?
Before asking how memorization is cleaned up, you need to know what it is mechanistically. The hypothesis is that memorized solutions use high-norm, high-rank weight configurations that are sensitive to specific input patterns. But the precise structure — which weight matrices are involved, what their singular value spectrum looks like, whether memorization is localized to specific heads or distributed — is not known. A checkpoint where the model perfectly reconstructs training scenes but fails on OOD is a clean sample: probe the weight matrices for those specific scenes and characterize the memorization geometry directly.

### What is the memorization → circuit formation → cleanup arc, and does it always complete?
Grokking is the behavioral jump. The arc beneath it has three mechanistically distinct phases: memorization using high-norm brittle circuits; formation of a cleaner generalizing circuit that competes; cleanup where the memorized solution's weights shrink as the generalizing circuit dominates. The cleanup phase is the most underexplored. What triggers it? How long does it take relative to the grokking event? Do models ever freeze in a hybrid state — some compositions handled by the generalizing circuit, others by residual memorization?

### Can you find robust internal markers during training that predict, before the behavioral grokking event, which direction the model is heading?
Candidates: the effective rank of the spatial relation weight matrix should be lower for a generalizing solution; attention entropy for spatial heads should be higher; cosine similarity of circuit representations across different scene instances should be higher; loss landscape sharpness should be lower. If any of these internal markers predict the upcoming grokking event before the behavioral jump, you have a training-time diagnostic for whether the model is building real structure or memorizing — and a mechanistic account of what generalization actually is at the weight level.

### Can activation steering improve performance without weight updates, and what does that reveal?
If adding a specific direction to intermediate activations at inference time makes the model reliably satisfy constraints it otherwise violates, the capability was latent — the weights encoded it but the default forward pass was not activating the circuit sufficiently. The follow-up question is why: was the training distribution too dominated by easy cases that did not require the constraint circuit to activate? Did the learning rate anneal before the circuit became fully dominant? Your curriculum control makes both hypotheses testable.

### Does patterned pretraining on structured non-target data produce circuits that transfer to target tasks?
Pretrain on cellular automata state transitions, formal language grammars with known rules, or graph traversal traces. Does this accelerate circuit formation on target tasks? Does the probe geometry of the pretraining circuits overlap with the probe geometry of the target circuits? If cross-task probe similarity predicts transfer speed, it is strong evidence that the circuits are structurally identical and that transfer works by circuit reuse rather than general capability uplift.

### How does loss landscape flatness correlate with internal circuit organization, and can curriculum design predictably target flatter minima?
Curricula presenting varied compositions, multiple descriptions of the same scene, and interleaved structural parallels should push the optimizer toward flatter minima by preventing any single weight configuration from being necessary for a specific training example. Test this by measuring loss landscape sharpness at convergence for different curricula and correlating with OOD structural generalization, error taxonomy profiles, and internal circuit geometry — a direct link from data design to loss geometry to circuit organization.

---

## III. Circuits: Formation, Composition, Interaction, and Evolution

### What is a circuit operationally, and what are its minimal units?
The circuit framing identifies circuits as subgraphs of the computation graph — specific attention heads and MLP neurons connected through the residual stream. But the definition is somewhat circular: a circuit is a part of the model that does a task, and we find circuits by finding task-relevant parts. Are circuits the right unit of analysis, or is the actual computational structure better described as a continuous field of interactions with no clean boundaries?

### How do circuits form — gradually or as phase transitions?
The induction head literature shows that induction heads form suddenly at a specific point in training, correlated with a sudden improvement in in-context learning. The question is how general this is. Do all circuits form via phase transitions, or only the most algorithmic ones? For circuits implementing fuzzier capabilities — stylistic consistency, pragmatic inference — does the phase transition model still apply, or does it look more like a gradual accumulation?

### Do circuits compose hierarchically, and if so how?
Does a higher-level circuit read a feature computed by a lower-level circuit via the residual stream, treating it as a subroutine? Or is it more continuous — does the higher circuit receive a superposition of many lower partial computations and recombine them in ways that cannot be cleanly decomposed? The former implies a modular architecture that emerges from training. The latter implies a more holographic computation that resists decomposition.

### Are circuits universal across models?
The universality hypothesis says different models trained on similar data develop similar circuits. Induction heads appear across many architectures. But universality has not been tested systematically at scale or across architectures. If circuits are universal, they represent the unique optimal algorithm for their task given the architecture constraints. If they are not, gradient descent finds multiple valid solutions and the circuit formed depends on initialization or curriculum — which has completely different implications for interpretability.

### How do circuits interact — competition, cooperation, forking?
When two circuits are activated simultaneously and their outputs conflict, how is the conflict resolved? Is there an explicit arbitration mechanism in the weights, or does the residual stream add the outputs and let downstream circuits deal with the superposition? Does high-attention head activation suppress low-attention head activation implicitly? Whether circuit interaction is explicit arbitration or implicit superposition has very different implications for whether circuits are truly modular.

### How do circuits evolve with new data — do they extend, fork, or rebuild?
When a model encounters a distribution shift, the circuits handling the old distribution are retrained. Do they extend gracefully — adding new directions to represent new features? Fork — creating a separate circuit for the new distribution that shares some components? Or rebuild from scratch, destroying previous knowledge? This is related to continual learning and catastrophic forgetting but is mechanistically more specific about what structurally changes.

### What is the relationship between a circuit and the training examples that formed it?
Can you reconstruct which training examples were responsible for a specific circuit's formation? Given a circuit implementing capability X, what is the minimal subset of training data such that removing it prevents the circuit from forming? This is data attribution at the circuit level rather than the output level, and your controlled synthetic data makes it tractable: you know exactly what data was used, so you can run ablation studies on the training set.

### How do circuits interact and compose into larger algorithms — is there an algebra of circuits?
We can find local circuits, but we do not have a good account of how one circuit recruits another, routes into another, or is scaffolded by later training. For image generation: does the "circle inside square" circuit decompose cleanly into a containment-relation circuit and two separate shape-generation circuits? Can you ablate the containment circuit while leaving the shape circuits intact? If yes, the algebra is modular. If ablating containment also degrades shape generation, the circuits are entangled and the algebra is not clean.

### When does training in domain A produce circuits that transfer to domain B — what structural property determines transfer?
Models trained on RL game environments perform better on code and logic problems. This is not explained by surface similarity. The mechanistic hypothesis: game environments require state tracking, planning, and rule application — circuits structurally identical to what code and logic require. Can you identify the specific circuits that transfer, verify they are structurally identical in both domains, and then predict in advance from the structure of the training domain whether a new domain will benefit from pretraining there?

### Does the model learn the theoretically optimal internal algorithm for its task, or a locally optimal shortcut?
For arithmetic, the carry algorithm is the theoretical optimum and models appear to learn something close to it. For spatial reasoning, sorting, and graph traversal, the theoretically optimal algorithm is known and the optimal attention pattern is derivable. Does the model's learned algorithm match the theoretical one? If not, in what way does it diverge — a simpler shortcut that works in-distribution but fails OOD, a more complex equivalent algorithm, or something genuinely novel?

### For datasets with theoretically predicted attention patterns, can you verify whether the model learns exactly the predicted circuit?
Cellular automata, sorting, and register machine datasets all have clear theoretical structure: you can derive what an optimal attention pattern should look like from first principles. This gives you not just a behavioral ground truth but an internal ground truth. If the model's attention matches the theoretical prediction exactly, the circuit hypothesis is confirmed for that task. If it deviates, the deviation tells you whether the model found a shortcut, an equivalent algorithm, or something you did not anticipate.

### Can you nudge a model to reuse an existing circuit for a new concept with minimal new data, by constructing data that makes the structural connection explicit?
If a model has learned a spatial containment circuit for circles in squares, can you teach containment for hexagons in rectangles with far fewer examples than training from scratch — not by showing many hexagon scenes, but by interleaving hexagon examples with structurally parallel circle examples so the gradient naturally connects the new shape to the existing circuit? The data construction is the intervention. Side-by-side structural parallels force the connection that random hexagon training would not.

### How can circuit reuse with minimal data be exploited across semantic domains — the low-resource transfer hypothesis?
If you can identify a structural connection between an existing circuit and a new capability, and construct data that makes that connection explicit, the model can reuse the existing circuit with far fewer examples. Applied to language: a low-resource language may share grammatical structure with a high-resource language in the model's weights. Interleaving the same content in both languages side by side in the training stream may force the model to find the shared structural representation, making the new language cheaper to acquire. What placement, ratio, and interleaving structure maximizes this transfer?

---

## IV. Q, K, V: Attention Geometry and Dynamics

### What does each matrix represent semantically?
The functional description — Q is query, K is key, V is value — is not a semantic description. What is the geometric structure of the subspace that Q maps into? Does each head's Q matrix project the input into a subspace that selects for a specific semantic or syntactic feature? Does K project into a complementary subspace representing which positions have what? The QK inner product determines attention weights, so the geometry of QK space determines what the head pays attention to — and this geometry is fully learned.

### How does attention distribution change as context grows?
As sequence length increases, the attention distribution over earlier tokens changes not just because there are more tokens competing, but because earlier tokens' representations are updated by later context via the residual stream. How does this dynamic play out for long sequences? Does the model develop attentional habits — patterns of which positions it systematically attends to regardless of content — or does every attention decision depend entirely on content?

### How invariant are attention patterns to surface variation?
If you paraphrase a sentence, do the same attention patterns activate? If yes, attention is tracking semantic structure robustly. If not, attention is tracking syntactic surface forms and semantic stability is achieved by other means. With controlled data you can generate two prompts that are semantically equivalent but syntactically different, and compare their attention patterns across all heads and layers. The invariance profile tells you whether each head is a semantic or syntactic detector.

### What causes an attention head to change its distribution when new context arrives?
A new token modifies the residual stream at its position, which propagates forward. But the attention weights of later heads are computed from the current residual stream, which incorporates updates from all previous heads. A new token can therefore change the attention pattern of a downstream head even for positions far from where it landed. Tracing this causal chain — new token → which residual stream positions change → which attention heads change → what output changes — is an important open problem.

### How do Q and K together define the semantic space that each head attends in?
The product W_Q^T W_K defines a bilinear form on the residual stream — the "attention metric" for that head. This bilinear form determines which pairs of positions are semantically related according to that head. Studying the eigenstructure of W_Q^T W_K for each head tells you what relationships the head is sensitive to. Heads specialized for syntactic dependencies should have W_Q^T W_K aligned with syntactic distance measures. Heads specialized for coreference should have it aligned with entity identity.

### How does the value matrix transform what gets routed?
W_V determines what information is extracted from each position and added to the output. A head might attend to the correct position (good W_Q and W_K) but extract the wrong information from it (bad W_V). The value matrix is the other half of what a head does and is often overlooked in favor of studying attention patterns. For a head detecting subject-verb agreement, W_V should extract the number feature from the verb position so it can update the subject's representation. The composition of attention pattern and value read-out is the full story.

### How does layer composition of attention heads work — do heads in later layers read specifically from earlier heads?
There is evidence that some heads specifically read features written by particular earlier heads, forming "virtual weights" — their interaction through the residual stream is as if they were directly connected. A complete map of which head reads which other head's output, for every head in a model, would give the full circuit topology. This is computationally expensive but your controlled data, with known ground-truth computation, makes it tractable to verify the map for specific circuits.

### Are there universally specialized attention head types across models?
The induction head is the clearest case — a head that attends to the previous occurrence of the current token. Copy heads also appear across models. What is the complete vocabulary of head functional types? Are the types emergent from training or predictable from the architecture prior to training? And for image generation specifically: does a new category of head type emerge — a cross-modal grounding head, a spatial relation head — that does not exist in text-only models?

### How does attention interact with the MLP in the same layer?
Within a transformer block, the attention sublayer and MLP sublayer interact through the residual stream. Does the attention sublayer select which MLP memory cells to activate by routing the right features into the stream? There is some evidence for this — attention sets up the context, MLP does the lookup — but it is not established whether this is the actual mechanism or a post-hoc description that happens to be consistent with the observations.

### When is intermediate computation — CoT, scratchpad, trace tokens — genuinely carrying causal state versus being decorative?
Induction heads explain in-context learning for simple pattern completion. For richer reasoning, it is not known when intermediate text is genuinely carrying useful state versus changing the probability distribution over next tokens in a way that correlates with correct answers. Ablating specific scratchpad tokens and measuring whether specific downstream capabilities degrade proportionally is the test. A genuine working memory scratchpad should show specific degradation on the capability that depends on the ablated tokens. Decorative CoT should be robust to ablation or paraphrase.

---

## V. VLMs: Cross-Modal Circuits and the Image-Text Interface

### What is the "language" that the image encoder outputs?
In a VLM with a pretrained image encoder, the adapter layer between encoder and language model must translate from the image encoder's representational geometry to the language model's residual stream geometry. What does the adapter learn to preserve and what does it discard? Does it learn to align image features with token embeddings, or a richer mapping between the semantic geometry of vision and the semantic geometry of language? The adapter is a bottleneck and its capacity determines how much visual information can enter the language model.

### Where in the language model does cross-modal binding actually happen?
When the language model processes "the red circle in the image is...", at which layers do image tokens and text tokens start influencing each other's representations? If early layers handle cross-modal binding, the model performs grounding early and uses higher layers for reasoning on the grounded representation. If late layers handle binding, the model reasons about image and text separately and integrates at the end. These are different computational architectures with different failure modes and different implications for where cross-modal circuits live.

### What information do individual image patch tokens carry, and how does it evolve through layers?
An image patch token after the encoder carries a mixed representation of local visual features. As it passes through the language model layers, it receives attention from text tokens and updates its representation. Does the patch token's representation become more linguistically structured over layers — developing feature dimensions aligned with textual descriptions? Or does it stay primarily visual throughout? Probing patch token representations at each layer for decodability of linguistic properties (color, object identity, spatial relation) reveals the cross-modal alignment trajectory.

### How does attention head specialization change in a VLM compared to a text-only model?
Some attention heads in the language model of a VLM may develop specializations that do not exist in a text-only model — heads that selectively attend to image tokens, heads that compute cross-modal correspondences, heads that ground noun phrases in specific patches. Comparing the head functional types of a VLM against a text-only model of the same architecture and scale tells you exactly what cross-modal circuits were created by multimodal training, and which circuits were inherited unchanged from text pretraining.

### How do image tokens interact with each other inside the language model?
In the encoder, patches interact through vision-specific attention. In the language model, patches can interact again through language-model attention. Is this second round of patch-to-patch attention doing something additional — perhaps implementing spatial reasoning the encoder did not capture — or is it mostly redundant? If it is doing something additional, the language model's attention mechanism is being repurposed for spatial computation, visible in the geometry of the relevant heads' QK matrices.

### What is lost when an image is compressed into a fixed number of tokens?
Most VLMs compress an image into a fixed token count regardless of complexity. A simple solid-color image and a complex multi-object scene are both represented by the same token count. What information is sacrificed in the complex case — fine spatial detail, specific object attributes, or relational structure? And does the language model "know" it has incomplete information — does it express appropriate uncertainty when asked about regions that were compressed away?

### How does the VLM handle conflicting information between image and text?
If the text says "a red square" but the image shows a blue circle, how does the model resolve this? Does the image always win (grounding wins), the text always win (language prior wins), or does it depend on the context and relative confidence? The resolution mechanism is itself a circuit, and which signal dominates is determined by the relative influence of image tokens and text tokens in the relevant attention computations. Your controlled data can parametrize this conflict precisely.

---

## VI. Native Multimodal: No Pretrained Encoder

### How do raw patch tokens become semantically structured without encoder pretraining?
In a model processing raw patches without a pretrained vision encoder, the early layers must learn to extract visual features from scratch while simultaneously aligning them with language. Do these circuits match the circuits in a separately pretrained vision model, or does joint training produce a different representational geometry? This is a direct test of the universality hypothesis for vision circuits — do the same early visual features (edges, textures, colors) form regardless of whether they are learned in isolation or jointly with language.

### How do text and image circuits co-evolve during joint training?
In a native multimodal model, text circuits and image circuits develop simultaneously and each influences the other through shared weights. Does this co-evolution produce better-aligned representations than the adapter approach? Does text training help regularize image circuits, preventing them from overfitting to visual patterns with no linguistic correlate? Or does joint training interfere — does learning to process images slow down language circuit formation?

### What is the topology of cross-modal circuits in a native multimodal model?
In an adapter-based VLM, the cross-modal interface is explicit. In a native multimodal model, cross-modal circuits must emerge from data. Where do they live — concentrated in middle layers or distributed throughout? Do cross-modal heads look like modified versions of same-modal attention heads, or are they functionally distinct? The topology of these circuits — which layers they occupy, which heads they involve, how they connect to purely text or purely image circuits — is a wide-open question.

### How does the model represent the spatial structure of an image using the same mechanism it uses for sequential text?
Images have a 2D spatial structure that text does not. A native multimodal model must represent this spatial structure using attention over a linearized sequence of patches. Does the model develop positional circuits that reconstruct the 2D spatial relationships from the linearized sequence? How do these circuits differ from the position circuits in a text-only model? And how do spatial circuits interact with semantic circuits — does the model identify what objects are present and then determine where they are, or does it do both simultaneously?

### What happens when the model must reason about spatial relationships across modalities?
"Is the red circle to the left of the blue square?" requires taking a spatial fact from the image (relative positions of objects) and aligning it with the linguistic representation of "left of." This requires a circuit bridging the spatial representation circuit (computing relative positions from patch activations) and the linguistic spatial relation circuit (mapping "left of" to a relation type). How many examples does it take to form this bridge? Does it transfer across different spatial relations, or does each relation require a separate bridge?

---

## VII. Knowledge Storage, Retrieval, Routing, and Editing

### Where is knowledge stored in the weight matrices, and is it localized or distributed?
The MLP-as-key-value-memory picture assigns factual knowledge to the MLP layers, with the first projection retrieving and the second projecting. But attention heads also carry knowledge about relations and structure, and some knowledge appears distributed across both. For controlled data, "knowledge" has a precise definition — the generator's rules. Does the rule "containment requires the inner object to be smaller" live in the MLP weights, the attention weights, or both? Ablating each separately and measuring which ablation causes the rule to fail gives a direct localization answer.

### When does stored knowledge become editable, and what determines editability?
Some facts in a trained model can be modified by targeted weight updates while others cannot. The hypothesis is that localized facts — stored in a small number of neurons — are editable, while distributed facts — spread across many weights — are not. For synthetic data, editability is well-defined: can you update the model to use a new generator rule without degrading performance on old rules? Does the editable subset correspond to the localized subset? And does editability emerge at a specific point in training, or is it a property of the initialization that persists?

### How does the model arbitrate between knowledge stored in weights and knowledge available in context?
A model has two sources of knowledge: weight-encoded priors from training, and information present in the current context. When these conflict — the context says the circle radius is 30, but the model's weight-stored prior says circles are usually small — which wins? The arbitration mechanism is itself a circuit. Your controlled data can make this conflict explicit and parametric: vary the strength of the context signal and the strength of the weight-stored prior by curriculum, and measure when each dominates and by how much.

### How is knowledge routed from where it is stored to where it is needed in the computation?
Even if knowledge is stored in a specific MLP layer, it must be retrieved and routed to the attention heads that need it downstream. The routing mechanism — which attention heads read from which MLP outputs, how the residual stream carries information between components — is what the circuit picture tries to describe. For a general account of how routing works across all knowledge types, your controlled data makes it tractable: the knowledge needed for each task is precisely defined, so you can trace the routing path from storage to use for each generator rule.

---

## VIII. Explanation, Abstraction, and Causal Verification

### What is the right level of abstraction for a mechanistic explanation?
You can explain a constraint violation at the level of pixels, features, circuits, training data, or algorithm. Each level is a valid description but not equally useful. The causal abstraction standard says the right level is the coarsest description that correctly predicts which interventions fix the problem and which do not. For geometric constraint violations: the right level is identified by whether the correct intervention is "add more training data," "activate the constraint circuit via steering," or "restructure the prompt to expose syntactic argument order." Your controlled data lets you run all three interventions and measure which is causal.

### Can you find robust internal markers that predict grokking direction before the behavioral event?
If internal markers — weight norm trajectory, attention entropy, probe decodability, singular value spectrum, effective rank — can predict the upcoming grokking event before it is behaviorally visible, you have a training-time diagnostic for whether the model is building real structure or memorizing. Finding predictive markers is finding the causal precursors of generalization, which is a mechanistic account of what generalization actually is at the weight level — not just a behavioral description of when it happens.

### When is an explanation genuinely causal versus merely correlational, and what is the minimal intervention that distinguishes them?
Finding that a specific attention head activates during containment reasoning is correlational. Showing that ablating that head causes containment reasoning to fail is causal. But causal experiments on neural networks are not clean — ablating a head changes the residual stream and affects downstream computations in hard-to-isolate ways. What is the minimal, most surgical intervention that isolates a specific circuit's causal contribution? Activation patching and causal scrubbing are the current tools. Your controlled data, with known ground-truth computation, lets you verify whether the intervention is isolating the right computation or confounding it with adjacent ones.

### How do you find the minimal faithful causal abstraction automatically, rather than by hand?
Causal abstraction is the strongest current framework: an explanation is valid if it is a faithful higher-level causal model of the low-level computation. But in practice, finding the minimal faithful abstraction requires search over a space of possible abstractions, and for large models this search is intractable without guidance. Your controlled data structures this search: because you know the generator's algorithm, you know what the correct high-level causal model should be, and you can test faithfulness by checking whether interventions on the model's circuit match predictions made by the generator's algorithm. This is the clearest path to automatic causal abstraction for a specific family of tasks.

---

## IX. Error Taxonomy as a Mechanistic Diagnostic

### Attribute binding errors
The right objects are present, the right attributes are present, but they are bound to the wrong objects. Red circle and blue square becomes blue circle and red square. These errors reveal that attribute and object circuits are learned separately but the binding mechanism is weak or entangled. These errors should decrease when training with more attribute-object pairs that have high variation in which attribute goes with which object.

### Compositional inversion errors
"Circle inside square" becomes "square inside circle." The relationship is recognized but the direction is inverted. This is a specific failure of the relational circuit — it knows containment is the relation but has not learned which argument is the container and which is the contained. Compositional inversion errors are systematic inverses of the correct answer, suggesting the circuit has almost the right structure but is missing the argument ordering component.

### Scale inconsistency errors
The objects are in the right relationship but the sizes are wrong relative to each other. This is a failure of the relative-size circuit independently of the containment circuit, and you can see this because sometimes the model gets the containment right but the scale wrong — suggesting the two circuits are somewhat independent and can fail independently.

### Cardinality errors
Wrong number of objects. "Exactly 3 circles" produces 2 or 4. These tend to be off-by-one errors, suggesting the counting circuit has a systematic bias. Whether the bias is toward undercounting or overcounting, and whether it scales with the target number, tells you whether the model has a counting algorithm or a magnitude estimator.

### Constraint satisfaction errors
The model ignores a stated constraint entirely — "circle inside square" produces a circle completely outside the square. These are different from inversion errors: the model is not confusing the direction of the relation, it is not applying the constraint at all. These errors decrease most strongly when the training curriculum emphasizes the constraint as a hard rule rather than a soft statistical tendency.

### Geometric calculation errors
The model generates a scene where the stated measurements are internally inconsistent. "Circle radius 30, square side 40, circle inside square" — geometrically possible only if the circle is at the corner, but the model places it centered, making it overflow. This is a failure of the geometric planning circuit — the model did not compute whether the configuration was achievable before drawing it.

### Preference errors
When given a range, the model chooses values that satisfy the constraint but are implausible given the prompt context. These are errors of calibration, not logical correctness, and they reveal the model's prior as baked in by training rather than updated by the prompt. They are the direct behavioral signature of training-prior dominance over in-context instruction.

### Error-type trajectories across training as a circuit formation map
For every training checkpoint, measure the prevalence of each error type separately. Plot each as a curve over training time. Different error types decreasing at different times corresponds to different circuits forming at different points. Scale inconsistency errors decreasing when the relative-size circuit forms. Binding errors decreasing when the attribute-binding circuit forms. If all error types decrease simultaneously, it suggests a single undifferentiated circuit rather than a modular structure — a strong architectural finding.

---

*These questions are ordered within sections roughly from most fundamental to most specific, and from architecture-general to task-specific. The underlying method is the same throughout: control the data-generating rule precisely enough that you know what the model should be computing, then measure what it is actually computing and when, at every level from gradient updates to behavioral error types.*
