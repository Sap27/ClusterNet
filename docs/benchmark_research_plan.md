# Community Detection Benchmarking: Research Plan

## Paper Narrative

> **"When and why do GNNs beat classical methods for community detection — and when don't they?"**

The benchmark is structured as three tiers of increasing real-world complexity:
LFR (synthetic, no attributes) → CORA (real, rich attributes) → Biological (real, functional features).
Each tier adds information that GNNs can exploit but classical methods cannot, letting us
**quantify the value of node attributes** for community detection.

---

## 1. LFR Synthetic Benchmarks

**Purpose:** Establish baselines on controlled topology — no natural node
attributes. GNNs use structural features (degree, clustering coefficient,
PageRank), making this a topology-only regime where classical methods are
expected to dominate. Any GNN that matches classical performance here is
extracting signal purely from structure; any that falls short reveals the
cost of learning without informative features.

| Benchmark       | X-axis               | Y-axis     | What it tests                          |
|-----------------|----------------------|------------|----------------------------------------|
| Accuracy        | Mixing parameter μ   | AMI        | Detection difficulty                   |
| Scalability     | Network size N       | Runtime    | Computational scaling                  |
| Hierarchical    | μ1 / μ2 (macro/micro)| AMI at each level | Multi-scale detection (see below) |

### 1.1 Hierarchical Benchmark: Three-Paradigm Comparison

The hierarchical benchmark uniquely tests whether algorithms detect the right
**scale** of community structure. Networks have two nested levels: ~130
micro-communities grouped into ~8 macro-communities, with independent mixing
parameters μ1 (macro) and μ2 (micro).

The benchmark contrasts three paradigms:

| Paradigm | Algorithms | Question answered |
|---|---|---|
| **Scale-blind** | Louvain, Leiden, Walktrap, Infomap, csbio_iitm2 | Which scale does the algorithm naturally gravitate toward, without any hint? How does this shift as μ1/μ2 change? |
| **Scale-guided** | DMoN, MinCut, VGAE, DGI (with explicit K) | Does providing the target K reliably steer the algorithm to the correct hierarchy level? |
| **Scale-aware** | NestedSBM (multi-level extraction) | Can a single run recover **both** levels simultaneously? |

**Key analyses:**

- **(μ1, μ2) phase diagram:** Plotting the "detected level" (micro vs macro)
  as a heatmap over the 3×3 mu grid reveals each algorithm's scale preference
  and the transition boundary where it flips from one level to the other.
- **num_detected as a proxy for scale:** The number of detected communities
  immediately reveals whether the algorithm is operating at micro scale (~130)
  or macro scale (~8), before looking at AMI.
- **NestedSBM as the upper bound:** NestedSBM's hierarchical state provides
  level 0 (finest partition, scored against micro ground truth) and coarser
  levels (scored against macro ground truth) from a single inference run —
  the principled reference for hierarchy-aware detection.
- **Scale-blind vs scale-guided:** If a GNN at K=macro outperforms a classical
  algorithm at macro detection, K-guidance adds value. If a classical algorithm
  still wins despite not being told K, that is a strong endorsement of
  classical methods on topology-only data.

---

## 2. Real-World Benchmarks (CORA)

**Purpose:** Test on a real citation network with node features (bag-of-words) and
semi-ground truth (topic labels). GNNs enter the picture here because features are
naturally available.

### 2.1 Original CORA (Baseline)

| Property    | Value                     |
|-------------|---------------------------|
| Nodes       | 2,485                     |
| Edges       | 10,138                    |
| Communities | 7 topics                  |
| μ (mixing)  | 0.329                     |
| Source      | `CORA_new/original/`      |

**Question:** How well do algorithms recover topic-based labels?

**Metrics:**
- AMI vs topic labels
- Modularity (unsupervised quality)
- Conductance (boundary sharpness)
- Cluster size distribution

### 2.2 Mesoscale Benchmark

| Variant              | What it does                       | μ change |
|----------------------|------------------------------------|----------|
| `inner_random_{25,50,75,100}` | Rewires edges **within** communities  | None (constant μ) |
| `inter_random_{25,50,75,100}` | Rewires edges **between** communities | None (constant μ) |

**Question:** Does the algorithm use internal topology or just community boundaries?

**Key insight:** μ stays constant. If AMI drops, the algorithm is sensitive to
internal/boundary structure beyond just density.

**Metrics:**
- AMI vs original communities
- Element-Centric Similarity (ECS) — per-node contribution to disagreement
- Stability (variance across shuffles)

### 2.3 Weaken Benchmark

| Variant                          | Edges rewired | Preserves         |
|----------------------------------|---------------|-------------------|
| `Q_decrease_1k_{10,20,30,40,50}` | degree-preserving | Degree distribution |
| `Q_decrease_2k_{10,20,30,40,50}` | correlation-preserving | Degree + correlations |
| `Q_decrease_3k_{10,20,30,40,50}` | clustering-preserving | Degree + correlations + triangles |

**Question:** How robust is each algorithm to community destruction, and which
topological properties does it depend on?

**Metrics:**
- AMI decay curve (AMI vs weakening %)
- Critical threshold (weakening % where AMI < 0.5)
- Correlation: AMI vs μ change
- 1k vs 2k vs 3k comparison (reveals which structural properties matter)

### 2.4 CORA Experiments for GNNs (the differentiator)

#### Experiment A: Feature Ablation

Run every GNN with **three feature settings** on the same CORA graph:

1. **Real features** — bag-of-words (1,433-dim)
2. **Random features** — Gaussian noise, same dimensionality
3. **Structural features** — degree, clustering coefficient, spectral embedding

Compare to classical (no features at all).

**Why this matters:** If GNN + random features ≈ classical, and GNN + real features
>> classical, we have **quantified the value of attributes**. If GNN + structural
features ≈ GNN + real features, the GNN is mostly learning topology anyway.

#### Experiment B: Mesoscale Sensitivity with Features

On `inner_random` and `inter_random` variants:
- Classical: AMI drops differently for inner vs inter rewiring.
- GNN **with** features: should be **more robust** (features still carry signal).
- GNN **without** features: should degrade like classical.

**Tests:** "Do GNNs use features to compensate for structural degradation?"

#### Experiment C: Weaken Degradation with Features

On `Q_decrease_{1k,2k,3k}`:
- Plot AMI vs weakening % for all methods.
- Classical degrades at rate X.
- GNN with features degrades **slower** (attributes informative even as topology weakens).
- GNN without features degrades at same rate as classical.

---

## 3. Biological Benchmarks

**Purpose:** Validate biological relevance of detected communities on real
gene/protein networks where no partition ground truth exists. Evaluation is
entirely functional: do the discovered modules correspond to known biological
processes (GO), pathways (KEGG), and regulatory motifs (FFLs)?

This is the hardest tier of the benchmark — algorithms must produce
communities that are not just topologically coherent but **biologically
meaningful**.

### 3.0 Network Inventory

| Network | Nodes | Edges | Avg deg | Type | Weight semantics |
|---|---|---|---|---|---|
| **GTEx co-expression (0.8)** | 14,347 genes | 3,456,602 | 481.9 | Undirected, weighted | Pearson correlation of TPM expression across ~17K tissue samples, thresholded ≥ 0.8 |
| **Human GRN** | 18,609 (TFs + targets) | 278,660 | 29.9 | Directed, weighted | Regulatory confidence [0.4, 1.0] |
| **BioGRID PPI** | 17,473 proteins | 944,933 | 108.2 | Undirected, unweighted | Binary experimental interaction |
| **SIGNOR directed** | 5,837 proteins | 17,484 | 6.0 | Directed, signed | Activation (+1) / inhibition (-1) |

**Source data:** GTEx V8 RNA-Seq TPM matrix (56,200 genes × 17,382 samples),
BioGRID experimental, SIGNOR curated signaling, Human GRN from regulatory
databases.

**GTEx threshold choice (0.8):** We use Pearson ≥ 0.8 as the single
operating point — it retains 14.3K genes with 3.5M edges, balancing coverage
(enough genes for meaningful enrichment) against density (avg degree ~482,
tractable for GNNs). The 0.7 threshold (20.7K nodes, 6M edges) includes
weaker correlations that may add noise, while 0.9 (5.7K nodes) discards too
many genes.

### 3.0.1 Symmetrization of Directed Networks (GRN, SIGNOR)

Most community detection algorithms — both classical (Louvain, Leiden,
Walktrap, Infomap, etc.) and GNN-based (DMoN, MinCut, VGAE, DGI, etc.) —
are defined on **undirected** graphs:

- **Modularity** (optimized by Louvain, Leiden, DMoN) is defined as
  Q = (1/2m) Σ [A_ij − k_i k_j / 2m] δ(c_i, c_j), which assumes a
  symmetric adjacency matrix.
- **GNN message passing** (GCN, GAT layers) propagates information in both
  directions along each edge. Even if we feed a directed adjacency, the
  learned representations are effectively influenced by both endpoints.
- **Normalized cut** (MinCut), **random walks** (Walktrap, Node2Vec), and
  **mutual information** (Infomap, DGI) all have standard formulations for
  undirected graphs.

For the Human GRN (TF → target) and SIGNOR (activator → target), we
**symmetrize** the adjacency for community detection: if TF₁ regulates Gene₁,
we create an undirected edge between them. The rationale is that regulatory
relationships indicate functional relatedness regardless of direction — a TF
and its target gene participate in the same biological process and should
tend to co-occur in the same module.

However, we **retain the original directed edges** for evaluation:
- **FFL motif preservation** (GRN) requires directed triangles: TF₁→TF₂,
  TF₁→Gene, TF₂→Gene. These are evaluated on the original directed graph.
- **Sign coherence** (SIGNOR) requires the original signed, directed edges
  to distinguish activation (+1) from inhibition (-1).

This separation — detect on undirected, evaluate on directed — is standard
practice in the GRN community detection literature.

### 3.1 GTEx Gene Co-expression Network

| Property | Value |
|---|---|
| Construction | ~24K genes, each represented by a TPM expression vector across ~17K tissue samples. Pearson correlation computed for all gene pairs; edges with Pearson ≥ 0.8 kept. |
| Node features (GNN) | Gene expression profile (TPM vector, ~17K dimensions, PCA-reduced to 128 for tractability) |
| Ground truth | None (no known gene partition). Evaluated via functional enrichment. |

**Experiment 3.1a: Community detection sweep**

Run all classical + GNN algorithms. For GNNs, provide PCA-reduced expression
profiles as node features; for classical, features are ignored.

| Metric | What it measures |
|---|---|
| Modularity | Topological quality of partition |
| Number of communities | Resolution / granularity |
| GO enrichment (median -log10 p-value across clusters) | Functional coherence |
| Fraction of clusters with significant GO enrichment (p < 0.05 after correction) | Coverage of meaningful modules |
| KEGG pathway enrichment | Pathway-level biological relevance |
| Cluster size distribution (Gini coefficient) | Balance — are communities useful or trivially small/large? |
| Runtime | Practical scalability on dense real networks |

**Experiment 3.1b: Feature ablation for GNNs**

Same as CORA Experiment A, but on biological data:

| Feature setting | Input to GNN | What it tests |
|---|---|---|
| **Expression features** | PCA-reduced TPM profiles | Full biological signal |
| **Structural features** | Degree, clustering coeff, PageRank | Topology only (GNN ≈ classical?) |
| **Random features** | Gaussian noise, same dim | Baseline — does the GNN extract anything beyond noise? |

Compare GO enrichment across settings. If expression features >> structural >>
random, we have quantified the value of gene expression for community detection
on real biological networks.

### 3.2 Gene Regulatory Network (Human GRN)

| Property | Value |
|---|---|
| Construction | Directed TF → target gene edges with regulatory confidence weights [0.4, 1.0]. ~18.6K nodes, ~279K edges. |
| Node features (GNN) | Expression profile (from GTEx TPM, mapped by gene symbol) + binary TF/target role indicator |
| Ground truth | None. Evaluated via motif preservation + GO enrichment. |
| Key structural property | Feed-forward loops (FFLs): TF₁ → TF₂, TF₁ → Gene, TF₂ → Gene. The canonical regulatory motif. |

**Note on directionality:** Most classical community detection algorithms and
all our GNN implementations operate on undirected graphs. For the GRN, we
symmetrize the adjacency (treat TF→target as bidirectional) for community
detection, but evaluate motif preservation on the original directed edges.

**Experiment 3.2a: Community detection + motif enrichment**

Run all algorithms on the symmetrized GRN. For each resulting partition:

1. **Enumerate FFLs** in the original directed network (all directed triangles
   matching TF₁→TF₂, TF₁→G, TF₂→G).
2. **Count intra-cluster FFLs** (all three nodes in the same community) vs
   **cut FFLs** (at least one node in a different community).
3. Compute **FFL preservation ratio** = intra-cluster FFLs / total FFLs.

A good partition keeps regulatory circuits intact. Higher FFL preservation =
better biological relevance.

| Metric | What it measures |
|---|---|
| FFL preservation ratio | Regulatory motif coherence |
| GO enrichment per cluster | Functional coherence |
| KEGG pathway enrichment | Pathway relevance |
| Modularity | Topological quality |
| Number of communities | Resolution |
| Fraction of TFs per community | Regulatory hub distribution |

**Experiment 3.2b: TF knockout perturbation**

Simulate gene knockout by removing high-degree TF nodes (the regulatory hubs):

| Perturbation | Nodes removed | Biological analogue |
|---|---|---|
| Top-5 TFs removed | 5 highest out-degree TFs | Key regulator knockout |
| Top-10 TFs removed | 10 highest out-degree TFs | Multi-regulator knockout |
| Random 5 nodes removed | 5 random genes | Control |
| Random 10 nodes removed | 10 random genes | Control |

For each perturbation:
1. Remove nodes and their edges from the network.
2. Rerun community detection on the perturbed network.
3. Measure **community stability**: Jaccard similarity between original and
   perturbed partitions (restricted to surviving nodes).
4. Measure **FFL preservation** on the perturbed network.

**Key question:** Do algorithms that preserve FFLs well on the intact network
also remain stable under TF knockout? Are GNNs (with expression features for
remaining genes) more robust because they can compensate for lost topology
with feature similarity?

### 3.3 BioGRID Protein-Protein Interaction Network

| Property | Value |
|---|---|
| Construction | Undirected, unweighted experimental protein-protein interactions. 17.5K proteins, 945K edges. |
| Node features (GNN) | GO term binary vectors (which GO terms each protein is annotated with) or structural features |
| Ground truth | None. Evaluated via GO/KEGG enrichment. |

**Experiment 3.3a: Community detection + functional enrichment**

Run all classical + GNN algorithms. Evaluate with GO and KEGG enrichment.
BioGRID's large size and moderate density (avg degree 108) makes it a good
test of scalability on real data — positioned between the sparse GRN and
the dense GTEx networks.

**Experiment 3.3b: Feature ablation**

| Feature setting | Input to GNN |
|---|---|
| **GO term vectors** | Binary vector of GO annotations per protein |
| **Structural features** | Degree, clustering coeff, PageRank |
| **Random features** | Gaussian noise |

This tests whether GO annotations (a form of prior biological knowledge)
help GNNs discover better protein complexes than topology alone.

### 3.4 SIGNOR Directed Signaling Network

| Property | Value |
|---|---|
| Construction | Curated directed signaling interactions with signed weights: +1 (activation), -1 (inhibition). 5.8K proteins, 17.5K edges. |
| Node features (GNN) | GO term vectors + sign-aware degree features (in-degree from activators, in-degree from inhibitors) |
| Ground truth | None. Evaluated via pathway enrichment. |

**Experiment 3.4a: Community detection + pathway enrichment**

SIGNOR is small and sparse (avg degree 6), making it the easiest network for
all algorithms computationally. The challenge is the signed, directed nature.
We symmetrize for community detection but evaluate whether discovered modules
respect known signaling pathways.

**Experiment 3.4b: Sign-aware evaluation**

For each detected community, compute:
- **Sign coherence**: Fraction of intra-cluster edges that are activating (+1)
  vs inhibiting (-1). Biologically coherent modules should be predominantly
  activating (co-regulated genes) with inhibition mostly at boundaries.
- Compare sign coherence across algorithms.

### 3.5 Perturbation Analysis (Cross-Network)

Applied to all four networks (parameters scaled to network size):

| Perturbation | Levels | Biological analogue |
|---|---|---|
| **Edge dropout** | 5%, 10%, 20%, 30% random edge removal | Missing/undetected interactions |
| **Node removal (random)** | 1%, 2%, 5% random nodes | Random gene loss / incomplete data |
| **Node removal (hub-targeted)** | Top 1%, 2%, 5% by degree | Hub gene knockout |
| **Edge noise** | Add 5%, 10% random false edges | False positive interactions |

**Protocol per perturbation level:**
1. Apply perturbation to the network.
2. Rerun all algorithms on the perturbed network.
3. Compare to unperturbed partition:
   - **Jaccard stability** of top-K largest communities
   - **NMI** between perturbed and unperturbed partition (treating unperturbed as reference)
   - **GO enrichment preservation** (does functional coherence survive?)
4. Repeat 5 times with different random seeds; report mean ± std.

**Key question:** Which algorithms are most robust to missing data (edge
dropout) and noise (false edges)? Do GNNs with informative features degrade
more gracefully than classical methods?

### 3.6 GNN Node Features for Biological Networks

| Network | Primary features | Dimension | Fallback |
|---|---|---|---|
| **GTEx** | PCA-reduced TPM expression profiles | 128–256 | Structural (degree, CC, PageRank) |
| **Human GRN** | TPM expression (mapped by gene symbol) + TF/target binary | 130–258 | Structural |
| **BioGRID** | GO term binary vectors | ~5,000 (sparse) | Structural |
| **SIGNOR** | GO term vectors + sign-aware degree | ~5,004 | Structural |

**Expression feature pipeline (GTEx, GRN):**
1. Load GTEx V8 TPM matrix (56,200 genes × 17,382 samples).
2. Filter to genes present in the target network.
3. Log-transform: log2(TPM + 1).
4. PCA to 128 dimensions (retaining ~80% variance).
5. Assign as node feature matrix X.

**GO term feature pipeline (BioGRID, SIGNOR):**
1. Download GO annotations (GAF format) from Gene Ontology.
2. Map protein IDs to GO terms.
3. Build binary matrix: rows = proteins, columns = GO terms.
4. Filter to GO terms annotating ≥ 10 and ≤ 1000 proteins (remove trivial/universal).
5. Assign as node feature matrix X.

### 3.7 External Resources Required

| Resource | Source | Format | Purpose |
|---|---|---|---|
| **GO annotations (human)** | `http://current.geneontology.org/annotations/goa_human.gaf.gz` | GAF | GO enrichment evaluation + GNN features for BioGRID/SIGNOR |
| **GO ontology (OBO)** | `http://purl.obolibrary.org/obo/go-basic.obo` | OBO | Hierarchy for enrichment tools (goatools) |
| **KEGG pathway gene sets** | MSigDB `c2.cp.kegg_medicus.v2024.2.Hs.symbols.gmt` from `https://www.gsea-msigdb.org/` | GMT | KEGG pathway enrichment |
| **Gene ID mapping** | Already in GTEx TPM header (Ensembl → Symbol via `Description` column) | GCT | Map between Ensembl IDs and gene symbols across networks |
| **gprofiler2** (Python) | `pip install gprofiler-official` | API | Programmatic GO/KEGG enrichment (alternative to local goatools) |

**Preferred enrichment approach:** Use `gprofiler-official` Python package —
it queries the g:Profiler server, handles multiple-testing correction
(g:SCS), and supports both gene symbols and Ensembl IDs. No local
annotation files needed for enrichment queries. Local GO/OBO files are
needed only if we want to compute GO-term feature vectors for BioGRID/SIGNOR.

### 3.8 Summary: Experiments × Networks

| Experiment | GTEx (0.8) | Human GRN | BioGRID | SIGNOR |
|---|---|---|---|---|
| Community detection sweep | Yes | Yes | Yes | Yes |
| GO enrichment evaluation | Yes | Yes | Yes | Yes |
| KEGG pathway enrichment | Yes | Yes | Yes | Yes |
| Feature ablation (GNN) | Yes (expression vs structural vs random) | Yes (expression vs structural) | Yes (GO terms vs structural) | Yes (GO terms vs structural) |
| FFL motif preservation | — | Yes | — | — |
| Sign coherence | — | — | — | Yes |
| TF knockout perturbation | — | Yes | — | — |
| Edge dropout perturbation | Yes | Yes | Yes | Yes |
| Node removal (random + hub) | Yes | Yes | Yes | Yes |
| Edge noise perturbation | Yes | Yes | Yes | Yes |

---

## 4. GNN Selection (6 SOTA Methods + 2 Baselines)

All methods are **published, peer-reviewed, and widely cited**. No custom
unpublished variants. All use **industry-standard loss functions** from the
original papers. All are **transductive** (train and evaluate on the same
graph — the natural setting for community detection).

### Non-GNN baselines

| #  | Method              | Uses edges? | Uses features? | Loss / objective               | Reference |
|----|---------------------|-------------|----------------|--------------------------------|-----------|
| B1 | **MLP + KMeans**   | No          | Yes            | KMeans objective (no NN loss)  | Standard ablation baseline |
| B2 | **Node2Vec + KMeans** | Yes       | No             | Skip-gram + negative sampling  | Grover & Leskovec, KDD 2016 |

B1 isolates the contribution of node features (if GNN ≈ MLP, edges add
nothing). B2 isolates the contribution of topology (if GNN ≈ Node2Vec,
features add nothing). Together they bracket the information spectrum.

### GNN methods

| #  | Method              | Paradigm           | Loss function (published standard)                                        | Reference                       |
|----|---------------------|--------------------|--------------------------------------------------------------------------|---------------------------------|
| 1  | **DMoN**            | Unsupervised, end-to-end | Modularity: -Tr(C^T B C)/(2m) + collapse reg + orthogonality reg   | Tsitsulin et al., JMLR 2023    |
| 2  | **MinCutPool**      | Unsupervised, end-to-end | Normalized cut: -Tr(S^T A S)/Tr(S^T D S) + orthogonality           | Bianchi et al., ICML 2020      |
| 3  | **VGAE + KMeans**   | Unsupervised, two-stage  | ELBO: reconstruction BCE + KL divergence to N(0,I)                  | Kipf & Welling, NeurIPS-W 2016 |
| 4  | **DGI + KMeans**    | Unsupervised, contrastive| MI maximization: BCE on bilinear discriminator (Jensen-Shannon)     | Velickovic et al., ICLR 2019   |
| 5  | **GCN semi-sup**    | Semi-supervised          | Cross-entropy (NLL) on labeled nodes                                | Kipf & Welling, ICLR 2017      |
| 6  | **GAT semi-sup**    | Semi-supervised          | Cross-entropy (NLL) on labeled nodes, with learned attention        | Brody et al. (GATv2), ICLR 2022|

### Paradigm coverage

- **2 end-to-end clustering** (DMoN, MinCut): GNN directly outputs community
  assignments. Tests modularity vs spectral objective.
- **2 two-stage** (VGAE, DGI): GNN learns embeddings, KMeans clusters them.
  Tests generative vs contrastive representation learning for communities.
- **2 semi-supervised** (GCN, GAT): use partial labels. Upper bound on
  performance — quantifies the value of ground truth information.
- **2 baselines** (MLP, Node2Vec): isolate feature vs structure contribution.

### What was dropped and why

| Dropped method       | Reason |
|---------------------|--------|
| GCNCluster (ours)   | Unpublished variant. DMoN already covers GCN encoder + modularity loss with a peer-reviewed architecture. |
| GATCluster (ours)   | Unpublished variant. Same paradigm as DMoN but with attention — not an established community detection method. |
| SAGECluster (ours)  | Unpublished variant. Tests concatenation ∧ function, but not a published community detection method. |
| GINCluster (ours)   | Unpublished variant. GIN designed for graph classification (WL test), not community detection. |
| Graph Transformer   | No published paper applies Graph Transformers to community detection with a standard loss. Our implementation (transformer + modularity) is a custom adaptation. |
| GRACE               | Replaced by DGI — the established predecessor with higher citations and a published PyG implementation. |

---

## 5. Training GNNs on Biological Networks (4 Networks, No Partition Ground Truth)

### The Challenge

Unlike LFR (known communities) or CORA (topic labels), biological networks
have **no partition ground truth**. We cannot compute AMI or train
semi-supervised GNNs in the standard way. All evaluation is functional
(GO enrichment, motif preservation, pathway overlap).

### Strategy 1: Transductive Unsupervised (DMoN, MinCut, VGAE, DGI)

All four unsupervised GNNs train directly on the target graph — no external
data, no labels. This is the natural setting for biological discovery.

**Training protocol per network:**
1. Construct adjacency + node feature matrix (see Section 3.6).
2. Initialize model; train for 600 epochs (based on convergence analysis).
3. Extract community assignments (or embeddings → KMeans).
4. Evaluate via GO/KEGG enrichment, motif preservation.
5. Repeat with 5 random seeds; report mean ± std of all metrics.

### Strategy 2: Semi-Supervised with GO-derived Labels (GCN/GAT)

Since no partition ground truth exists, we derive pseudo-labels from GO:

1. Assign each gene/protein its most specific (deepest) GO Biological Process term.
2. Group genes by top-level GO slim categories (e.g., "cell cycle", "signal transduction") — these become pseudo-community labels.
3. Use 20% of labeled nodes for training, predict the rest.
4. Report AMI of predictions vs GO slim labels, and separately report GO enrichment of the GNN's own discovered communities.

This is an approximation — GO slim categories are not true communities — but
it lets GCN/GAT participate in the benchmark and tests whether graph-based
label propagation recovers functional modules.

### Strategy 3: Baselines (MLP + KMeans, Node2Vec + KMeans)

- **MLP + KMeans:** Uses node features only (expression profiles or GO vectors). Tests whether features alone recover functional modules.
- **Node2Vec + KMeans:** Uses topology only. Tests whether random-walk embeddings capture biological modularity.

### Summary Table

| Method | Training | Features | Labels | Works on all 4 networks? |
|---|---|---|---|---|
| DMoN, MinCut | Transductive unsupervised | Yes | No | Yes |
| VGAE, DGI | Transductive unsupervised | Yes | No | Yes |
| GCN, GAT | Semi-supervised (GO slim pseudo-labels) | Yes | Partial | Yes |
| MLP + KMeans | Feature clustering only | Yes | No | Yes |
| Node2Vec + KMeans | Topology embedding | No | No | Yes |

### Node Features (detailed in Section 3.6)

| Network | Primary features | Dimension |
|---|---|---|
| GTEx | PCA-reduced TPM expression | 128 |
| Human GRN | TPM expression + TF/target role | ~130 |
| BioGRID | GO term binary vectors | ~5,000 (sparse) |
| SIGNOR | GO term vectors + sign-aware degree | ~5,004 |

---

## 6. Dataset Characterization (ref: Tanis et al., arXiv:2412.19419)

For every test network, compute and report two complexity measures that
jointly predict GNN effectiveness:

| Measure | Definition | Why it matters |
|---------|-----------|----------------|
| Edge homophily | Fraction of edges connecting same-community nodes | High → message-passing helps; low → it can hurt. Our mesoscale rewiring keeps homophily constant while changing internal structure — a unique control. |
| Feature SNR | Between-class feature variance / within-class variance (Eq. 58 in Tanis et al.) | High SNR → features alone informative (MLP baseline strong). Low SNR + low homophily = "hard" regime where GNNs struggle. |

Compute both for: CORA original, every mesoscale variant, every weaken variant,
and both biological networks. This creates a 2D landscape (homophily × SNR)
that contextualizes all results.

Key prediction from the literature: GNNs add the most value over baselines in
the "medium difficulty" regime (moderate homophily, moderate SNR). In "easy"
(high both) classical methods already suffice; in "hard" (low both) nothing
works well. Our weaken experiments sweep from easy → hard, letting us verify
this prediction in the community detection setting.

---

## 7. Metrics Summary

| Metric                    | Type           | Where used                     |
|---------------------------|----------------|--------------------------------|
| AMI                       | Partition      | All benchmarks                 |
| NMI                       | Partition      | All benchmarks                 |
| ARI                       | Partition      | Biological (cell types)        |
| ECS                       | Per-node       | Mesoscale, weaken, perturbation|
| Modularity                | Unsupervised   | All (note: biased toward Louvain) |
| Conductance               | Boundary       | Mesoscale                      |
| Edge homophily            | Dataset char.  | All (characterize, don't optimize) |
| Feature SNR               | Dataset char.  | CORA + biological (wherever features exist) |
| Silhouette score          | Embedding      | Biological                     |
| GO enrichment             | Functional     | Biological only                |
| KEGG pathway enrichment   | Functional     | Biological only                |
| Stability (variance)      | Robustness     | Perturbation experiments       |
| Runtime                   | Efficiency     | All benchmarks                 |

---

## 8. What Makes This Paper Publishable

1. **Novel experimental design:** Mesoscale and weaken experiments go beyond
   standard "run algorithms, report AMI." They dissect **what information**
   each method uses.

2. **Feature ablation:** Systematically quantifying when attributes help
   community detection. No prior benchmarking paper does this cleanly across
   mesoscale/weaken/perturbation variants.

3. **Actionable takeaway:** "Use GNNs when node attributes are informative and
   topology is degraded. Use classical when topology alone is strong."
   Practical guidance for practitioners.

4. **Homophily × SNR landscape:** Plotting results on the 2D (homophily, SNR)
   plane generalizes findings beyond specific datasets. This connects our
   benchmark to the broader GNN literature (Tanis et al., 2024).

5. **Biological validation at scale:** Four real biological networks (GTEx
   co-expression, Human GRN, BioGRID PPI, SIGNOR signaling) evaluated with
   domain-specific metrics — GO/KEGG enrichment, feed-forward loop preservation,
   sign coherence, and gene knockout perturbation. This elevates the paper from
   a CS benchmarking exercise to an interdisciplinary contribution with direct
   relevance to systems biology.

6. **Reproducibility:** LFR (seeded), CORA (public), biological data
   (downloadable). Code published.

7. **Fair GNN evaluation:** All GNNs are established SOTA methods with published
   loss functions. Per-graph transductive training (not pre-trained on secret
   data), feature ablation (not unfair feature advantage), multiple seeds
   (not cherry-picked).

8. **Hierarchical scale analysis:** The three-paradigm comparison (scale-blind
   classical, K-guided GNNs, hierarchy-aware NestedSBM) on the (μ1, μ2)
   grid is a novel experimental design that tests multi-scale detection —
   an aspect missing from standard flat-community benchmarks.
