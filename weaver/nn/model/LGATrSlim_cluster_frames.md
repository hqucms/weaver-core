# Per-cluster light-cone frames for L-GATr-slim (proposal, not implemented)

Design note for extending the light-cone frame of `LGATrSlim.py` (`vector_coord="lightcone"`)
from single jets to event-level inputs, so that half-precision attention stays accurate there
too. Nothing in this note is implemented yet.

## 1. Background: why the jet light-cone frame works

L-GATr-slim contracts Lorentz vectors with the Minkowski metric in the q/k/v and block RMS
norms, the vector GLU gates, and attention logits (`q_i . k_j`). For nearly collinear, nearly
massless constituents the product

    p_i . p_j = E_i E_j (1 - cos theta_ij) ~ E_i E_j theta^2 / 2

is a small difference of two large numbers. Rounding Cartesian components to bf16/fp16 (e.g.
when attention runs in half precision) destroys it: on an 800-step checkpoint, bf16 attention
gave 30% logit error, fp16 10%.

`get_lightcone_frame` maps every vector, spurions included, with a per-jet orthogonal matrix T
to `(x+, x-, x1, x2)` about the jet axis `n`, and every contraction uses the light-cone metric
`eta' = T eta T^T` (`_apply_metric`, `_lightcone_product`). Because every operation is either
linear over channels, elementwise, or a bilinear form `v^T eta w`, the network computes **the
same function** (fp64 difference ~1e-12); only the conditioning changes: the small
`E - p.n` is formed once in float64 and stored as a number of its own.

Measured on JetClass (flex backend, compiled, batch 2048, RTX 5090; 2-epoch runs):

| config | val acc | ms/step |
|---|---|---|
| fp32 | 0.8481 | 170 |
| bf16 AMP, Cartesian (attention pinned to fp32) | 0.8286 / 0.8345 | 143 |
| bf16 AMP + light-cone (attention half precision) | 0.8465 - 0.8483 | 74 |
| + `vector_linear_fp32="embed"` | 0.8471 / 0.8473 | 64 |
| + `vector_linear_fp32=False`, bf16 AMP | 0.8474 | ~61 (61.1 measured with fp16 attention) |
| + `vector_linear_fp32=False`, fp16 AMP | 0.8477 | ~61-64 (est.) |

This relies on one axis being close to *all* particles, which holds for a jet but not for an
event with several jets.

## 2. The problem for event-level inputs

1. **Degenerate axis.** If the summed 3-momentum is ~0 (balanced e+e- event, or an empty
   event), `n = p / |p|.clamp_min(1e-12)` becomes 0, the first two rows of T coincide, and T is
   no longer orthogonal: every vector maps to ~`(E/sqrt2, E/sqrt2, 0, 0)`. Same for `e1` when
   `n` is exactly along the beam (`rho < 1e-12`). No error, no NaN: silently wrong. Needed fix
   regardless of this proposal: fall back to `n = z` when `|p| < eps * sum_i |p_i|`, and to
   `e1 = x` when `n || z`, so T is always orthogonal.
2. **No conditioning benefit.** For any non-degenerate `n` the frame is still exact, but jets
   point in different directions; particles inside each jet are collinear with each other but
   at a large angle to `n`, so their products cancel as badly as in Cartesian coordinates, and
   half-precision attention loses accuracy again.

## 3. Why not per-particle frames

Per-token operations (all GEMMs, norms, gates) work unchanged with a *different* frame per
token: all light-cone frames share the same metric `eta'`, and these operations only mix
channels of one token or contract a token's vector with itself.

Attention is the only operation that pairs vectors of different tokens:

    q_i . k_j = q'_i^T eta' R_ab k'_j,   R_ab = T_a T_b^T

A fused kernel computes plain dot products, so `q_i` and `k_j` must be in one basis. Options:

- per-particle frames, transforming every key into every query's frame: O(N^2 d) extra work and
  memory (a materialized pair bias); loses flash/flex fusion. Not viable.
- map everything back to one shared frame: cheap, but brings the cancellation back.
- **share a frame among particles that are collinear with each other** -- this proposal.

## 4. Proposal: per-cluster light-cone frames

1. **Cluster** each event's particles into K groups: anti-kT jets (precomputed in the data
   pipeline, e.g. R = 0.4), or on-the-fly kNN / kT grouping in (eta, phi). Unclustered
   wide-angle particles form one extra group. The clustering must be fixed per event (all
   layers use the same assignment).
2. **One light-cone frame per (event, cluster)**: `get_lightcone_frame` vectorized over clusters,
   with the degenerate-axis fallback of section 2. Each particle's vectors are stored in its
   cluster's frame from the input embedding on; the per-token layers are untouched.
3. **Attention per query cluster**: for the queries of cluster a, rotate *all* of the event's
   keys and values into frame a in fp32 (`R_ab` on the 4-component axis of the vector parts;
   scalar parts unchanged), then cast to the attention dtype. Intra-cluster pairs get the full
   light-cone benefit; cross-cluster pairs are at wide angles and well-conditioned in any frame.
   The attention output then comes out in frame a, i.e. the query token's own frame.
4. **Kernel layout**: one varlen segment per (event, cluster): q-length `N_a`, kv-length
   `N_event`. Flash varlen takes separate `cu_seqlens_q` / `cu_seqlens_k`; for flex, a block
   mask over the concatenated (cluster-copy) key axis. Standard fused kernels still apply.
5. **Spurions / global token**: a single-token cluster each, with any frame (e.g. the event
   frame or Cartesian); as keys they are rotated like every other token.

### Exactness

The network computes the same function for **any** clustering and any choice of (orthogonal,
light-cone-metric) frames: a poor or even random clustering only forgoes the precision
benefit, it never changes the result. The frames need not be learned, equivariant, or even
good -- a much weaker requirement than in approaches where the frames are part of the model
(LLoCa).

Unit test to add: Cartesian vs per-cluster model with identical weights, fp64, random and
physical clusterings, dense and packed layouts -> outputs and gradients equal to ~1e-12.

## 5. Generality and limits

- Precision loss comes from Minkowski products of nearly lightlike, nearly parallel vectors
  (jets, subjets, showers); clustering at the collinearity scale covers exactly that.
- Not fixed (same as the current jet frame):
  - single-particle masses `2 x+ x- - x_perp^2` still cancel for light hadrons (the input
    embedding was flagged by probes, though training with it in half precision was fine for
    jets);
  - pairs close to each other but far from their cluster axis keep a residual factor
    ~(theta / delta theta)^2;
  - forward particles against the beam spurion.
- LLoCa (`port-lgatr/lloca`) learns per-particle Lorentz frames for expressivity and transports
  features between frames, but its attention contracts in a shared frame; its frame-transport
  code could be reused, but it does not by itself address half precision.

## 6. Computational cost (per attention layer; event with N particles in K clusters)

| component | cost |
|---|---|
| frame construction | O(B K), float64, negligible |
| per-token ops (GEMMs, norms, gates) | unchanged |
| attention FLOPs | unchanged: sum_a N_a x N = N^2 |
| keys/values in K frames | K x K/V activation memory and bandwidth (also saved for backward); 4x4 rotations of vector parts, O(K N C_v 16) flops |
| kernels | standard varlen/flex; KV token count K N per event |

- Jet tagging (K = 1): identical to the current implementation.
- Full events (N ~ 100-1000, K ~ 3-10): attention K/V tensors grow K-fold. Rough, unmeasured
  guess: +10-30% step time and several-fold attention activation memory -- still most of the
  ~2x half-precision speedup over fp32 attention.
- Fallback when memory is the binding constraint: `vector_coord="cartesian"` (attention pinned to
  fp32), half precision everywhere else.

## 7. Implementation sketch

- `get_lightcone_frames(fourmomenta, mask, cluster_idx, num_clusters)` -> `(N_ev, K, 4, 4)`,
  float64, with the degenerate-axis fallback; `get_lightcone_frame` becomes the K = 1 case.
- Tagger forward: per-token frame `T[event, cluster_idx[token]]`; transform momenta and
  spurions (einsum in float64) before the dense/packed branch, exactly as today.
- `SlimSelfAttention`: when a cluster layout is given, build the K rotated copies of k/v in
  fp32 (`R_ab = T_a T_b^T`, one 4x4 per (event, a, b) pair), cast, and call the varlen / flex
  kernel with per-(event, cluster) q segments and per-(event, cluster) copies of the event's
  kv. Scalar channels of k/v are copied, not rotated.
- Everything else (`_apply_metric`, `_lightcone_product`, the unpinned light-cone attention path,
  `vector_linear_fp32`) is reused as is.

Validation plan: (1) the fp64 equivalence test above; (2) probe logit error under bf16/fp16
attention on a trained event-level checkpoint, Cartesian vs single event frame vs per-cluster;
(3) matched 2-epoch trainings (fp32, bf16 AMP Cartesian, bf16 AMP per-cluster) with repeats --
run-to-run spread on JetClass was up to ~0.6 pp; (4) step time and peak memory vs K.
