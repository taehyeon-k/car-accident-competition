Please add and test a new **phase-based temporal loss** for Stage 2 using the current strong compact temporal model and frozen geometry-tuned DINOv3-S features. Do not redesign the backbone; isolate the effect of this loss.

For sampled temporal positions \(t=1,\dots,T\), define 3 phase labels from GT ENTRY \(e^*\) and COLLISION \(c^*\):

- `PRE` if \(t < e^*\)
- `BETWEEN` if \(e^* \le t < c^*\)
- `POST` if \(t \ge c^*\)

Add a 3-class phase head and train with weighted CE:

\[
L_{phase}=CE_{weighted}(p_t,y_t)
\]

using initial class weights:

\[
w_{PRE}=1,\quad w_{BETWEEN}=3,\quad w_{POST}=1.
\]

From phase probabilities derive directional transition scores:

\[
s_E(t)=\log p_{t-1}^{PRE}+\log p_t^{BETWEEN}
\]

\[
s_C(t)=\log p_{t-1}^{BETWEEN}+\log p_t^{POST}.
\]

Normalize each across time with softmax and supervise them using Gaussian targets centered at the GT ENTRY/COLLISION sampled positions with \(\sigma=1\):

\[
L_{transition}
=
-\frac12
\left[
\sum_t g_E(t)\log q_E(t)
+
\sum_t g_C(t)\log q_C(t)
\right].
\]

Keep the existing direct ENTRY/COLLISION heads as a weak auxiliary and supervise them with the same Gaussian targets:

\[
L_{direct}.
\]

Also add a small monotonicity penalty against illegal backwards phase transitions:

\[
L_{mono}
=
\frac1{T-1}\sum_t
[
p_t^{BETWEEN}p_{t+1}^{PRE}
+
p_t^{POST}p_{t+1}^{PRE}
+
p_t^{POST}p_{t+1}^{BETWEEN}
].
\]

Use initially:

\[
\boxed{
L =
1.0L_{phase}
+0.75L_{transition}
+0.25L_{direct}
+0.05L_{mono}
+0.5L_{side}
+0.5L_{evasion}
}
\]

At inference, also test a **structured phase decoder** instead of only argmax event logits. For every valid \(e<c\), score how well the entire clip fits:

\[
PRE_{<e}\rightarrow BETWEEN_{e:c}\rightarrow POST_{\ge c}
\]

using summed log phase probabilities, optionally adding transition and direct-event scores.

Compare against the current baseline using the same split/seeds. Report overall metrics and especially NEXAR / >1000-frame ENTRY-COLLISION accuracy, normalized MAE, catastrophic false-event errors, and whether the structured decoder corrects long-video wrong-event selections. Keep all old experiments untouched and save this as a separate ablation.