## Setup
- There are $N$ agents.  
- The ground truth attacker set is $A = \text{atk\_idx}$, with $|A| = k$.  
- At each timestep $t = 1, \dots, T$, the prediction set is  
  $$
  P_t = \text{pred}[t]
  $$  
  i.e., the set of agents predicted as attackers at time $t$. The size of $P_t$ can vary.

---

## 1. Basic Set Metrics (per-timestep + aggregate)

For each timestep $t$:
$$
\text{TP}_t = |P_t \cap A|, \quad
\text{FP}_t = |P_t \setminus A|, \quad
\text{FN}_t = |A \setminus P_t|
$$

$$
\text{Precision}_t = \frac{\text{TP}_t}{\text{TP}_t+\text{FP}_t+\varepsilon}, \quad
\text{Recall}_t = \frac{\text{TP}_t}{\text{TP}_t+\text{FN}_t+\varepsilon}, \quad
\text{F1}_t = \frac{2 \cdot \text{Precision}_t \cdot \text{Recall}_t}{\text{Precision}_t + \text{Recall}_t + \varepsilon}
$$

**Jaccard (IoU):**
$$
\text{Jaccard}_t = \frac{|P_t \cap A|}{|P_t \cup A|+\varepsilon}
$$

**Aggregate (macro over time):**
$$
\overline{\text{F1}} = \frac{1}{T}\sum_{t=1}^{T}\text{F1}_t, \qquad
\overline{\text{Jaccard}} = \frac{1}{T}\sum_{t=1}^{T}\text{Jaccard}_t
$$

> These metrics naturally handle variable prediction set sizes and are robust to imbalance when $k \ll N$.

---

## 2. Weighted Early-Detection Version (optional)

To emphasize earlier detection, assign higher weights to earlier timesteps.  
Let $\gamma \in (0,1]$ (e.g., $\gamma=0.95$):

$$
w_t = \gamma^{\,t-1}, \quad
\text{WF1} = \frac{\sum_{t} w_t \cdot \text{F1}_t}{\sum_t w_t}, \quad
\text{WJaccard} = \frac{\sum_{t} w_t \cdot \text{Jaccard}_t}{\sum_t w_t}
$$

---

## 3. Event-level Metric (per-agent first detection + false positives)

Define the **first detection time** for each true attacker $i \in A$:

$$
\tau_i = \min\{t:\ i \in P_t\}, \quad \tau_i=\infty \ \text{if never detected.}
$$

**Timeliness score per attacker:**
$$
c_i = \begin{cases}
1 - \dfrac{\tau_i - 1}{T}, & \tau_i \le T \\
0, & \text{otherwise}
\end{cases}
$$

**False positive rate per normal agent** $j \in A^c$:
$$
b_j = \frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[j \in P_t]
$$

**Latency-Aware Detection Score (LADS):**
$$
\text{LADS} =
\underbrace{\frac{1}{k}\sum_{i\in A}c_i}_{\text{earlier is better}}
- \lambda \underbrace{\left(\frac{1}{N-k}\sum_{j\in A^c} b_j\right)}_{\text{fewer false positives is better}}, \quad
\lambda \in [0.5,2] \ (\text{default } \lambda = 1)
$$

> Intuition: Earlier detection yields higher $c_i$; frequent false positives reduce the score.  
> LADS $\in [-\lambda,1]$.

---

## 4. Stability (optional)

Measures prediction "flips" across timesteps:

$$
\text{FlipRate} = \frac{1}{N(T-1)} \sum_{i=1}^{N}\sum_{t=2}^{T} 
\mathbf{1}\big[\mathbf{1}_{\{i \in P_t\}} \ne \mathbf{1}_{\{i \in P_{t-1}\}}\big]
$$

> Lower is better. You can report this as a "regularizer" to discourage unstable predictions.

---

## Recommended Reporting Template (for papers/reports)

- **Main metric:** $\overline{\text{Jaccard}}$ or $\overline{\text{F1}}$ (choose one as the headline metric, report the other as supplement).  
- **Early detection ability:** WJaccard ($\gamma=0.95$) or LADS ($\lambda=1$).  
- **False positives & stability:** Report average false positive rate $\frac{1}{N-k}\sum b_j$ and FlipRate.  
- **Appendix/Table:** Per-agent $\tau_i$ distribution (median, P90) and missed detection ratio.