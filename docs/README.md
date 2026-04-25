# LIF Neural Network Framework

A PyTorch implementation of Leaky Integrate-and-Fire (LIF) spiking neural networks with a **custom (non-autograd) backward pass**. Backpropagation is handled explicitly per learning rule, and weight updates are driven by hand-derived spike-time gradients (`SequentialSingleSpikeLR`) and a non-Hebbian plasticity-induction term (`PlasticityInduction`).

---

## 1. Repository Layout

```
src/
├── main.py                                # entry point (Experiment runner)
├── data/
│   ├── data_sample.py                     # raw (vector, label) container
│   ├── dataset/
│   │   ├── dataset.py                     # base Dataset (pickled DataSamples)
│   │   ├── random_dataset.py              # i.i.d. random vectors with ±1 labels
│   │   └── mnist_dataset.py
│   └── spike/
│       ├── spike_data.py                  # per-neuron spike train
│       └── spike_sample.py                # per-sample spike train + collate_fn + digest_batch
├── encoders/
│   ├── identity_encoder.py
│   └── spike/
│       ├── latency_encoder.py             # single-spike time encoding
│       └── rate_encoder.py
├── network/
│   ├── topology/
│   │   ├── network.py                     # base Network (Module list)
│   │   ├── sequential_network.py          # time-stepped forward / explicit backward
│   │   ├── neuron.py                      # NeuronLayer = (kernel ∘ connection ∘ activation)
│   │   ├── connection.py                  # abstract linear connection
│   │   └── simple_connection.py           # dense weight W ~ 𝒩(0, 1)
│   ├── kernel/
│   │   ├── seq_den.py                     # 2-stage dendritic (synaptic + soma) filter
│   │   ├── seq_lif.py                     # single-stage LIF filter
│   │   └── functional/{cpu_based,gpu_based}.py   # numba/CUDA kernels
│   ├── learning/
│   │   ├── learning_rule.py               # abstract: forward / backward / reset
│   │   ├── single_spike_lr.py             # vectorised single-spike LR
│   │   └── sequential/
│   │       ├── single_spike_lr.py         # time-stepped SingleSpikeLR (max-pooling over t)
│   │       └── plasticity_induction.py    # silent-neuron plasticity rule
│   ├── activation/sub_activation.py       # f(x) = x − ϑ ; identity gradient
│   ├── loss/binary_loss.py                # margin loss   L = ReLU(−y·(s − ϑ_loss))
│   └── optimizer/momentum_opt.py          # custom Polyak momentum
└── pipeline/
    ├── sequential_pipeline.py             # epoch / batch loop, evaluation
    └── callbacks/visualization_callback.py
```

The training harness (`Experiment`, `Pipeline`, `Environment`, `RunStatus`, `Metric`, `YAMLSerializable`) is provided by the external [`experiment_manager`](https://pypi.org/) package. Components are constructed from YAML via `*_factory.py` modules.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $T,\;\Delta t$ | total simulation time and step (env: `T=500`, `dt=1`) |
| $L = \lceil T/\Delta t \rceil$ | number of time steps (`tools.utils.SEQ_LEN`) |
| $N_\text{in},\;N_\text{out}$ | layer input / output dimensions |
| $B$ | mini-batch size |
| $\tau_s,\;\tau_m,\;\tau$ | synaptic, membrane, and (single-stage) LIF time constants |
| $\beta = 1 - \Delta t/\tau_s$ | synaptic leak (per step) |
| $\alpha = 1 - \Delta t/\tau_m$ | somatic leak (per step) |
| $\vartheta$ | spiking / loss threshold (env: `v_th=1.0`) |
| $\mathbf{x}_t \in \{0,1\}^{B\times N_\text{in}}$ | input spike tensor at step $t$ |
| $\mathbf{W}\in\mathbb{R}^{N_\text{in}\times N_\text{out}}$ | synaptic weights (`SimpleConnection.w`) |
| $y \in \{-1,+1\}$ | binary label |

Tensor convention is `(B, N_in)` per timestep, stacked into `(B, N_out, L)` over time.

---

## 3. Encoding — Latency Code

[latency_encoder.py:42-62](src/encoders/spike/latency_encoder.py#L42-L62) implements deterministic single-spike latency encoding. For each input feature $x_i \in [0, x_\text{max}]$ (clamped):

$$
t_i \;=\; \left\lfloor \frac{x_i}{x_\text{max}} \cdot (L-1) \right\rfloor,
\qquad
s_i(t) \;=\; \delta_{t,\,t_i}\;\;(t = 0,\dots,L-1).
$$

A larger value spikes **later**. Each input neuron emits exactly one spike. The output `SpikeSample` is materialised as a sparse `(L, N_in)` binary tensor by [spike_sample.py:36-45](src/data/spike/spike_sample.py#L36-L45).

---

## 4. Neuron Dynamics

A `NeuronLayer` ([neuron.py:34-63](src/network/topology/neuron.py#L34-L63)) is a composition

$$
\text{layer}(\mathbf{x}_{1:L}) \;=\; \sigma \,\circ\, \text{conn} \,\circ\, K(\mathbf{x}_{1:L}),
$$

where $K$ is a stateful temporal filter (`Kernel`), `conn` is a learned linear map, and $\sigma$ is the activation. The kernel is iterated in a Python generator that yields one timestep at a time ([seq_den.py:90-107](src/network/kernel/seq_den.py#L90-L107)).

### 4.1 Single-stage LIF (`SequentialLeakyKernel`)

[seq_lif.py:42-79](src/network/kernel/seq_lif.py#L42-L79) — one IIR pole:

$$
v_t \;=\; \beta\, v_{t-1} \;+\; x_t,
\qquad
v_0 = 0,\qquad
\beta = 1 - \frac{\Delta t}{\tau}.
$$

Optional spike emission $s_t = \mathbb{1}[v_t \ge \vartheta]$. With `hard_reset=False` (default in all current configs) the membrane is **not reset** after a spike — i.e. the kernel is a pure linear filter and spikes are reported only as side information.

### 4.2 Dendritic (two-stage) kernel (`SequentialDenKernel`)

[seq_den.py:65-88](src/network/kernel/seq_den.py#L65-L88) and the fused numba kernel [cpu_based.py:12-32](src/network/kernel/functional/cpu_based.py#L12-L32) implement a *current → voltage* cascade:

$$
\begin{aligned}
i_t &= \beta\, i_{t-1} + x_t \\
v_t &= \alpha\, v_{t-1} + i_t
\end{aligned}
\qquad
\beta = 1 - \tfrac{\Delta t}{\tau_s},\;\;\alpha = 1 - \tfrac{\Delta t}{\tau_m}.
$$

Closed-form (no reset) — $v_t$ is the **double-exponential post-synaptic potential** convolved with the input:

$$
v_t \;=\; \sum_{k=0}^{t}\;\sum_{j=0}^{k}\;\alpha^{\,t-k}\,\beta^{\,k-j}\,x_j
\;=\;\bigl(h_\alpha * h_\beta * x\bigr)_t,
$$

with $h_\beta[n]=\beta^n,\;h_\alpha[n]=\alpha^n$ for $n\ge 0$. With $\alpha\ne\beta$ the impulse response is

$$
h(t) \;=\; \frac{\alpha^{\,t+1} - \beta^{\,t+1}}{\alpha - \beta},
$$

equivalent in the continuous limit to $h(t) = \frac{1}{\tau_m-\tau_s}\bigl(e^{-t/\tau_m} - e^{-t/\tau_s}\bigr)$.

Default config: $\tau_s = 10$, $\tau_m = 2.5$ ⇒ $\beta = 0.9$, $\alpha = 0.6$.

The companion analytic check `assimulate_response` ([seq_lif.py:113-133](src/network/kernel/seq_lif.py#L113-L133)) builds the closed-form response

$$
r_i(t) \;=\; \sum_{t_k \in \mathcal{S}_i,\; t_k \le t}\;\frac{1}{\tau}\,e^{-(t-t_k)\Delta t/\tau}.
$$

### 4.3 Connection (`SimpleConnection`)

[simple_connection.py:34-61](src/network/topology/simple_connection.py#L34-L61) — linear projection per timestep:

$$
\mathbf{u}_t \;=\; \mathbf{v}_t\,\mathbf{W} \in \mathbb{R}^{B\times N_\text{out}}.
$$

`partial_forward` returns $\mathbf{u}_t$. `forward` *also* returns the per-rule spike contribution $\sum_r \text{LR}_r.\text{forward}(\mathbf{v}_t,\mathbf{u}_t)$ — see §5. Weights are initialised i.i.d. $\mathcal{N}(0,1)$ ([connection.py:69-83](src/network/topology/connection.py#L69-L83)).

### 4.4 Activation (`SubtractActivation`)

[sub_activation.py:13-18](src/network/activation/sub_activation.py#L13-L18):

$$
\sigma(u) \;=\; u - \vartheta_\sigma,
\qquad
\sigma'(u) \;=\; 1.
$$

This is a passthrough whose only effect is shifting the decision boundary; with `threshold: 0.0` (current configs) it is the identity.

---

## 5. Learning Rules

Each `LearningRule` exposes `forward(input, output)` and `backward(input, E)` and is collected per-layer in `lr_list`. The connection sums their gradients ([simple_connection.py:79-92](src/network/topology/simple_connection.py#L79-L92)):

$$
\nabla_{\mathbf{W}} \;=\; \frac{1}{B}\sum_{b=1}^{B}\;\sum_{r}\; G_r^{(b)},\qquad
\nabla_{\mathbf{v}} \;=\; \mathbf{E}\,\mathbf{W}^\top,
$$

where $\mathbf{E}$ is the upstream error (loss-side) tensor and $G_r^{(b)}$ is rule $r$'s per-sample gradient.

### 5.1 `SequentialSingleSpikeLR` — argmax-spike rule

[sequential/single_spike_lr.py](src/network/learning/sequential/single_spike_lr.py)

**Forward.** Online max-pool over time. With $\mathbf{u}_t = \mathbf{v}_t\mathbf{W}$ at step $t$, maintain

$$
M^{(b)}_n \;=\; \max_{t\le L}\;u^{(b)}_{n,t},
\qquad
t^*_{b,n} \;=\; \arg\max_{t\le L}\;u^{(b)}_{n,t},
$$

and store the **input snapshot at the argmax**: $\widehat{\mathbf{v}}^{(b)}_{:,n} \;=\; \mathbf{v}_{t^*_{b,n}}^{(b)}$. The forward returns $M - \vartheta$ which the loss uses as the per-sample "spike value".

**Backward.** Given upstream error $\mathbf{E} \in \mathbb{R}^{B\times N_\text{out}\times 1}$ ([single_spike_lr.py:71](src/network/learning/sequential/single_spike_lr.py#L71)):

$$
G^{(b)} \;=\; \widehat{\mathbf{V}}^{(b)} \cdot \mathbf{E}^{(b)}
\;\in\; \mathbb{R}^{N_\text{in}\times N_\text{out}},
\qquad
\widehat{\mathbf{V}}^{(b)}_{i,n} \;=\; \widehat{v}^{(b)}_{t^*_{b,n}, i}.
$$

This is the gradient of $M^{(b)}_n$ w.r.t. $\mathbf{W}_{:,n}$ under the locally-linear assumption $\partial t^*/\partial \mathbf{W} = 0$ — i.e. the spike-time argmax is treated as a stop-gradient and only the input pattern at that timestep contributes to the weight gradient. This is the spike-timing analogue of straight-through max-pool gradients.

### 5.2 `PlasticityInduction` — induced plasticity for silent neurons

[sequential/plasticity_induction.py](src/network/learning/sequential/plasticity_induction.py)

**Forward.** Maintain a **silent-neuron mask** $\mathbf{S}^{(b)}\in\{0,1\}^{N_\text{in}\times N_\text{out}}$ initialised to all ones. At each step zero out columns of neurons that have crossed threshold:

$$
\mathbf{S}^{(b)}_{:,n} \;\leftarrow\; \mathbf{0}\quad\text{if}\;\;u^{(b)}_{n,t} > \vartheta\;\text{at any}\;t.
$$

The forward returns 0 (no contribution to the output statistic).

**Backward.** Let $E^{(b)}_n$ be the per-output-neuron error from the loss. For each output $n$ that the loss says should have fired but didn't ($E^{(b)}_n < 0$), apply the constant-magnitude push:

$$
G^{(b)}_{:,n} \;=\; -\,\varepsilon \cdot \mathbf{S}^{(b)}_{:,n}
\quad\text{iff}\quad E^{(b)}_n < 0.
$$

Because momentum-SGD does $\mathbf{W}\leftarrow \mathbf{W} - \eta\,\nabla$, the resulting update is $\Delta \mathbf{W}_{:,n} = +\eta\varepsilon\,\mathbf{S}^{(b)}_{:,n}$ — **strengthening every input weight that did not yet drive the silent neuron**. Effectively this is a non-Hebbian "wake the dead neuron" term that fires only when the standard spike-based gradient would otherwise be zero.

`epsilon` is the only hyperparameter; sweeps over $\varepsilon\in\{10^{-1},\dots,10^{-5}\}$ are visible in `configs/grid_experiment` and `configs/focused_induction`.

---

## 6. Loss — Hinge Margin on Spike Value

[binary_loss.py:19-62](src/network/loss/binary_loss.py#L19-L62) implements a binary perceptron-style hinge:

$$
s^{(b)} \;=\; M^{(b)} - \vartheta_\text{loss}
\qquad\Longrightarrow\qquad
\boxed{\;\mathcal{L}^{(b)} \;=\; \bigl[\,-\,y^{(b)}\, s^{(b)}\,\bigr]_+\;}
$$

with $y\in\{-1,+1\}$, ReLU $[\cdot]_+$, and $\vartheta_\text{loss}=1$. The classifier is the sign decision $\hat{y} = \text{sign}(s)$ (mapped to $\pm1$).

**Backward** ([binary_loss.py:43-62](src/network/loss/binary_loss.py#L43-L62)) is the subgradient

$$
\frac{\partial \mathcal{L}^{(b)}}{\partial s^{(b)}}
\;=\;
\begin{cases}
-y^{(b)} & \text{if } y^{(b)} s^{(b)} \le 0\\
0 & \text{otherwise}
\end{cases},
$$

returned with shape `(B, 1, 1)` and propagated through `model.backward` ([sequential_network.py:58-64](src/network/topology/sequential_network.py#L58-L64)).

---

## 7. Optimization — Polyak Momentum

[momentum_opt.py:30-61](src/network/optimizer/momentum_opt.py#L30-L61). With learning rate $\eta$ and momentum coefficient $\mu$ (defaults `lr=0.01`, `momentum=0.99`):

$$
\mathbf{m}_t \;=\; \mathbf{g}_t + \mu\,\mathbf{m}_{t-1},\qquad
\mathbf{W}_t \;=\; \mathbf{W}_{t-1} - \eta\,\mathbf{m}_t,\qquad \mathbf{m}_0 = \mathbf{0}.
$$

Note this is the unscaled additive form (no $(1-\mu)$ factor on the gradient). With $\mu=0.99$ the effective step size is $\eta/(1-\mu) = 100\eta$ in steady state, which is why all tuned configs use small base $\eta$ (e.g. $0.01$). `Adam` is also available ([optimizers.py](src/network/optimizer/optimizers.py)) but unused in current experiments.

`ConstantLR` is the default scheduler; `PolynomialLR` is wired in factory but disabled.

---

## 8. End-to-End Training Pipeline

[sequential_pipeline.py](src/pipeline/sequential_pipeline.py) — single-layer, two-rule pipeline. For each batch:

1. **Forward** through the sequential network ([sequential_network.py:32-41](src/network/topology/sequential_network.py#L32-L41)):
   - For each layer: stream `kernel(x_{1:L})` as a generator, project with $\mathbf{W}$, accumulate the rule outputs ($M$ from `SequentialSingleSpikeLR`, silent mask updates from `PlasticityInduction`).
   - Stack per-timestep activations into shape `(B, N_out, L)`.
   - Return `(outputs, final_spikes)` where `final_spikes` is the per-sample max-margin tensor.
2. **Loss** $\mathcal{L} = [\,-y(s-\vartheta_\text{loss})\,]_+$ on `final_spikes`.
3. **Backward** ([binary_loss.backward → sequential_network.backward](src/network/topology/sequential_network.py#L58-L64)):
   - Loss returns $-y\,\mathbb{1}[ys\le 0]$.
   - Activation passes the gradient through (identity).
   - Connection sums per-rule gradients: $\nabla_\mathbf{W} = \frac{1}{B}\sum_b (G_\text{SingleSpike}^{(b)} + G_\text{Plasticity}^{(b)})$.
4. **`optimizer.step()`** applies the momentum update.

**Validation** evaluates `final_accuracy = (sign(s) == y).mean()` on a held-out subset (`validation_split=0.9` in the optimal config — i.e. trains on 10%, validates on 90%, intentional small-data regime). Early-stop trigger at $\ge 99.9\%$ validation accuracy ([sequential_pipeline.py:187-190](src/pipeline/sequential_pipeline.py#L187-L190)).

---

## 9. Hyperparameters that Matter (empirical)

From `outputs/grid_experiment` and `outputs/optimal_experiment`:

| Knob | Range explored | Optimum (val acc 92.59%) |
|---|---|---|
| learning rate $\eta$ | $\{0.1, 0.05, 0.01\}$ | **0.01** |
| batch size $B$ | $\{1, 4, 16\}$ | **1** |
| plasticity $\varepsilon$ | $\{10^{-1},\dots,10^{-5}\}$ | **$10^{-4}$** |
| epochs | up to 200 | early-stop on plateau |

Architecture knobs are fixed across the grid: $N_\text{in}=500$, $\tau_s=10$, $\tau_m=2.5$, $\vartheta=1$, momentum $0.99$, `SubtractActivation(0)`, `BinaryLoss(threshold=1)`. `validation_split=0.9` keeps the training fold deliberately small.

---

## 10. Running an Experiment

```bash
cd src
python main.py --config-dir ../configs/optimal_experiment
```

Configs are layered: `base.yaml` is the model/pipeline/optimizer/loss spec, `trials.yaml` declares hyperparameter overrides per trial, `env.yaml` injects $T,\Delta t,\vartheta$, etc. Each trial materialises into `outputs/<experiment>/<trial>/{configs, logs, artifacts}`.
