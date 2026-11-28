# Prometheus-RL

**A research-oriented reinforcement learning framework, built one algorithm at a time.**

Status: Block 11 / v1.2.0 — PPO and episodic REINFORCE implemented; full multi-seed benchmarks pending.

Prometheus-RL aims to implement interpretable, testable reinforcement learning algorithms from their mathematical definitions. We use Python for orchestration, NumPy for numerical computation, Gymnasium for environments, and PyTorch for differentiable function approximation. We will implement the learning logic ourselves rather than wrap an existing RL training library. CartPole is the first deep-RL benchmark; Atari pixel observations and vector-observation policy gradients are now supported.

## Getting started (Ubuntu)

Requires Python 3.11+, Git, and optionally the GitHub CLI and a TeX installation. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
prometheus-rl --version
prometheus-rl
python -m pytest -q
ruff check .
```

The baseline does not install heavy numerical dependencies. For the full RL dependency stack, `python -m pip install -e '.[dev,rl]'` installs the planned RL stack. For a CPU-only PyTorch build, consult the official [installation selector](https://pytorch.org/get-started/locally/) first. Use `python -m pip freeze` to record the exact environment for experiments.

## Mathematical foundations

An agent interacting with an environment is represented by a Markov decision process $(\mathcal S,\mathcal A,p,r,\gamma)$. At time $t$ it observes a state, chooses an action, receives a reward and transitions. Its goal is to maximize expected discounted return, not merely the next reward.

$$G_t=\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}$$

For a finite discounted MDP, the optimal action-value function is a fixed point of the Bellman optimality operator:

$$Q^*(s,a)=\mathbb E[R_{t+1}+\gamma\max_b Q^*(S_{t+1},b)\mid S_t=s,A_t=a].$$

Q-learning approximates this fixed point with sampled transitions. DQN replaces the table with a neural network, uses replay to sample prior transitions and a target network to stabilize bootstrap targets. See the [LaTeX manuscript](docs/paper/main.tex) for derivations, assumptions, convergence conditions, worked examples, CartPole dynamics and evaluation methodology.

## Architecture

```text
prometheus-rl/
├── .github/
│   ├── workflows/ci.yml
│   └── pull_request_template.md
├── docs/
│   └── paper/main.tex
├── src/prometheus_rl/
│   ├── __init__.py
│   ├── __main__.py
│   ├── core/ (mdp.py, planning.py, examples.py)
│   ├── agents/ (q_learning.py, dqn.py, dqn_demo.py)
│   ├── environments/ (gridworld.py)
│   ├── networks/ (q_network.py, demo.py)
│   ├── memory/ (replay.py, batch.py)
│   └── training/ (tabular.py)
├── tests/ (test_smoke.py, test_mdp.py, test_q_learning.py, test_q_network.py, test_replay*.py)
├── .gitignore
├── CONTRIBUTING.md
├── pyproject.toml
└── README.md
```

Implemented: `core` for finite transition kernels and planning, `agents` for tabular Q-learning and DQN, `environments` for GridWorld, and `training` for the tabular loop. `networks` implements neural approximators and `memory` implements replay storage and tensor batching. Future modules: `environments` for benchmarks; `training` and `evaluation` for experiment loops; `visualization` for plots and trajectories. Keep policy, data collection and optimization separate so individual components can be tested independently.

## Block 01: finite MDP and dynamic programming

Run `prometheus-rl mdp-demo` (or `python -m prometheus_rl mdp-demo`). The delayed-reward example has optimal values `(4.5, 5.0, 0.0)`: wait for the larger discounted reward rather than cash out immediately.

```python
from prometheus_rl.core import evaluate_policy, value_iteration
from prometheus_rl.core.examples import delayed_reward

mdp = delayed_reward()
optimal = value_iteration(mdp, tolerance=1e-9)
random_at_start = evaluate_policy(mdp, ((0.5, 0.5), (1.0,), ()))
print(optimal.values)          # (4.5, 5.0, 0.0)
print(random_at_start.values)  # (3.25, 5.0, 0.0)
```

`FiniteMDP.transitions[state][action]` is a tuple of `Outcome(probability, next_state, reward)` objects; state and local action indices start at zero. Empty action rows represent terminal states, whose continuation value is zero. Nonempty distributions must sum to one within absolute tolerance 1e-12; floating-point arithmetic is used. The optional initial distribution supports expected initial return. `sample` takes a caller-owned `random.Random` instance.

`bellman_optimality` and `bellman_policy` compute one synchronous update. `value_iteration` and `evaluate_policy` stop when the Bellman residual divided by `1 - gamma` is below the requested tolerance (an error bound in exact arithmetic, not an interval-arithmetic certificate). They raise on iteration-budget exhaustion. Greedy ties choose the lowest local action index. Policy rows contain probabilities; terminal rows are empty. The implementation supports finite discounted problems only (`0 <= gamma < 1`) and needs no third-party runtime dependencies. Planning uses the known model; it is not model-free learning.

## Block 02: model-free tabular Q-learning

Run `prometheus-rl q-demo` to train five independent agents on a deterministic 4×4 GridWorld. Each cell has four actions (north, east, south, west); boundary moves stay in place. The start is the top-left cell and the goal is the bottom-right cell. Entering the goal yields +1; every other action yields -0.05. The discount factor is 0.9. No model probabilities or planning results enter the agent's update.

```python
from prometheus_rl.environments.gridworld import gridworld
from prometheus_rl.training.tabular import train, evaluate

mdp = gridworld()
result = train(mdp, episodes=1000, seed=0)
held_out = evaluate(mdp, result.agent, episodes=100, seed=10000)
print(result.agent.values)  # immutable Q-table snapshot
```

The behavior policy is epsilon-greedy (linear decay from 1 to 0.1); alpha is constant at 0.2. The episode cap is 100 steps. A true terminal transition masks the bootstrap, whereas a time limit ends data collection without masking it. Evaluation freezes the table and compares greedy actions with uniform random actions. Separate random generators control environment and action sampling.

Observed CPU results from the five seeds 0–4 (100 held-out episodes per policy per seed):

| Seed | Greedy discounted return | Random discounted return | Greedy steps | Goal rate |
|---|---:|---:|---:|---:|
| 0 | 0.385735 | -0.359807 | 6 | 100% |
| 1 | 0.385735 | -0.388980 | 6 | 100% |
| 2 | 0.385735 | -0.338829 | 6 | 100% |
| 3 | 0.385735 | -0.393943 | 6 | 100% |
| 4 | 0.385735 | -0.390399 | 6 | 100% |

The optimal discounted start value from value iteration is 0.385735. Each trained policy attains it, but this deterministic toy experiment does not establish general Q-table convergence or performance on other environments. The 37-test suite checks hand-calculated updates, masking, exploration, reproducibility, planning agreement and frozen evaluation.

## Block 03: neural action-value approximation

Install the optional neural dependency. For a CPU-only Linux setup:

```sh
python -m pip install 'torch>=2.5,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,neural]'
python -m pytest -q
prometheus-rl network-demo
```

The default `QNetwork(4, 2)` uses two 128-unit ReLU hidden layers and a linear output (17,410 trainable parameters). It accepts a floating-point tensor `[n]` or `[B,n]` and returns `[A]` or `[B,A]`. Match the input device and dtype to the model; no softmax is applied. Hidden widths may be configured, including an empty sequence for linear regression. PyTorch is imported only for neural commands and modules.

```python
import torch
from prometheus_rl.networks import QNetwork, fixed_target_loss

model = QNetwork(4, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
observations = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
actions = torch.tensor([0, 1])
targets = torch.tensor([1., -1.])
optimizer.zero_grad(set_to_none=True)
loss = fixed_target_loss(model(observations), actions, targets)
loss.backward()
optimizer.step()
```

`select_q` gathers one action per observation. `fixed_target_loss` computes mean Huber loss (threshold 1), detaches the targets, and rejects shapes that would accidentally broadcast. The demo uses a fixed CPU batch, seed 7 and 80 SGD updates; observed loss was 0.44456556 initially and below 1e-8 after training (PyTorch 2.10.0 CPU). This is regression, not a DQN agent or CartPole learning result. Tests include parameter counts, invalid shapes, selected-output gradients, detached targets and a central finite-difference check of an early-layer weight. The dependency-light test job skips neural tests; the separate CI neural job installs CPU PyTorch and runs the full suite.

## Block 04: uniform experience replay

```sh
prometheus-rl replay-demo
python -m pytest -q
```

A fixed-capacity ring retains the most recent transitions. Appending takes amortized constant time after constructing the immutable transition, and never shifts existing entries. The buffer samples distinct occupied slots uniformly without replacement using its own seeded random generator. Equal-valued transitions in different slots remain distinct sampling units. An undersized buffer raises an error rather than silently returning a smaller batch; the DQN agent waits for sufficient warm-up. Storage needs no PyTorch. Tensor conversion requires the neural extra installed in Block 03.

```python
from prometheus_rl.memory import ReplayBuffer, Transition
from prometheus_rl.memory.batch import collate

memory = ReplayBuffer(capacity=10_000, observation_dim=4, action_dim=2, seed=7)
memory.append(Transition((0., 0., 0., 0.), 1, 1.,
                         (0.1, 0., 0.02, 0.), False, True))
batch = collate(memory.sample(1))
print(batch.observations.shape)  # torch.Size([1, 4])
print(batch.bootstrap_mask)     # tensor([1.])
```

| Tensor | Shape | Default dtype |
|---|---|---|
| observations, next_observations | [B,n] | float32 |
| actions | [B] | int64 |
| rewards | [B] | float32 |
| terminated, truncated | [B] | bool |
| bootstrap_mask | [B] | float32 |

Both flags end the episode, but only `terminated` removes the bootstrap term. Time-limit truncation here assumes an external limit, not a terminal state of a finite-horizon MDP. Store the actual final observation, not the reset observation. `collate` supports CPU or a specified PyTorch device and float32/float64. Stored observation tuples own their numeric data; the tensor batch is newly allocated. This is a flat-vector, single-threaded buffer, not an image-optimized Atari store (prioritized replay is a separate class). `clear` resets the data and insertion index but preserves the RNG stream.

For the demonstration, six experiences with rewards 0 through 5 enter a capacity-four buffer. The retained rewards are `[2, 3, 4, 5]`, and seed 7 samples `[2, 4, 5]` in the tested Python runtime. The corresponding masks are `[1, 0, 1]`; reward 4 is a termination and reward 5 a truncation. The tests exercise wraparound, immutability, validation, reproducibility, sampling frequencies, flag combinations, dtype and network compatibility. Block 05 uses this buffer for bootstrapped optimization.

## Block 05: vanilla deep Q-learning

Install the neural extra (`python -m pip install -e '.[dev,neural]'`) and run
`prometheus-rl dqn-demo`. The agent owns independent online and frozen target
networks, uniform replay, Adam, epsilon-greedy action selection, gradient clipping
and a counter of successful optimizer steps. The environment loop is deliberately
separate and arrives in Block 06.

```python
from prometheus_rl.agents.dqn import DQNAgent, DQNConfig
from prometheus_rl.memory import Transition

agent = DQNAgent(DQNConfig(), seed=7)
s = (0., 0., 0., 0.)
a = agent.act(s, epsilon=0.1)
agent.observe(Transition(s, a, 1., (0.1, 0., 0., 0.), False))
result = agent.update()  # None until replay warm-up is satisfied
agent.save("agent.pt")
restored = DQNAgent.load("agent.pt")
```

Default configuration: two hidden layers of 128, discount 0.99, Adam learning
rate 0.001, capacity 50,000, batch size 64, warm-up 1,000 retained transitions,
hard target copy every 250 successful optimizer steps, global gradient norm
bound 10. Epsilon is supplied by the caller; training schedules are Block 06.
The target is `r + gamma * max Q_target(s_next)` for nonterminal rows and `r`
otherwise. External truncation does not disable the bootstrap. Targets are
computed without gradients; only online parameters are optimized. The reported
`grad_norm` is the pre-clipping norm. Ties use the lowest action index.

The checkpoint contains configuration, both network state dictionaries, Adam
state and optimizer-step count. Loading uses PyTorch's weights-only loader;
load only trusted checkpoint files. Replay contents, random streams, environment
state and training schedules are **not** saved: this supports inference and
warm-start training, not exact experiment resumption.

The demo uses four transitions from a fully covered deterministic two-stage MDP.
Immediate cash yields 2, waiting yields 0 then a choice of 5 or 1; discount 0.9.
After 400 updates with seed 7, the start Q-values were (2.0000, 4.5000), the
waiting state's maximum was 5.0000, and greedy action 1 was selected (CPU,
PyTorch 2.10.0). The targets are bootstrapped, unlike Block 03, but this is a
fixed-replay sanity experiment, not online exploration or a CartPole benchmark.
Tests cover the analytic solution, target flags, warm-up, gradient clipping,
seed isolation, synchronization and checkpoint round trips.

## Block 06: online CartPole training

The DQN agent now collects transitions from Gymnasium `CartPole-v1` and updates
its weights online. The training loop accepts a factory for isolated testing;
observations are snapshotted, true terminations disable bootstrapping, and
truncations end episodes without disabling the continuation target. The initial
environment reset uses the training seed; later resets continue that RNG stream.
The schedule is linear from epsilon 1 to 0.05 over 30,000 interactions, with
one optimization attempt every four steps after the existing replay warmup.
The default run is 100,000 interactions, not a fixed number of episodes.

```sh
python -m pip install -e '.[dev,rl]'
prometheus-rl cartpole train --steps 1200 --eval-episodes 3 --output runs/smoke
prometheus-rl cartpole train --steps 100000 --seed 0 --output runs/cartpole-0
prometheus-rl cartpole evaluate runs/cartpole-0/model.pt --seed 2000000 --episodes 100
prometheus-rl cartpole play runs/cartpole-0/model.pt --episodes 3
```

Each run uses a new output directory. It contains `model.pt`, `episodes.csv`,
`summary.json`, `returns.png`, and `loss.png` (omit the plots with `--no-plots`).
The CSV records completed episodes only; an unfinished episode at the exact
interaction budget still contributes transitions and updates. `summary.json`
records individual evaluation scores, means, sample standard deviations,
configuration, dependency versions, Git revision, elapsed time and thread count.
The final greedy model and a uniform random policy are evaluated using the same
held-out reset seeds, without optimizer or replay updates. Evaluation does not
select checkpoints. Human playback needs a desktop display and pygame.

For the original five-seed benchmark, run:

```sh
for seed in 0 1 2 3 4; do
  prometheus-rl cartpole train --steps 100000 --seed "$seed" \
    --eval-seed "$((1000000 + 1000 * seed))" --eval-episodes 100 \
    --output "runs/cartpole-$seed"
done
```

The target remains mean >=475 for **each** agent; the smoke run is not evidence
of achieving it. Locally, 111 tests passed and the real-Gymnasium smoke test was
skipped because Gymnasium could not be installed in the build environment. A
controlled one-step online task learned the better action (mean reward 1.0 vs
random 0.5 on 50 episodes); this validates the loop, not CartPole control. The CI
CartPole job installs the real dependency and performs a short interaction test.
No CartPole performance numbers are claimed before the full benchmark runs.

## Block 07: Double and Dueling DQN

Two independent switches allow a 2-by-2 factorial comparison. `--double`
changes the bootstrap rule, selecting an action with the online network and
evaluating it with the frozen target. `--dueling` changes the network to shared
ReLU features and separate linear value and advantage heads. Both switches are
off by default, and existing vanilla checkpoints remain readable. New checkpoint
configuration records both switches. Checkpoints still omit replay and RNG state.

```sh
prometheus-rl variants-demo
prometheus-rl cartpole train --double --dueling --steps 1200 \
  --eval-episodes 3 --output runs/variants-smoke
```

The four variants reproduce the fixed-replay analytic example (Q at start:
2.0000, 4.5000; wait value: 5.0000; greedy action: 1), in 400 optimizer steps
with seed 7. This is a correctness check, not evidence of a performance gain.
The default dueling architecture has 17,539 parameters versus 17,410 for vanilla.

For an equal-interaction-budget comparison, use matched training and evaluation
seeds. Each output directory must be new. This is a proposed benchmark, not a
completed experiment:

```bash
for seed in 0 1 2 3 4; do
  for variant in vanilla double dueling double-dueling; do
    flags=()
    case "$variant" in
      double) flags=(--double);;
      dueling) flags=(--dueling);;
      double-dueling) flags=(--double --dueling);;
    esac
    prometheus-rl cartpole train "${flags[@]}" --steps 100000 \
      --seed "$seed" --eval-seed "$((1000000 + 1000 * seed))" \
      --eval-episodes 100 --output "runs/$variant-$seed"
  done
done
```

Compare per-seed paired returns from `summary.json` rather than pooling all
episodes as independent training runs. Keep all seeds, including failures. The
same interaction budget does not imply identical parameter counts or compute.
The local suite passed 126 tests; one Gymnasium test was skipped because the
package is unavailable. Remote CI and real CartPole performance remain pending.

## Block 08: Prioritized experience replay

Uniform replay remains the default. Enable proportional prioritized replay with
`--prioritized` (compatible with `--double` and `--dueling`). The dependency-free
`PrioritizedReplayBuffer` extends the circular store with sum and minimum trees.
Each occupied slot has mass `(abs(TD error) + epsilon)**alpha`. New transitions
receive the largest raw priority observed since the last clear (at least 1).
Sampling draws independently with replacement, so a minibatch can repeat slots.
The normalized importance weight is `(minimum_mass / sampled_mass)**beta`.
The DQN computes a weighted mean of per-example Huber losses, and updates sampled
priorities from detached, pre-optimization TD errors; duplicate slots use the
maximum absolute error. Beta anneals from 0.4 to 1 over 25,000 optimizer updates.

```sh
prometheus-rl per-demo
prometheus-rl cartpole train --double --dueling --prioritized --steps 1200 \
  --eval-episodes 3 --output runs/per-smoke
prometheus-rl cartpole train --double --dueling --prioritized --steps 100000 \
  --per-alpha 0.6 --per-beta-start 0.4 --per-beta-steps 25000 \
  --seed 0 --output runs/per-0
```

Both tree operations (one insertion or priority update, one draw) take O(log N)
time and trees use O(N) memory in addition to stored experiences. Slot indices
are not persistent handles: update before another append in the synchronous
training loop. Alpha zero gives uniform draws **with replacement**, not the
original buffer's sampling without replacement. At beta=1 the normalized weights
correct the relative sampling bias up to a common scale conditional on the buffer;
this does not correct behavior-policy or changing-data distribution bias. Very
extreme priority ratios may be limited by floating-point precision.

Local tests cover empirical frequencies, weights, eviction, duplicate draws,
invalid input, all four DQN variants, checkpoint configuration and an exact
weighted Huber loss calculation. Checkpoints still omit replay and RNG state.
The real CartPole smoke test is added to CI. Full comparative performance is
not yet measured. For a paired comparison, run the same configuration with and
without `--prioritized`, using equal interaction budgets and five matched seeds.

## Block 09: custom game environments

The original tabular `gridworld()` remains unchanged. The new online `HazardGrid`
and `Snake` implement dependency-free five-return step semantics; `GameEnv`
provides Gymnasium spaces and optional RGB-array or Pygame human rendering.
All eight DQN configurations share the same training loop. Install `.[dev,rl]`.

```sh
prometheus-rl snake train --double --dueling --prioritized --steps 100000 --output runs/snake-0
prometheus-rl snake play runs/snake-0/model.pt --episodes 3
prometheus-rl gridworld train --double --dueling --prioritized --steps 30000 --output runs/grid-0
prometheus-rl gridworld play runs/grid-0/model.pt --episodes 3
```

Snake uses a 6x6 board, three relative actions (left, straight, right), and
76 observations: 36 normalized ordered body ranks, 36 food indicators and
four heading indicators. Food spawns uniformly on vacant cells. Eating yields
+1, ordinary moves -0.01, collision -1. Filling the board is terminal; the
150-step external limit truncates. Moving into the vacating tail is legal.
The rank encoding preserves the order of the body, not just occupancy.

Hazard GridWorld uses a fixed 5x5 map, 25 one-hot position features and four
absolute actions (north, east, south, west). Walls and boundaries block moves;
ordinary steps cost 0.04, hazards terminate with -1, and the goal terminates
with +1. The external step limit is 80. Custom layouts can be supplied in
Python, with S=start, G=goal, H=hazard, #=wall and .=free. An optional
`shaping_gamma` uses negative Manhattan distance as potential, with zero
terminal potential. The CLI deliberately uses unshaped rewards; when enabling
shaping in Python, match the agent discount and report base rewards separately.
The finite time limit is an external interruption, not an observed clock.

The following single-seed CPU pilot uses the dependency-free cores (not Gym
adapters), 10,000 interactions per game, seed 0, 30 evaluation episodes at
seed 1,000,000, hidden widths (64,64), batch 32, replay 10,000, warmup 256,
target interval 100, epsilon decay 7,000, double/dueling/PER enabled and other
DQN defaults. The table reports undiscounted shaped-by-task-rule returns (no
potential shaping), not score or multi-seed evidence. Each run performed 2,437
updates. Exact pilot configuration can be reconstructed using `make_game`,
`DQNConfig`, `TrainConfig` and `run_experiment(factory=..., environment_id=...)`.

- gridworld: greedy mean 0.7200; random mean -1.9067.
- snake: greedy mean -0.1377; random mean -0.8007.

The Snake pilot does not establish game mastery. Gymnasium checker and headless RGB rendering tests require Gymnasium; human display requires a graphical session. Use Ctrl-C to stop playback early.

## Block 10: visual DQN / Atari

Install the optional Atari stack (Gymnasium, ALE, OpenCV, NumPy and PyTorch):

```sh
python -m pip install -e '.[dev,rl,atari]'
ruff check .
python -m pytest -q
OMP_NUM_THREADS=1 prometheus-rl atari train --game breakout --steps 1200 \
  --eval-episodes 1 --eval-max-steps 1000 --output runs/atari-smoke
```

The ALE adapter uses `ALE/Breakout-v5` or `ALE/Pong-v5`, with base frame skip 1,
preprocessing frame skip 4, sticky action probability 0.25, random reset no-ops,
max-pooling of the most recent two frames, grayscale resizing to 84×84, and four
stacked frames. Life loss alone does not terminate an episode. No automatic FIRE
reset is used: the policy must learn any required start/serve actions. See the
[ALE interface](https://ale.farama.org/main/gymnasium-interface/) and
[Gymnasium preprocessing docs](https://gymnasium.farama.org/api/wrappers/misc_wrappers/).

The visual network uses 32 8×8/4, 64 4×4/2 and 64 3×3/1 convolutions,
then 3136 → 512 → action values. Byte observations become float32 in the forward
pass with division by 255. Default Double-Dueling DQN uses uniform replay;
`--vanilla` turns off both improvements. Pixel-based PER is not implemented.
Independent current/next frame stacks are copied into the circular replay:
56,448 bytes per transition, or 282,240,000 bytes (~269 MiB) at capacity 5,000,
excluding metadata, batches, models and optimizer state. This is deliberately
simpler than frame-deduplicated replay and is not suitable for million-item
buffers. Training rewards are clipped to their sign; episode scores are raw.

```sh
OMP_NUM_THREADS=1 prometheus-rl atari train --game pong --steps 100000 \
  --seed 0 --output runs/pong-0
prometheus-rl atari evaluate runs/pong-0/model.pt --episodes 5
prometheus-rl atari play runs/pong-0/model.pt --episodes 3
```

Use `--device cuda` only with a compatible CUDA PyTorch installation. The output
contains `model.pt`, `episodes.csv`, and `summary.json`, including elapsed training
time, interaction and update budgets, versions, commit, seed, and raw greedy/random
evaluation scores. Evaluation is capped at 27,000 decisions per episode by default;
a smaller cap in smoke tests is not comparable with a full-game score. The 100,000
decision command is a pilot, not a mastery claim; it corresponds to at most 400,000
repeated-action frames excluding reset no-ops. Checkpoints are not exact training
resumption snapshots: replay and RNG state are not persisted. Visual checkpoints
are distinct from vector DQN checkpoints.

Synthetic pixel tests verify shapes, gradients, storage, flags, target synchronization,
checkpoint restoration and online collection. Real ALE tests and training smoke
are configured in CI; no real-game scores or playback were verified in the build
container. Full Atari experiments need considerably larger compute budgets.

## Block 11: policy gradients and PPO

PPO is an on-policy actor–critic algorithm, independent of the DQN replay buffers.
The actor produces categorical logits for discrete actions or a diagonal Gaussian
for continuous actions, with a tanh transformation and affine scaling to bounded
Gymnasium action spaces. The critic estimates a scalar state value. The rollout
stores behavior-policy log probabilities and fixed value predictions. GAE uses
separate termination and truncation masks; the final observation of a truncation
is bootstrapped, but advantages never cross episode resets.

```bash
prometheus-rl ppo train --game cartpole --steps 2048 --eval-episodes 1 --output runs/ppo-smoke
prometheus-rl ppo train --game cartpole --steps 50000 --seed 0 --output runs/ppo-cartpole-0
prometheus-rl ppo train --game pendulum --steps 50000 --seed 0 --output runs/ppo-pendulum-0
prometheus-rl ppo evaluate runs/ppo-cartpole-0/model.pt --episodes 100
prometheus-rl ppo play runs/ppo-cartpole-0/model.pt --episodes 3
```

Snake and GridWorld are also available through `--game`. Training collects 256
transitions per rollout by default (short final batches are supported), performs
four epochs over minibatches of 64, and uses clipped likelihood-ratio loss,
value MSE, entropy regularization, Adam and gradient-norm clipping. For the
continuous policy, the entropy bonus is the base Gaussian entropy proxy, not
true squashed-policy entropy. Evaluation uses argmax or squashed Gaussian mean,
not sampling from the trained stochastic policy. The checkpoint stores the model,
optimizer, game/configuration and update count; it does not save environment or
random-generator state for exact resumption. Each run uses a new output directory
and saves `model.pt`, `episodes.csv`, `updates.csv` and `summary.json`.

Episodic REINFORCE is additionally available through the Python API
`training.ppo.train_reinforce`; it optimizes discounted reward-to-go with the
outer discount and treats truncation as the finite-horizon cutoff. PPO has no
such Monte Carlo cutoff assumption for its critic bootstrap. Unit tests cover
both policy objectives, GAE masks, continuous action statistics, checkpointing,
and learning on a two-action bandit. Real CartPole/Pendulum smoke tests run in
CI; no solved-environment or comparative-performance claim is made here.

## Roadmap and acceptance criteria

| Block | Branch | Acceptance criteria |
|---|---|---|
| 0 | main | Package, CLI, smoke tests, CI and mathematical manuscript |
| 1 | feature/01-mdp-core | Finite MDP, validated transitions, Bellman operators, exact value iteration, analytic toy problem |
| 2 | feature/02-tabular-q-learning | Seeded Q-learning on GridWorld or FrozenLake, terminal masking and learning evaluation |
| 3 | feature/03-neural-networks | PyTorch Q-network, batched shapes, gradient tests |
| 4 | feature/04-experience-replay | Bounded replay buffer, deterministic sampling, correct transition flags |
| 5 | feature/05-dqn-agent | DQN targets, epsilon-greedy policy, target synchronization and optimization tests |
| 6 | feature/06-cartpole-training | Reproducible CPU training and held-out evaluation vs random policy |
| 7 | feature/07-double-dueling-dqn | Double and dueling variants, ablations |
| 8 | feature/08-prioritized-replay | Priority sampling and importance weights |
| 9 | feature/09-custom-environments | GridWorld and Snake with environment contract tests |
| 10 | feature/10-atari | Visual DQN and preprocessing, compute-budget reporting |
| 11 | feature/11-policy-gradients | Policy gradients and PPO |
| 12 | feature/12-benchmark-dashboard | Comparative experiments and visualization |

## Reproducibility and evaluation

For every training run we will record environment ID/version, configuration, seed, device, dependency versions, Git commit, training steps, and wall-clock time. Evaluation uses separate seeds and greedy policies, reports individual returns and aggregate statistics across independent training seeds, and compares equal interaction budgets. CartPole initial target: mean undiscounted return at least 475/500 across 100 held-out episodes for each of five independently trained agents (a project goal, not an achieved result). No results will be claimed until experiments are actually run.

## Build the paper

```sh
sudo apt install latexmk texlive-latex-extra
cd docs/paper
latexmk -pdf -halt-on-error -interaction=nonstopmode main.tex
```

The paper is a living technical manuscript, not a published study. Each algorithm PR must update it with any new mathematical assumptions and actual findings.

## References

- Sutton & Barto (2018), [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html).
- Watkins & Dayan (1992), [Q-learning](https://doi.org/10.1007/BF00992698).
- Mnih et al. (2015), [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236).
- van Hasselt et al. (2016), [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461).
- [Gymnasium: Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

## Contributing and licensing

See [CONTRIBUTING.md](CONTRIBUTING.md). The repository owner has not selected a license yet; no open-source license is asserted. Choose a license before inviting external contributions or advertising this as open source.
