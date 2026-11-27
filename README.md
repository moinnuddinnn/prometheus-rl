# Prometheus-RL

**A research-oriented reinforcement learning framework, built one algorithm at a time.**

Status: Block 04 / v0.5.0 — finite MDPs, tabular Q-learning, neural Q-networks and uniform experience replay implemented.

Prometheus-RL aims to implement interpretable, testable reinforcement learning algorithms from their mathematical definitions. We use Python for orchestration, NumPy for numerical computation, Gymnasium for environments, and PyTorch for differentiable function approximation. We will implement the learning logic ourselves rather than wrap an existing RL training library. CartPole is the first deep-RL benchmark; Atari and policy gradients are later milestones.

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
│   ├── agents/ (q_learning.py)
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

Implemented: `core` for finite transition kernels and planning, `agents` for Q-learning, `environments` for GridWorld, and `training` for the tabular loop. `networks` implements neural approximators and `memory` implements replay storage and tensor batching. Future modules: `environments` for benchmarks; `training` and `evaluation` for experiment loops; `visualization` for plots and trajectories. Keep policy, data collection and optimization separate so individual components can be tested independently.

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

A fixed-capacity ring retains the most recent transitions. Appending takes amortized constant time after constructing the immutable transition, and never shifts existing entries. The buffer samples distinct occupied slots uniformly without replacement using its own seeded random generator. Equal-valued transitions in different slots remain distinct sampling units. An undersized buffer raises an error rather than silently returning a smaller batch; a future DQN training loop must wait for sufficient warm-up. Storage needs no PyTorch. Tensor conversion requires the neural extra installed in Block 03.

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

Both flags end the episode, but only `terminated` removes the bootstrap term. Time-limit truncation here assumes an external limit, not a terminal state of a finite-horizon MDP. Store the actual final observation, not the reset observation. `collate` supports CPU or a specified PyTorch device and float32/float64. Stored observation tuples own their numeric data; the tensor batch is newly allocated. This is a flat-vector, single-threaded buffer, not prioritized replay or an image-optimized Atari store. `clear` resets the data and insertion index but preserves the RNG stream.

For the demonstration, six experiences with rewards 0 through 5 enter a capacity-four buffer. The retained rewards are `[2, 3, 4, 5]`, and seed 7 samples `[2, 4, 5]` in the tested Python runtime. The corresponding masks are `[1, 0, 1]`; reward 4 is a termination and reward 5 a truncation. The tests exercise wraparound, immutability, validation, reproducibility, sampling frequencies, flag combinations, dtype and network compatibility. No DQN training has yet been performed.

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
