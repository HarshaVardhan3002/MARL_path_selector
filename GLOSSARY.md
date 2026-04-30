# Glossary — ML/RL Terms You'll Hear

Grounded for someone with an AI/DS bachelor's. Not a textbook — just enough that when Claude Code says something, you know whether it's reasonable or hallucinated.

## The big picture

**Reinforcement Learning (RL)** — an agent takes actions in an environment, gets rewards, learns a policy that maximizes total reward over time. Different from supervised learning: no fixed input→output labels, the agent generates its own training data by acting.

**Multi-Agent RL (MARL)** — multiple agents share an environment. Each agent's actions affect the others. Much harder than single-agent because the environment is non-stationary from any one agent's view (other agents are also learning).

**Policy** — the function the agent learns. Input: state. Output: action (or distribution over actions). What ships at deployment.

**Value function / Critic** — separate function that estimates "how good is being in this state." Used during training to tell the policy what was a good action. Thrown away at deployment in our setup.

**Episode** — one full run of the environment from reset to done. Lots of steps. Training is many episodes.

## Core concepts you'll see in our codebase

**State (or observation)** — what an agent sees at one timestep. In our project: this agent's recent RTT readings, current path's properties, time-since-path-refresh, etc. Note: in MARL we distinguish *observation* (what one agent sees, partial) from *state* (the full global picture, which only the critic sees during CTDE training).

**Action** — what the agent does. In our project: which path to use for the next interval. Discrete action space (pick one of K available paths).

**Reward** — scalar feedback after each action. In our project: a weighted sum of throughput term + congestion-contribution penalty + path-diversity bonus. We log them separately so we can ablate.

**Reward shaping** — adding extra terms to the reward to guide learning. Risky because agents will hack any shaping term you give them. The "lossless operating point" framing from Keshvadi's paper is what protects us — we have a formal target, not a hand-wave.

**Trajectory / rollout** — sequence of (state, action, reward, next state) tuples from one episode. The training data RL generates from itself.

## CTDE and friends

**CTDE** — Centralized Training, Decentralized Execution. During training, a critic sees the full global state. The policy sees only its local observation. After training, the critic is discarded. The policy ships alone. This is the architecture that makes our deployment story work.

**Parameter sharing** — instead of training N separate policies for N agents, train one policy that all agents use. Each agent feeds its own observation in and gets its own action out. Faster, more sample-efficient, and matches our "deploy the same artifact everywhere" story.

**MAPPO** — Multi-Agent Proximal Policy Optimization. Our chosen algorithm. PPO (a single-agent algorithm) extended to multi-agent with a centralized critic.

**MADDPG** — Multi-Agent Deep Deterministic Policy Gradient. The other obvious MARL choice. We didn't pick it because it's finicky, harder to defend in a thesis, and our actions are discrete (path indices) — DDPG-family is happier with continuous actions.

## PPO terms (because we're using MAPPO)

**Policy gradient** — the family of RL algorithms that directly optimize the policy via gradient ascent on expected reward. PPO is one of these.

**On-policy vs off-policy** — PPO is *on-policy*: it learns from data the current policy generated. Throws data away after each update. Sample-inefficient but stable. Off-policy methods (DQN, SAC) reuse old data via replay buffers.

**PPO clip ratio** — PPO's safety mechanism. After computing how much the policy wants to change, clip the change to a small range (default ±0.2). Prevents the policy from updating too aggressively and collapsing. If you see "clip ratio = 0.2" that's the standard; rarely needs tuning.

**GAE (Generalized Advantage Estimation)** — a way to compute "how much better was this action than the average action from this state." Better signal than raw reward. Has two knobs: γ (gamma, discount factor, default 0.99) and λ (lambda, default 0.95). "GAE lambda tuning" means adjusting that λ. Higher λ = lower bias, higher variance. Default usually fine.

**Entropy bonus** — small reward for the policy being uncertain. Forces exploration. If you see the policy "collapsing to one action," you raise the entropy coefficient.

**Value function loss** — the critic's loss term. Critic is trained to predict expected future reward. If this loss doesn't go down during training, the critic isn't learning, and the policy gradients are noisy.

## Distillation terms

**Teacher / student** — in distillation, the large trained model (teacher) generates predictions on a dataset, and a smaller model (student) is trained to match those predictions. We're distilling the MARL policy into XGBoost.

**Soft labels** — the teacher's full output distribution (probabilities over actions), not just the argmax. Distillation usually works better with soft labels.

**Behavioral cloning** — train the student to match the teacher's action choices on states the teacher visited. Simplest distillation. Risk: covariate shift (states the student visits at runtime differ from states the teacher trained on).

**XGBoost** — gradient-boosted decision tree library. Trees are sequential, each one fixes the previous one's mistakes. We're using it for distillation because trees compile to integer comparisons (no floating-point math, no neural net runtime needed at deployment).

**Treelite** — compiles XGBoost models into C code. Future-work for us (Phase 4 stretch ring).

## Things you'll see in TensorBoard

**Episode reward** — total reward summed over an episode. Should trend up during training. If flat, something's wrong.

**Policy loss / value loss** — the two main loss curves. Policy loss should decrease then plateau. Value loss should decrease and stay decreased.

**Policy entropy** — uncertainty in the policy. Starts high (random), decreases as the policy commits to actions. If it crashes to zero too fast, the policy collapsed (bad).

**KL divergence** — how much the policy changed in one update. PPO clips this implicitly. If you see KL spikes, something's unstable.

**Approximate KL / clip fraction** — diagnostics specific to PPO. clip fraction tells you what fraction of updates hit the clip ratio. If it's >0.3 consistently, the learning rate is too high.

## Things specific to our project

**Path fingerprint** — SCION's identifier for a specific path through the network. Hash of the AS-hop sequence. Stable for the path's lifetime; new path = new fingerprint.

**Path discrepancy** — Herschbach paper term. The phenomenon where A→B and B→A see different sets of available paths due to routing policy.

**Lossless operating point** — Keshvadi paper concept. The configuration where Efficiency, Convergence, and Loss Avoidance are jointly satisfied. Our reward target.

**Path churn** — paths appearing/disappearing over time. Empirically ~8.6h average lifetime per Herschbach. The thing that makes our action space non-stationary.

**Action masking** — at any timestep, only some paths exist. The policy's output over "all possible paths" gets masked to only currently-valid paths. Standard technique, we'll use it.

## When Claude Code says something and you're not sure

If the term isn't here and isn't obvious, ask. "What does X mean and is using X here standard?" gets you a real answer. Don't nod through unfamiliar acronyms — that's how silent design errors creep in.
