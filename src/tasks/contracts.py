"""Contract for RaceTask, library adapters and task-driven collectors.

Ownership
---------
RaceEnv owns physical cars, maps, spawning, sensing, race lifecycle, physical
terminal-car behavior, and the scenario's episode-ending policy. RaceTask owns
per-policy-agent observation, action, and reward composers, fixed controllers,
and advancing one joint policy decision. The native TorchRL environment translates
this contract and explicitly maps individual/shared learning rewards; learners own
policies, value estimates, returns, buffers, updates, budgets and checkpoints. RaceEnv
does not inherit from TorchRL. Training hooks and evaluation reporting stay outside the task.

Agent identities
----------------
``physical_agents`` is the stable ordered set of cars in RaceEnv. It is not
necessarily four cars. ``possible_agents`` is the stable ordered subset whose
actions the caller supplies; in existing experiments these are the learners.
``fixed_policy_agents`` is the complementary set, controlled inside the task.
Every physical car must have exactly one action owner. Explicit RaceEnv team
and trainable-agent metadata keep their existing meaning; adapter exposure is
not a replacement for those roles.

``agents`` contains currently active caller-controlled agents, in possible-agent
order. A finished/crashed agent leaves this set, but RaceEnv can keep its car
collidable or coasting. Every agent that acted gets exactly one AgentDecision,
including its terminal next observation. Later steps produce no decisions or
individual rewards for that inactive agent. Full physical diagnostics remain
available in snapshots and substeps.

Decision sequence
-----------------
Validate all supplied action IDs and values before changing composer/controller
state: keys must exactly equal ``agents``; values must be finite float32 vectors
of shape (2,) in [-1, 1]. Reject missing, extra, inactive, or fixed-agent actions.
Snapshot the current local observations and global state, then transform each
policy action exactly once. Evaluate each active fixed controller exactly once
on its raw observation; its output is already in the physical command units
(including the existing wheel-reference adapter when applicable).

Hold the resulting physical actions for at most ``action_repeat`` RaceEnv steps.
After each step, capture its StepFacts and compute individual rewards for the
agents that acted, using the existing build_reward_context and each composer's
local components. Compute configured shared team components once per physical
step with the designated team composer. Sum rewards and component breakdowns
over the actual substeps without additional discounting. Reward context retains
the decision's original local observation/normalized action and each physical
step's raw next observations, infos, timestep, and global state.

Stop repetition immediately when RaceEnv ends the episode or any car receiving
an action leaves RaceEnv.agents, including a fixed opponent. Survivors make their
next decision from the latest state. Integrated speed/wheel-acceleration commands
retain the configured nominal decision_dt = timestep * action_repeat, even when
an event shortens that decision; changing this would change existing experiments.
Composed observations are made once at reset and once after each decision.

Rewards and boundaries
----------------------
AgentDecision.individual_reward excludes shared components. TaskStep.team_reward
is their accumulated bonus, not the mean/sum of individual rewards. The adapter
preserves current mapping: individual mode uses individual rewards; team_shared
uses their sum or their sum divided by the configured team size, then adds the
shared bonus once. Inactive teammates contribute zero to that fixed denominator.
Keep the current restrictions on team terms and joint returns (2v2 MAPPO,
shared_team critic, team_shared rewards, joint returns, action_repeat=1); this
contract does not authorize relaxing scenario validation.

AgentDecision flags preserve RaceEnv's per-agent terminated/truncated signals
across executed substeps. Physical finish, collision, and boundary failure are
terminations; time-limit and no-progress stops are truncations. ``any_agent``
also terminates surviving decisions according to the existing scenario policy,
without inventing physical crashes in lifecycle records. If a policy agent is
removed solely by a joint ending rule with neither flag, expose a task truncation
and ``info['task_boundary'] = 'episode_policy'`` while preserving its physical
lifecycle facts. Final next observations/global state remain available for value
bootstrapping. This closure rule does not rewrite RaceEnv's lifecycle records.

TaskSnapshot.episode_done follows RaceEnv. An empty policy-agent set can precede
race completion under ``all_agents``: step({}) may then advance active fixed
controllers and report physical facts, with zero actor samples. Adapters must
handle this explicitly when translating to a library's episode lifecycle; they
must not silently change the scenario's ending policy. A collector budget or
rollout cut is not a task termination/truncation. Per-agent GAE bootstraps at a
truncation and stops recurrence across the reset. Existing joint team GAE instead
ends when no learner can act (including the finite race timeout), even if fixed
opponents remain; this stays a learner responsibility.

Reset and published data
------------------------
reset(seed=..., options=...) forwards RaceEnv's seeding/map/spawn options intact,
including SpawnPlan and episode indices. Curriculum scheduling builds those
options outside the task. Reset every stateful composer and fixed controller
once per reset before creating the initial composed observations. Each policy
agent owns an independent action composer. reset sets both episode counters to
zero, emits no rewards, and runs no policy inference. There is no automatic reset.
Reject step before reset or after episode_done; an explicit reset starts a new
episode. close releases the owned RaceEnv; render delegates to it.

Local composed observations and actions are float32 NumPy arrays. Centralized
state is a separate GlobalState with its existing ordering, masks, and vector
version; actors do not receive it through their composed observations. Snapshots
retain raw observations/infos for all physical cars, but exposed policy observation
keys exactly match their active agents. Returned arrays and nested payloads must
remain stable across later steps/resets: the task detaches reused buffers or
share immutable snapshots. Frozen dataclasses alone do not enforce deep ownership.

One TaskStep counts as one joint environment decision, len(decisions) counts
actor samples, len(substeps) counts physics steps, and elapsed_seconds reports
actual simulated time. These quantities remain separate for budgets/metrics.
Substeps preserve every executed physics step for race evaluation and rendering;
callers can observe them without moving training hooks into the task.

Extraction compatibility
------------------------
Preserve observation layout and checkpoint action/physics contracts, existing
reward contexts, fixed-controller units and zero-action exception fallback,
substep event ordering, and seeded map/spawn behavior during extraction. Changes
to controller failure handling or numerical behavior belong in separate changes.
Regression checks compare deterministic action traces through trainers and
RaceTask, covering terminal reward delivery, survivors, repeat interruption,
fixed-only continuation, shared credit, reset ownership, and snapshot stability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple

import numpy as np

from env.spaces import SpaceSpec
from env.types import GlobalState, StepFacts
from tasks.specification import TaskSpec, EpisodeMetadata, EpisodeLimits


@dataclass(frozen=True)
class TaskSnapshot:
    """State after reset or a decision; observations are for active policy agents.

    ``active_physical_agents`` reflects RaceEnv.agents, whereas global lifecycle
    facts can still describe physical cars whose decision episodes have ended.
    Counters count completed joint decisions and physics steps in this episode.
    ``raw_observations`` and ``infos`` describe all physical agents, including
    retired cars. This is deliberately separate from adapter-visible observations.
    """

    agents: Tuple[str, ...]
    active_physical_agents: Tuple[str, ...]
    observations: Mapping[str, np.ndarray]
    raw_observations: Mapping[str, Mapping[str, Any]]
    infos: Mapping[str, Mapping[str, Any]]
    global_state: GlobalState
    decision_steps: int
    physics_steps: int
    episode_done: bool


@dataclass(frozen=True)
class AgentDecision:
    """One real policy action and its outcome, including terminal next observation.

    ``info`` includes final physical facts and any task boundary annotation.
    ``action_physical`` is the composer output supplied for this decision;
    TaskSubstep.physical_actions records commands supplied on each physics step.
    No log probability, pre-tanh sample, or value estimate belongs in this record:
    those are learner-owned data associated with action_normalized.
    """

    observation: np.ndarray
    action_normalized: np.ndarray
    action_physical: np.ndarray
    individual_reward: float
    reward_components: Mapping[str, float]
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]


@dataclass(frozen=True)
class TaskSubstep:
    """An executed RaceEnv step with a one-based episode physics-step index.

    ``physical_actions`` captures active cars' task-supplied commands before
    RaceEnv applies its own terminal-car controls. ``facts`` is its post-step
    StepFacts snapshot; masks may differ from the agents that acted on this step.
    """

    physics_step: int
    timestep: float
    physical_actions: Mapping[str, np.ndarray]
    facts: StepFacts


@dataclass(frozen=True)
class TaskStep:
    """One decision's before/after states, actual actor samples, and physics trace.

    Decision keys exactly match before.agents. after.observations only contains
    survivors; retiring agents' final observations are in their AgentDecision.
    Reward fields remain factual; the learner selects its learning objective.
    Substeps are nonempty for every successful step, including fixed-only steps.
    """

    before: TaskSnapshot
    after: TaskSnapshot
    decisions: Mapping[str, AgentDecision]
    team_reward: float
    team_reward_components: Mapping[str, float]
    substeps: Tuple[TaskSubstep, ...]

    @property
    def agent_steps(self) -> int:
        return len(self.decisions)

    @property
    def physics_steps(self) -> int:
        return len(self.substeps)

    @property
    def elapsed_seconds(self) -> float:
        return sum(substep.timestep for substep in self.substeps)


class RaceTaskProtocol(Protocol):
    """Interface for RaceTask and library adapters."""

    @property
    def physical_agents(self) -> Tuple[str, ...]: ...

    @property
    def possible_agents(self) -> Tuple[str, ...]: ...

    @property
    def fixed_policy_agents(self) -> Tuple[str, ...]: ...

    @property
    def agents(self) -> Tuple[str, ...]: ...

    @property
    def episode_done(self) -> bool: ...

    @property
    def timestep(self) -> float: ...

    @property
    def action_repeat(self) -> int: ...

    @property
    def render_mode(self) -> Optional[str]: ...

    @property
    def spec(self) -> TaskSpec: ...

    @property
    def episode_metadata(self) -> EpisodeMetadata: ...

    @property
    def episode_limits(self) -> EpisodeLimits: ...

    def state_space(self) -> SpaceSpec: ...

    def action_space(self, agent: str) -> SpaceSpec:
        """Normalized (2,) float32 space with bounds [-1, 1]."""
        ...

    def observation_space(self, agent: str) -> SpaceSpec:
        """Composed local float32 vector; bounds reflect component clipping."""
        ...

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None,
    ) -> TaskSnapshot: ...

    def step(
        self,
        actions: Mapping[str, np.ndarray],
        *,
        on_physics_step: Optional[Callable[[TaskSubstep], None]] = None,
    ) -> TaskStep:
        """Advance one decision; optional observer runs after each physics step.

        Observers may render or collect metrics, but must not mutate task/core
        state. TaskStep.substeps remains available when no observer is supplied.
        """
        ...

    def render(self) -> Any: ...

    def close(self) -> None: ...
