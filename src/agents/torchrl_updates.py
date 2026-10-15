from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
from torchrl.objectives.utils import distance_loss

from training.runtime import profiling_sample_indices


# ClipPPOLoss/MAPPOLoss consume precomputed advantages and targets here: their
# fallback value estimator (and its next-state in_keys) is never invoked.
# Keep state_value for the critic contract, and bounded/latent actions for
# entropy, likelihood/KL, and saturation diagnostics. loc/scale are recomputed.
PPO_OPTIMIZATION_KEYS = ("observation", "action", "raw_action", "raw_log_prob",
                         "advantage", "value_target", "state_value")
# TorchRL's loss reduction also reads these when supplied by direct PPO callers.
# The synchronous, unpadded collector does not normally produce either key.
PPO_OPTIONAL_OPTIMIZATION_KEYS = (("collector", "mask"), "shifted_valid")
MAPPO_OPTIMIZATION_KEYS = tuple(("agents", key) for key in PPO_OPTIMIZATION_KEYS) + (
    ("agents", "index"), "state")


def mean_update_metrics(rows, *, profile=None):
    """Transfer averaged optimizer metrics to the CPU once."""
    names = ("train/policy_loss", "train/value_loss", "train/entropy", "train/approx_kl")
    if rows:
        means = torch.stack(rows).mean(dim=0)
        cpu = (profile("metrics_device_to_cpu", means.cpu, device=means.device)
               if profile and means.device.type != "cpu" else means.cpu())
        means = profile("tensor_to_python", cpu.tolist) if profile else cpu.tolist()
    else:
        means = [0.0] * len(names)
    return dict(zip(names, means))


def _sample_minibatch(region, replay):
    with region("sampled_minibatch_sampling"):
        return replay.sample()


@torch.no_grad()
def _approx_kl(agent, batch, key, parameters=None):
    current = batch.clone(False)
    if parameters is None:
        dist = agent.probabilistic_actor.get_dist(current)
    else:
        # Match the PPO/MAPPO parameter modules' expansion and agent shape,
        # using the actor outputs already evaluated by the TorchRL objective.
        mean, std = parameters
        shape = batch[key("raw_action")].shape
        current[key("loc")] = mean.reshape(shape)
        current[key("scale")] = std.expand_as(mean).reshape(shape)
        dist = agent.probabilistic_actor.build_dist_from_params(current)
    log_ratio = dist.log_prob(batch[key("raw_action")]) - batch[key("raw_log_prob")]
    return (torch.expm1(log_ratio) - log_ratio).mean()


def optimize_ppo(agent, data, *, group=None):
    def key(name):
        return (group, name) if group else name

    region = getattr(agent, "profile_region", None)
    count = getattr(agent, "profile_count", None)
    with region("advantage_normalization") if region else nullcontext():
        advantage = data[key("advantage")]
        data[key("advantage")] = (advantage - advantage.mean()) / (advantage.std(correction=0) + 1e-8)
    with region("replay_construction") if region else nullcontext():
        replay = TensorDictReplayBuffer(
            storage=LazyTensorStorage(data.numel(), device=agent.device),
            sampler=SamplerWithoutReplacement(drop_last=False), batch_size=agent.batch_size,
        )
    with region("replay_population") if region else nullcontext():
        replay.extend(data)
    rows, early_stop = [], False
    last_kl = torch.zeros((), device=agent.device)
    target_kl = getattr(agent, "target_kl", None)
    samples = (profiling_sample_indices(agent.n_epochs * math.ceil(data.numel() / agent.batch_size),
                                        getattr(agent, "profiling_detail_samples", 0) or 0)
               if region else ())
    minibatches = replay_samples = actor_forwards = 0
    actor_parameters = None

    def capture_actor(module, inputs, output):
        nonlocal actor_parameters, actor_forwards
        actor_parameters = output
        actor_forwards += 1  # Top-level actor calls, including routed actors.

    # A public PyTorch hook observes the unchanged TorchRL objective's actor
    # call. It exists only during this update, is removed even on exceptions,
    # and keeps no parameter cache or graph across minibatches.
    with (agent.actor.register_forward_hook(capture_actor),
          region("optimizer_loop") if region else nullcontext()):
        for _ in range(agent.n_epochs):
            for _ in range(math.ceil(data.numel() / agent.batch_size)):
                detail = region if minibatches in samples else None
                batch = replay.sample() if detail is None else _sample_minibatch(detail, replay)
                minibatches += 1
                if count:
                    replay_samples += batch.numel()
                if target_kl is not None:
                    with detail("sampled_kl") if detail else nullcontext():
                        kl = _approx_kl(agent, batch, key)
                        last_kl = kl
                        measured_kl = float(kl)
                        if not math.isfinite(measured_kl) or measured_kl > target_kl:
                            early_stop = True
                            break
                actor_parameters = None
                if detail:
                    with detail("sampled_loss_forward"):
                        losses = agent.loss_module(batch)
                else:
                    losses = agent.loss_module(batch)
                if actor_parameters is None:
                    raise RuntimeError("The PPO objective did not evaluate the current actor")
                parameters, actor_parameters = actor_parameters, None
                if target_kl is None:
                    # No gate needs a separate actor precheck. This is still
                    # the pre-step KL, under no_grad, with the original formula.
                    # sampled_kl now times only distribution/ratio arithmetic;
                    # the shared actor forward belongs to sampled_loss_forward.
                    with detail("sampled_kl") if detail else nullcontext():
                        kl = _approx_kl(agent, batch, key, parameters)
                        last_kl = kl
                with detail("sampled_entropy") if detail else nullcontext():
                    # Preserve the actor's original, unexpanded scale so the
                    # quadrature uses the same broadcast and reduction order.
                    entropies = agent.actor.entropy_from_parameters(*parameters)
                    entropy = entropies.mean()
                del parameters
                critic_loss = losses.get("loss_critic", torch.zeros((), device=agent.device))
                loss = losses["loss_objective"] + critic_loss - agent.ent_coef * entropy
                agent.optimizer.zero_grad()
                if detail:
                    with detail("sampled_backward"):
                        loss.backward()
                    with detail("sampled_gradient_clipping"):
                        torch.nn.utils.clip_grad_norm_(agent._optim_parameters, agent.max_grad_norm)
                    with detail("sampled_optimizer_step"):
                        agent.optimizer.step()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent._optim_parameters, agent.max_grad_norm)
                    agent.optimizer.step()
                if agent.vf_coef:
                    value_loss = critic_loss / agent.vf_coef
                else:
                    with torch.no_grad():
                        values = agent.value_module(batch.clone(False))[key("state_value")]
                        value_loss = distance_loss(values, batch[key("value_target")], agent.critic_loss).mean()
                rows.append(torch.stack((losses["loss_objective"].detach(), value_loss.detach(),
                                         entropy.detach(), kl)))
            if early_stop:
                break
    if count:
        count("ppo_minibatches", minibatches)
        count("replay_samples", replay_samples)
        count("optimization_actor_forwards", actor_forwards)
    with region("final_metric_aggregation") if region else nullcontext():
        profile = getattr(agent, "profile_call", None)
        metrics = (profile("metric_reduction", mean_update_metrics, rows, profile=profile, device=agent.device)
                   if profile else mean_update_metrics(rows))
        log_stds = [p.detach().reshape(-1) for name, p in agent.actor.named_parameters()
                    if name == "log_std" or name.endswith(".log_std") or name.startswith("log_stds.")]
        metrics.update({
            "train/learning_rate": agent.optimizer.param_groups[0]["lr"],
            "train/rollout_steps": data.numel(), "train/optimizer_steps": len(rows),
            "train/kl_early_stop": float(early_stop), "train/last_checked_kl": float(last_kl),
            "train/action_saturation": float((data[key("action")].abs() >= 1 - 1e-6).float().mean()),
            "train/action_std": float(torch.cat(log_stds).clamp(agent.actor.LOG_STD_MIN, agent.actor.LOG_STD_MAX).exp().mean()),
        })
    agent.step_update_scheduler(len(rows))
    return metrics
