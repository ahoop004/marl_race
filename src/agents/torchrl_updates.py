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
    names = ("train/policy_loss", "train/value_loss", "train/entropy", "train/approx_kl", "train/clip_fraction")
    if rows:
        means = torch.stack(rows).mean(dim=0)
        cpu = (profile("metrics_device_to_cpu", means.cpu, device=means.device)
               if profile and means.device.type != "cpu" else means.cpu())
        means = profile("tensor_to_python", cpu.tolist) if profile else cpu.tolist()
    else:
        means = [0.0] * len(names)
    return dict(zip(names, means))


@torch.no_grad()
def _approx_kl(agent, batch, key, parameters=None):
    current = batch.clone(False)
    if parameters is None or getattr(agent, "hierarchical", None):
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


def loss_forward(agent, batch, *, group=None, detail=None):
    """The complete differentiable objective, used directly or through DDP."""
    def key(name):
        return (group, name) if group else name
    parameters = None
    def capture(module, inputs, output):
        nonlocal parameters
        parameters = output
    with agent.actor.register_forward_hook(capture):
        with detail("sampled_loss_forward") if detail else nullcontext():
            losses = agent.loss_module(batch)
    if parameters is None:
        raise RuntimeError("The PPO objective did not evaluate the current actor")
    with detail("sampled_entropy") if detail else nullcontext():
        entropy = agent.actor.entropy_from_parameters(*parameters).mean()
    critic_loss = losses.get("loss_critic", torch.zeros((), device=agent.device))
    loss = losses["loss_objective"] + critic_loss - agent.ent_coef * entropy
    with torch.no_grad():
        if agent.target_kl is None:
            with detail("sampled_kl") if detail else nullcontext():
                kl = _approx_kl(agent, batch, key, parameters)
        else:
            kl = torch.zeros((), device=agent.device)
        if agent.vf_coef:
            value_loss = critic_loss / agent.vf_coef
        else:
            values = agent.value_module(batch.clone(False))[key("state_value")]
            value_loss = distance_loss(values, batch[key("value_target")], agent.critic_loss).mean()
    clip_fraction = losses.get('clip_fraction', None)
    if clip_fraction is None:
        with torch.no_grad():
            distribution = agent.probabilistic_actor.get_dist(batch.clone(False))
            ratio = (distribution.log_prob(batch[key('raw_action')]) - batch[key('raw_log_prob')]).exp()
            clip_fraction = ((ratio - 1).abs() > agent.clip_range).float().mean()
    clip_fraction = clip_fraction.detach().mean()
    row = torch.stack((losses["loss_objective"].detach(), value_loss.detach(), entropy.detach(), kl, clip_fraction))
    return loss, row


def optimize_ppo(agent, data, *, group=None):
    def key(name):
        return (group, name) if group else name

    execution = getattr(agent, "distributed", None)
    local_samples = data.numel() if data is not None else 0
    total_samples = int(execution.sum(local_samples)) if execution else local_samples
    if not total_samples:
        return {}
    composition_stats = {}
    if local_samples and getattr(agent.actor, 'kind', None):
        with torch.no_grad():
            actions = data[key('raw_action')]
            old_log_prob = data[key('raw_log_prob')]
            if not torch.isfinite(actions).all() or not torch.isfinite(old_log_prob).all():
                raise ValueError('Nonfinite composition action or rollout likelihood')
            parameters = agent.actor(data[key('observation')][:min(local_samples, 1024)])
            coefficient = agent.actor.coefficients(actions)
            composition_stats = {'train/distribution_entropy': float(agent.actor.entropy_from_parameters(*parameters).mean()),
                'train/composition/log_probability_mean': float(old_log_prob.mean()),
                'train/composition/log_probability_std': float(old_log_prob.std(correction=0)),
                'train/composition/coefficient_mean': float(coefficient.mean()),
                'train/composition/coefficient_std': float(coefficient.std(correction=0))}
            names = [entry['name'] for entry in agent.hierarchical['bank']['skills']]
            if agent.actor.kind in ('positive_gaussian', 'signed_gaussian'):
                names = names[1:]
            for i, name in enumerate(names):
                composition_stats[f'train/composition/mean/{name}'] = float(coefficient[..., i].mean())
                composition_stats[f'train/composition/std/{name}'] = float(coefficient[..., i].std(correction=0))
            if agent.actor.kind == 'dirichlet':
                composition_stats['train/composition/concentration_mean'] = float(parameters[0].mean())
                for i, name in enumerate(names):
                    composition_stats[f'train/composition/concentration/{name}'] = float(parameters[0][..., i].mean())
            else:
                composition_stats['train/composition/latent_mean'] = float(parameters[0].mean())
                composition_stats['train/composition/latent_std'] = float(parameters[1].mean())
                for i in range(agent.actor.action_dim):
                    composition_stats[f'train/composition/latent_mean/{i}'] = float(parameters[0][..., i].mean())
                    composition_stats[f'train/composition/latent_std/{i}'] = float(parameters[1][..., i].mean())
    region = getattr(agent, "profile_region", None)
    count = getattr(agent, "profile_count", None)
    with region("advantage_normalization") if region else nullcontext():
        advantage = data[key("advantage")] if local_samples else None
        normalized = (execution.normalize_advantage(advantage, total_samples) if execution else
                      (advantage - advantage.mean()) / (advantage.std(correction=0) + 1e-8))
        if local_samples:
            data[key("advantage")] = normalized
    with region("replay_construction") if region else nullcontext():
        replay = (TensorDictReplayBuffer(
            storage=LazyTensorStorage(local_samples, device=agent.device),
            sampler=SamplerWithoutReplacement(drop_last=False),
            batch_size=None if execution else agent.batch_size,
        ) if local_samples else None)
    with region("replay_population") if region else nullcontext():
        if replay is not None:
            replay.extend(data)
    rows, early_stop = [], False
    last_kl = torch.zeros((), device=agent.device)
    target_kl = getattr(agent, "target_kl", None)
    samples = (profiling_sample_indices(agent.n_epochs * math.ceil(total_samples / agent.batch_size),
                                        getattr(agent, "profiling_detail_samples", 0) or 0)
               if region else ())
    minibatches = replay_samples = actor_forwards = 0
    with region("optimizer_loop") if region else nullcontext():
        for _ in range(agent.n_epochs):
            quotas = (execution.minibatches(local_samples, agent.batch_size) if execution else
                      ((None, None) for _ in range(math.ceil(local_samples / agent.batch_size))))
            for quota, global_size in quotas:
                detail = region if minibatches in samples else None
                def sample():
                    return replay.sample(batch_size=quota) if execution else replay.sample()
                if execution and not quota:
                    batch = None
                elif detail:
                    with detail("sampled_minibatch_sampling"):
                        batch = sample()
                else:
                    batch = sample()
                minibatches += 1
                size = batch.numel() if batch is not None else 0
                replay_samples += size
                if target_kl is not None:
                    actor_forwards += bool(size)
                    with detail("sampled_kl") if detail else nullcontext():
                        kl = (_approx_kl(agent, batch, key) if size else torch.zeros((), device=agent.device))
                        if execution:
                            kl = execution.sum(kl * size) / global_size
                        last_kl = kl
                        measured_kl = float(kl)
                        if not math.isfinite(measured_kl) or measured_kl > target_kl:
                            early_stop = True
                            break
                if execution:
                    execution.begin_backward(bool(size))
                loss, row = (agent.distributed_loss(batch, group, detail) if execution else
                             loss_forward(agent, batch, group=group, detail=detail))
                actor_forwards += bool(size)
                if target_kl is not None:
                    row[3] = last_kl
                if execution:
                    row = execution.sum(row * size) / global_size
                    # DDP averages rank gradients: undo that averaging before
                    # applying the exact sample-weighted global mean.
                    loss = loss * (execution.world_size * size / global_size)
                last_kl = row[3]
                agent.optimizer.zero_grad()
                with detail("sampled_backward") if detail else nullcontext():
                    loss.backward()
                    if execution:
                        execution.finish_backward()
                with detail("sampled_gradient_clipping") if detail else nullcontext():
                    torch.nn.utils.clip_grad_norm_(agent._optim_parameters, agent.max_grad_norm)
                with detail("sampled_optimizer_step") if detail else nullcontext():
                    agent.optimizer.step()
                rows.append(row)
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
        saturation = ((data[key("action")].abs() >= 1 - 1e-6).float() if local_samples else None)
        if execution:
            saturated, actions = execution.sum([
                float(saturation.sum()) if saturation is not None else 0.,
                saturation.numel() if saturation is not None else 0.]).tolist()
            saturation_mean = saturated / max(actions, 1)
        else:
            saturation_mean = float(saturation.mean())
        metrics.update({
            "train/learning_rate": agent.optimizer.param_groups[0]["lr"],
            "train/rollout_steps": local_samples, "train/optimizer_steps": len(rows),
            "train/kl_early_stop": float(early_stop), "train/last_checked_kl": float(last_kl),
            "train/action_saturation": saturation_mean,
            "train/action_std": float(torch.cat(log_stds).clamp(agent.actor.LOG_STD_MIN, agent.actor.LOG_STD_MAX).exp().mean()) if log_stds else 0.0,
        })
    agent.step_update_scheduler(len(rows))
    metrics.update(composition_stats)
    if getattr(agent, "hierarchical", None):
        metrics.pop("train/action_std", None)
        metrics.pop("train/action_saturation", None)
        metrics.update({"hierarchical/" + key.removeprefix("train/"): value
                        for key, value in list(metrics.items())
                        if key in {"train/policy_loss", "train/value_loss", "train/entropy", "train/approx_kl"}})
    return metrics
