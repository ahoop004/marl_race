from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
from torchrl.objectives.utils import distance_loss

from training.runtime import profiling_sample_indices


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
    minibatches = replay_samples = 0
    with region("optimizer_loop") if region else nullcontext():
        for _ in range(agent.n_epochs):
            for _ in range(math.ceil(data.numel() / agent.batch_size)):
                detail = region if minibatches in samples else None
                batch = replay.sample() if detail is None else _sample_minibatch(detail, replay)
                minibatches += 1
                if count:
                    replay_samples += batch.numel()
                with detail("sampled_kl") if detail else nullcontext():
                    with torch.no_grad():
                        dist = agent.probabilistic_actor.get_dist(batch.clone(False))
                        log_ratio = dist.log_prob(batch[key("raw_action")]) - batch[key("raw_log_prob")]
                        kl = (torch.expm1(log_ratio) - log_ratio).mean()
                        last_kl = kl
                    if target_kl is not None:
                        measured_kl = float(kl)
                        if not math.isfinite(measured_kl) or measured_kl > target_kl:
                            early_stop = True
                            break
                if detail:
                    with detail("sampled_loss_forward"):
                        losses = agent.loss_module(batch)
                else:
                    losses = agent.loss_module(batch)
                with detail("sampled_entropy") if detail else nullcontext():
                    actor_kwargs = {}
                    if getattr(agent, "routed_actor", False):
                        actor_kwargs["adapter_indices"] = batch[key("index")].reshape(-1)
                    _, entropies = agent.actor.evaluate_actions(
                        batch[key("observation")].reshape(-1, agent.obs_dim),
                        batch[key("action")].reshape(-1, agent.action_dim),
                        batch[key("raw_action")].reshape(-1, agent.action_dim), **actor_kwargs,
                    )
                    entropy = entropies.mean()
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
