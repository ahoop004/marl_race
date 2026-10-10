from __future__ import annotations

import math

import torch
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer

from agents.common import mean_update_metrics


def optimize_ppo(agent, data, *, group=None):
    def key(name):
        return (group, name) if group else name

    advantage = data[key("advantage")]
    data[key("advantage")] = (advantage - advantage.mean()) / (advantage.std(correction=0) + 1e-8)
    replay = TensorDictReplayBuffer(
        storage=LazyTensorStorage(data.numel(), device=agent.device),
        sampler=SamplerWithoutReplacement(drop_last=False), batch_size=agent.batch_size,
    )
    replay.extend(data)
    rows, early_stop, measured_kl = [], False, 0.0
    target_kl = getattr(agent, "target_kl", None)
    for _ in range(agent.n_epochs):
        for _ in range(math.ceil(data.numel() / agent.batch_size)):
            batch = replay.sample()
            with torch.no_grad():
                dist = agent.probabilistic_actor.get_dist(batch.clone(False))
                log_ratio = dist.log_prob(batch[key("raw_action")]) - batch[key("raw_log_prob")]
                kl = (torch.expm1(log_ratio) - log_ratio).mean()
                measured_kl = float(kl)
            if target_kl is not None and (
                not math.isfinite(measured_kl) or measured_kl > target_kl
            ):
                early_stop = True
                break
            losses = agent.loss_module(batch)
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent._optim_parameters, agent.max_grad_norm)
            agent.optimizer.step()
            if agent.vf_coef:
                value_loss = critic_loss / agent.vf_coef
            else:
                with torch.no_grad():
                    values = agent.value_module(batch.clone(False))[key("state_value")]
                    value_loss = torch.nn.functional.mse_loss(values, batch[key("value_target")])
            rows.append(torch.stack((losses["loss_objective"].detach(), value_loss.detach(),
                                     entropy.detach(), kl)))
        if early_stop:
            break
    metrics = mean_update_metrics(rows)
    log_stds = [p.detach().reshape(-1) for name, p in agent.actor.named_parameters()
                if name == "log_std" or name.endswith(".log_std") or name.startswith("log_stds.")]
    metrics.update({
        "train/learning_rate": agent.optimizer.param_groups[0]["lr"],
        "train/rollout_steps": data.numel(), "train/optimizer_steps": len(rows),
        "train/kl_early_stop": float(early_stop), "train/last_checked_kl": measured_kl,
        "train/action_saturation": float((data[key("action")].abs() >= 1 - 1e-6).float().mean()),
        "train/action_std": float(torch.cat(log_stds).clamp(agent.actor.LOG_STD_MIN, agent.actor.LOG_STD_MAX).exp().mean()),
    })
    return metrics
