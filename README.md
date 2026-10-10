# MARL racing

Reinforcement learning experiments for racing.

# 1. Solo PPO pretraining
python run.py --scenario scenarios/ppo_lap_completion_pretrain.yaml \
  --output-dir outputs/pretrain

# 2. PPO track transfer: circle
python run.py --scenario scenarios/ppo_lap_completion_transfer.yaml \
  --checkpoint outputs/pretrain/best.pt \
  --output-dir outputs/transfer_circle

# 3. PPO track transfer: Budapest, three laps
python run.py --scenario scenarios/ppo_lap_completion_transfer_3lap.yaml \
  --checkpoint outputs/pretrain/best.pt \
  --output-dir outputs/transfer_budapest

# 4. MAPPO from scratch
python run.py --scenario scenarios/mappo_2v2_completion_scratch.yaml \
  --output-dir outputs/mappo_scratch

# 5. MAPPO: independent pretrained actors
python run.py --scenario scenarios/mappo_2v2_completion_pretrained.yaml \
  --pretrained-actor outputs/pretrain/best.pt \
  --output-dir outputs/mappo_pretrained

# 6. MAPPO: shared pretrained actor
python run.py --scenario scenarios/mappo_2v2_completion_pretrained_shared.yaml \
  --pretrained-actor outputs/pretrain/best.pt \
  --output-dir outputs/mappo_shared

# 7. MAPPO: per-agent LoRA
python run.py --scenario scenarios/mappo_2v2_completion_lora.yaml \
  --pretrained-actor outputs/pretrain/best.pt \
  --output-dir outputs/mappo_lora