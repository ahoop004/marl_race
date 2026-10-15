"""Run with torchrun to check the allocated single-node NCCL device mapping."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from training.distributed import distributed_execution


def main():
    with distributed_execution(True) as execution:
        properties = torch.cuda.get_device_properties(execution.device)
        record = {"rank": execution.rank, "local_rank": execution.local_rank,
                  "device": str(execution.device), "name": properties.name, "uuid": str(properties.uuid)}
        devices = execution.gather(record)
        if len({device["uuid"] for device in devices}) != execution.world_size:
            raise RuntimeError("Learner ranks must use distinct physical GPUs")
        total = int(execution.sum(execution.rank + 1))
        if total != execution.world_size * (execution.world_size + 1) // 2:
            raise RuntimeError("NCCL all-reduce returned an incorrect result")
        if execution.primary:
            print(json.dumps({"backend": "nccl", "world_size": execution.world_size,
                              "devices": devices, "all_reduce_sum": total}), flush=True)


if __name__ == "__main__":
    main()
