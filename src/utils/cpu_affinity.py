"""Read Linux CPU placement without changing process or memory affinity."""
import os
from pathlib import Path


def cpu_placement():
    """Report the allowed CPUs, including partial SLURM/socket allocations."""
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    topology = []
    for cpu in cpus or []:
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        def number(name):
            try:
                return int((root / "topology" / name).read_text())
            except (OSError, ValueError):
                return None
        nodes = sorted(int(path.name[4:]) for path in root.glob("node[0-9]*"))
        topology.append(dict(cpu=cpu, socket=number("physical_package_id"),
                             core=number("core_id"), numa_node=nodes[0] if nodes else None))
    sockets = sorted({row["socket"] for row in topology if row["socket"] is not None})
    nodes = sorted({row["numa_node"] for row in topology if row["numa_node"] is not None})
    return dict(pid=os.getpid(), cpu_affinity=cpus,
                cpu_affinity_count=len(cpus) if cpus is not None else os.cpu_count(),
                cpu_sockets=sockets, numa_nodes=nodes, cpu_topology=topology)


def slurm_allocation():
    """Keep allocation/binding evidence; never infer CPU IDs from SLURM counts."""
    names = ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE",
             "SLURM_JOB_CPUS_PER_NODE", "SLURM_CPU_BIND_TYPE", "SLURM_CPU_BIND_LIST",
             "SLURM_MEM_BIND_TYPE", "SLURM_MEM_BIND_LIST")
    return {name: os.environ[name] for name in names if name in os.environ}
