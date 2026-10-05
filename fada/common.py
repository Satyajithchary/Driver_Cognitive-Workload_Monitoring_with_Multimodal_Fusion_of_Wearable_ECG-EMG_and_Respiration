import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path=None):
    path = Path(path) if path else ROOT / "configs" / "adabase.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = str((ROOT / v).resolve())
    excl = Path(cfg["paths"]["data"]) / "exclusions.yaml"
    cfg["exclude"] = yaml.safe_load(excl.read_text()) if excl.exists() else {}
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seg_key(study, phase, level):
    return f"{study}|{phase}|{int(level)}"


def subjects(cfg, task=None):
    """All subject ids with extracted data, minus exclusions (global and per task)."""
    d = Path(cfg["paths"]["data"]) / "raw_segments"
    subs = sorted(p.stem for p in d.glob("*.npz"))
    ex = set(cfg["exclude"].get("all", []))
    if task:
        ex |= set(cfg["exclude"].get(task, []))
    return [s for s in subs if s not in ex]
