"""LOSO training for one label task and one protocol (sharded, resumable):
  python scripts/a06_train.py --task nback_bin --protocol enroll --shard 0 --nshards 8"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fada.common import load_cfg, subjects
from fada.data import Segments, load_task, loso_splits
from fada.trainer import train_run

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--protocol", required=True, choices=["strict", "enroll", "calib"])
ap.add_argument("--models", nargs="*")
ap.add_argument("--seeds", nargs="*", type=int)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--out", default=None)
ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. window.length_s=30")
args = ap.parse_args()

cfg = load_cfg()
for kv in args.set:
    k, v = kv.split("=")
    sec, key = k.split(".")
    cfg[sec][key] = type(cfg[sec][key])(float(v))
models = args.models or cfg["unimodal"] + cfg["fusions"]
seeds = args.seeds if args.seeds is not None else cfg["train"]["seeds"]
out = Path(args.out or Path(cfg["paths"]["results"]) / "runs") / args.task / args.protocol
segs, arrays = load_task(cfg, args.task)
data = Segments(cfg, segs, arrays)
subs = subjects(cfg, args.task)
jobs = [(m, seed, sp) for seed in seeds for sp in loso_splits(subs, cfg["train"]["n_val_subjects"], seed) for m in models]
jobs = jobs[args.shard::args.nshards]
print(f"{len(jobs)} jobs | {args.task} {args.protocol} shard {args.shard}/{args.nshards} -> {out}", flush=True)
for name, seed, sp in jobs:
    info = train_run(cfg, data, args.task, name, args.protocol, seed, sp, out)
    if info:
        print(f"{name:14s} {info['test_subject']} seed={seed} ep={info['best_epoch']:2d}/{info['epochs_run']:2d} "
              f"val_f1={info['val_f1']:.3f} test_f1={info['test_f1']:.3f} ({info['train_time_s']:.0f}s)", flush=True)
