import os
import gzip
import json
import platform
import subprocess

import yaml
import torch
import transformers

ROOT= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_jsonl(path):
    opener= gzip.open if path.endswith(".gz") else open

    with opener(path, "rt", encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    #write to a temp file then rename, so a killed session never leaves a half-written result
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp= path + ".tmp"

    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)

    os.replace(tmp, path)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp= path + ".tmp"

    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    os.replace(tmp, path)


def load_config(path):
    #task config on top of configs/shared.yaml
    with open(os.path.join(ROOT, "configs", "shared.yaml")) as f:
        cfg= yaml.safe_load(f)

    with open(path) as f:
        cfg.update(yaml.safe_load(f))

    return cfg


def default_output_dir():
    return "/kaggle/working/outputs" if os.path.exists("/kaggle") else "outputs"


def data_lock(cfg):
    with open(os.path.join(ROOT, cfg["data_lock_copy"])) as f:
        return json.load(f)


def model_revision(cfg, model_key):
    lock= data_lock(cfg)

    return lock["revisions"][cfg["model_revision_keys"][model_key]]


def git_commit():
    try:
        commit= subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty= subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

    return commit + ("-dirty" if dirty else "")


def sdpa_flags():
    #which sdpa backends are enabled; the dispatched kernel itself is measured in task 2a
    if not torch.cuda.is_available():
        return {"device": "cpu", "backend": "math"}

    return {
        "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_enabled": torch.backends.cuda.math_sdp_enabled(),
        "compute_capability": ".".join(str(x) for x in torch.cuda.get_device_capability())
    }


def run_info(cfg, device, dtype, model_key=None):
    info= {
        "git_commit": git_commit(),
        "seed": cfg["seed"],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu",
        "dtype": str(dtype).replace("torch.", ""),
        "attn_implementation": cfg["attn_implementation"],
        "sdpa": sdpa_flags(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda
    }

    if model_key is not None:
        info["model"]= cfg["models"][model_key]
        info["model_revision"]= model_revision(cfg, model_key)

    return info
