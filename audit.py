from __future__ import annotations

import hashlib
import json
import subprocess


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def fingerprint(jd: str, candidates: dict, config: dict, prompts: dict, model: str) -> str:
    payload = {
        "git_commit": git_commit(), "model": model,
        "jd_sha256": hashlib.sha256(jd.encode()).hexdigest(),
        "candidate_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sorted(candidates.items())},
        "config": config, "prompts": prompts,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
