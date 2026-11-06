"""Select CI targets and checks affected by a push or pull request."""

import json
import os
import re
import subprocess
from pathlib import Path

CHECKS = {"check_backend", "check_ops", "check_nn", "check_functions", "check_amp"}


def gh_api(path):
    return json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{os.environ['GITHUB_REPOSITORY']}/{path}"], text=True
        )
    )


def select_checks(paths):
    all_abis = paths is None
    all_gpu_abis = all_abis
    build = all_abis
    selected = set()
    checks = CHECKS.copy() if all_abis else set()
    for name in paths or ():
        if name.startswith("docs/") or name == ".gitignore":
            continue
        build = True
        profile = re.fullmatch(r"native/profiles/(cp3[0-9]+)\.toml", name)
        if name in {"README.md", "LICENSE", "NOTICE", "tools/ci/cpu.ps1"}:
            all_abis = True
        elif profile:
            selected.add(profile[1])
            checks.update(CHECKS)
        elif name in {
            "tests/check_backend.py",
            "native/resample.cpp",
        } or name.startswith("tests/native/"):
            all_abis = all_gpu_abis = True
            checks.add("check_backend")
        elif name in {"tests/check_functions.py", "src/torch_directml/functions.py"}:
            all_abis = all_gpu_abis = True
            checks.add("check_functions")
        elif name == "tests/check_ops.py":
            all_abis = all_gpu_abis = True
            checks.add("check_ops")
        elif name == "tests/check_nn.py":
            all_abis = all_gpu_abis = True
            checks.update({"check_nn", "check_amp"})
        elif name == "tests/check_amp.py":
            all_abis = all_gpu_abis = True
            checks.add("check_amp")
        else:
            # Shared ABI, dependencies and unknown paths can affect either test suite.
            all_abis = all_gpu_abis = True
            checks.update(CHECKS)
    # An empty ABI filter means all targets from the checked-out revision's profiles.
    return {
        "build": build,
        "abis": [] if all_abis else sorted(selected),
        "gpu": bool(checks),
        "gpu_abis": [] if all_gpu_abis else sorted(selected),
        "checks": sorted(checks),
    }


def changed_paths(event, revision):
    if os.environ["GITHUB_EVENT_NAME"] == "workflow_dispatch":
        return None
    base = (
        event["pull_request"]["base"]["sha"]
        if "pull_request" in event
        else event["before"]
    )
    if base == "0" * 40 or event.get("forced"):
        return None
    if "pull_request" not in event:
        # Pending runs can be replaced; include changes not covered by a successful push CI.
        runs = gh_api(
            "actions/workflows/ci.yml/runs?branch=main&event=push&status=success&per_page=1"
        )["workflow_runs"]
        if not runs:
            return None
        base = runs[0]["head_sha"]
    comparison = gh_api(f"compare/{base}...{revision}")
    if "pull_request" not in event and comparison["status"] not in {
        "ahead",
        "identical",
    }:
        return None
    # The compare API returns at most 300 files; an incomplete diff must not narrow CI.
    if len(comparison["files"]) >= 300:
        return None
    # Adding, removing or renaming a profile changes the supported target matrix.
    if any(
        entry["status"] != "modified"
        and (
            entry["filename"].startswith("native/profiles/")
            or entry.get("previous_filename", "").startswith("native/profiles/")
        )
        for entry in comparison["files"]
    ):
        return None
    return {
        name
        for entry in comparison["files"]
        for name in (
            entry["filename"],
            entry.get("previous_filename", entry["filename"]),
        )
    }


def main():
    event = json.loads(
        Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8")
    )
    plan = select_checks(changed_paths(event, os.environ["REVISION"]))
    print(json.dumps(plan, indent=2))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(f"plan={json.dumps(plan)}\n")


if __name__ == "__main__":
    main()
