"""Prepare source-bound execution records before a requested full regeneration."""

from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """Return the SHA-256 of an existing input or source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def begin_execution() -> None:
    """Discard prior generated execution freezes and disposable caches.

    This is called only when the user requests numerical regeneration.
    Scientific comparison targets and declared experiment settings remain fixed.
    """
    for path in ROOT.rglob("*.json"):
        relative = path.relative_to(ROOT)
        if "results" in relative.parts and (
            "freeze" in path.name or path.name == "pre_phase1_gate.json"
        ):
            path.unlink()
    for path in list(ROOT.rglob(".work")):
        resolved = path.resolve()
        if resolved.is_dir() and resolved.is_relative_to(ROOT):
            shutil.rmtree(resolved)


def bind_metric_inputs() -> None:
    """Bind regenerated upstream records before any new complete-system draw."""
    path = ROOT / "stage4/r41/design.json"
    design = json.loads(path.read_text())
    for name in design["input_sha256"]:
        design["input_sha256"][name] = digest(ROOT / name)
    path.write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    path = ROOT / "stage4/r41/metric_definition_freeze.json"
    frozen = json.loads(path.read_text())
    for name in frozen["source_sha256"]:
        frozen["source_sha256"][name] = digest(ROOT / name)
    path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")


def bind_pilot_inputs() -> None:
    """Record current upstream artifact fingerprints before pilot generation."""
    path = ROOT / "stage4/r42/design.json"
    design = json.loads(path.read_text())
    for name in design["fixed_inputs_sha256"]:
        design["fixed_inputs_sha256"][name] = digest(ROOT / name)
    path.write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
