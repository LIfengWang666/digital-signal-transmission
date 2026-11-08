"""Fingerprint scientific artifacts independently of export metadata."""

from __future__ import annotations
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from reproduction.product_comparison import svg_digest


def scientific_digest(path: Path, mode: str) -> str:
    """Return a content hash under the recorded artifact comparison mode.

    Args:
        path: Existing generated scientific artifact.
        mode: Explicit baseline comparison policy.

    Returns:
        SHA-256 of numerical/text content or rendered image pixels. SVG
        metadata and opaque identifiers are excluded; geometry is retained.
    """
    h = hashlib.sha256()
    if mode == "CSV rows and fields":
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                h.update(json.dumps(row, separators=(",", ":")).encode())
                h.update(b"\n")
    elif mode == "RGBA pixels":
        with Image.open(path) as original:
            image = original.convert("RGBA")
            h.update(str(image.size).encode())
            h.update(image.tobytes())
    elif mode == "SVG geometry and visible text":
        return svg_digest(path)
    elif mode == "UTF8 text LF":
        h.update(path.read_text(encoding="utf-8").encode())
    elif mode == "JSON values except source fingerprints":

        def content(value):
            if isinstance(value, dict):
                return {
                    k: content(v)
                    for k, v in value.items()
                    if not k.endswith(("sha256", "hash"))
                }
            if isinstance(value, list):
                return [content(v) for v in value]
            return value

        h.update(
            json.dumps(
                content(json.loads(path.read_text())),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    else:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()
