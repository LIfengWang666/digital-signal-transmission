"""Compare full SVG structure modulo export dates and opaque object IDs."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def svg_digest(path: Path) -> str:
    """Hash complete vector contents modulo timestamps and consistent ID names.

    Args:
        path: Matplotlib SVG export.

    Returns:
        SHA-256 preserving geometry, text, styles, colours and references;
        remove nonvisual metadata and consistently rename opaque object IDs.
    """
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {
                "{http://purl.org/dc/elements/1.1/}date",
                "{http://www.w3.org/2000/svg}metadata",
            }:
                parent.remove(child)
    identities = {
        element.attrib["id"]: f"object_{index}"
        for index, element in enumerate(root.iter())
        if "id" in element.attrib
    }
    reference = re.compile(r"#[A-Za-z0-9_:.-]+")
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            if key == "id":
                element.attrib[key] = identities[value]
            else:
                element.attrib[key] = reference.sub(
                    lambda m: "#" + identities.get(m[0][1:], m[0][1:]), value
                )
    return hashlib.sha256(ET.tostring(root, encoding="utf-8")).hexdigest()
