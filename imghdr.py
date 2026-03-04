"""
Compatibility shim for Python 3.13 where the standard-library ``imghdr`` module
was removed.

Streamlit<=1.19 still imports ``imghdr`` to guess image types. To keep the app
working on Streamlit Cloud (Python 3.13), we provide a very small substitute
that exposes the same public function ``what``.

This implementation is intentionally simple: it mostly relies on the file
extension, which is sufficient for our use case (displaying PNGs and other web
images in the app).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, BinaryIO


def what(
    file: Union[str, bytes, Path, BinaryIO],
    h: Optional[bytes] = None,  # kept for API compatibility, ignored here
) -> Optional[str]:
    """
    Best-effort guess of image type.

    For file paths, we return the lowercase file extension without the leading
    dot (e.g. ``\"png\"``, ``\"jpg\"``). For file-like objects or unknown cases,
    we return ``None``.
    """

    # If given a path-like object, infer from the suffix.
    if isinstance(file, (str, bytes, Path)):
        path = Path(file)
        suffix = path.suffix.lower().lstrip(".")
        return suffix or None

    # For file-like objects we do not attempt header sniffing; Streamlit
    # gracefully handles ``None`` by falling back to other heuristics.
    return None

