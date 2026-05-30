"""Persistent registry of rule signatures reserved for validation only."""

import json
from pathlib import Path


class HoldoutRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, set[str]] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self._data = {cat: set(sigs) for cat, sigs in raw.items()}

    def is_reserved(self, category: str, signature: str) -> bool:
        return signature in self._data.get(category, set())

    def reserve(self, category: str, signature: str) -> bool:
        """Reserve a signature. Returns True if newly reserved, False if already."""
        if self.is_reserved(category, signature):
            return False
        self._data.setdefault(category, set()).add(signature)
        self._flush()
        return True

    def _flush(self) -> None:
        serializable = {cat: sorted(sigs) for cat, sigs in self._data.items()}
        self.path.write_text(json.dumps(serializable, indent=2))
