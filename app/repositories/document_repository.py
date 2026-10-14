"""Persistence for document metadata (registry).

A JSON file is intentionally used instead of a full database: document
counts are small (single-user/portfolio scale), the data is a flat list of
records, and a database would add operational weight without a real
benefit here. This keeps the "modular monolith" promise explicit - the
repository interface is the seam where a real database could be swapped
in later without touching services or routes.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from app.schemas.document import DocumentMetadata


class DocumentRepository:
    def __init__(self, registry_path: str):
        self._path = Path(registry_path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write_all({})

    def _read_all(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        with self._path.open("r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)

    def _write_all(self, data: dict[str, dict]) -> None:
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def save(self, document: DocumentMetadata) -> None:
        with self._lock:
            data = self._read_all()
            data[document.document_id] = json.loads(document.model_dump_json())
            self._write_all(data)

    def get(self, document_id: str) -> DocumentMetadata | None:
        with self._lock:
            data = self._read_all()
        record = data.get(document_id)
        return DocumentMetadata.model_validate(record) if record else None

    def get_by_content_hash(self, content_hash: str) -> DocumentMetadata | None:
        with self._lock:
            data = self._read_all()
        for record in data.values():
            if record.get("content_hash") == content_hash:
                return DocumentMetadata.model_validate(record)
        return None

    def list_all(self) -> list[DocumentMetadata]:
        with self._lock:
            data = self._read_all()
        return [DocumentMetadata.model_validate(r) for r in data.values()]

    def delete(self, document_id: str) -> bool:
        with self._lock:
            data = self._read_all()
            if document_id not in data:
                return False
            del data[document_id]
            self._write_all(data)
            return True
