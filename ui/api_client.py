"""Thin HTTP client wrapping the FastAPI backend.

Kept separate from streamlit_app.py so the request/response handling can be
reasoned about (and in principle tested) independently of Streamlit widgets.
"""
from __future__ import annotations

import os

import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
_TIMEOUT = 90  # analysis/chat calls involve an LLM round trip


class APIError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _handle(response: requests.Response) -> dict:
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise APIError(response.status_code, detail)
    return response.json()


def upload_document(filename: str, file_bytes: bytes) -> dict:
    files = {"file": (filename, file_bytes, "application/pdf")}
    response = requests.post(f"{API_BASE_URL}/api/v1/documents/upload", files=files, timeout=_TIMEOUT)
    return _handle(response)


def list_documents() -> dict:
    response = requests.get(f"{API_BASE_URL}/api/v1/documents", timeout=_TIMEOUT)
    return _handle(response)


def get_document(document_id: str) -> dict:
    response = requests.get(f"{API_BASE_URL}/api/v1/documents/{document_id}", timeout=_TIMEOUT)
    return _handle(response)


def delete_document(document_id: str) -> dict:
    response = requests.delete(f"{API_BASE_URL}/api/v1/documents/{document_id}", timeout=_TIMEOUT)
    return _handle(response)


def analyze_document(document_id: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/documents/{document_id}/analyze", json={}, timeout=_TIMEOUT
    )
    return _handle(response)


def send_chat_message(document_id: str, message: str, history: list[dict]) -> dict:
    payload = {"document_id": document_id, "message": message, "history": history}
    response = requests.post(f"{API_BASE_URL}/api/v1/chat", json=payload, timeout=_TIMEOUT)
    return _handle(response)


def check_backend_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False
