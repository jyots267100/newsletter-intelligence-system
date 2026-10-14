from app.core.exceptions import DocumentNotFoundError, LLMTimeoutError
from app.schemas.chat import ChatResponse


class _FakeChatService:
    def __init__(self):
        self.raise_error = None
        self.last_call = None

    def chat(self, document_id, message, history):
        self.last_call = (document_id, message, history)
        if self.raise_error:
            raise self.raise_error
        return ChatResponse(
            answer="This is the answer.", citations=[], document_id=document_id, grounded=True
        )


def test_chat_success(client, override_chat_service):
    override_chat_service(_FakeChatService())
    response = client.post(
        "/api/v1/chat", json={"document_id": "doc-1", "message": "What happened?", "history": []}
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "This is the answer."


def test_chat_document_not_found_returns_404(client, override_chat_service):
    fake = _FakeChatService()
    fake.raise_error = DocumentNotFoundError("missing")
    override_chat_service(fake)

    response = client.post(
        "/api/v1/chat", json={"document_id": "missing-id", "message": "Hi", "history": []}
    )
    assert response.status_code == 404


def test_chat_llm_timeout_returns_504(client, override_chat_service):
    fake = _FakeChatService()
    fake.raise_error = LLMTimeoutError("timed out")
    override_chat_service(fake)

    response = client.post(
        "/api/v1/chat", json={"document_id": "doc-1", "message": "Hi", "history": []}
    )
    assert response.status_code == 504


def test_chat_rejects_empty_message(client, override_chat_service):
    override_chat_service(_FakeChatService())
    response = client.post(
        "/api/v1/chat", json={"document_id": "doc-1", "message": "", "history": []}
    )
    assert response.status_code == 422


def test_chat_passes_history_through(client, override_chat_service):
    fake = _FakeChatService()
    override_chat_service(fake)
    history = [{"role": "user", "content": "earlier question"}]

    client.post(
        "/api/v1/chat", json={"document_id": "doc-1", "message": "follow up", "history": history}
    )
    document_id, message, passed_history = fake.last_call
    assert document_id == "doc-1"
    assert message == "follow up"
    assert passed_history[0].content == "earlier question"
