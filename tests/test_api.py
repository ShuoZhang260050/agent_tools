from fastapi.testclient import TestClient
from agent.api import app


def test_health():
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_tools():
    with TestClient(app) as c:
        names = [t["name"] for t in c.get("/tools").json()]
        assert "calculator" in names and "web_search" in names


def test_chat_returns_200(monkeypatch):
    class FakeGraph:
        async def astream_events(self, inp, config, version="v2"):
            return
            yield  # async generator yielding no events

    monkeypatch.setattr("agent.api.get_graph", lambda: FakeGraph())
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "hi", "thread_id": "t1"})
        assert r.status_code == 200
