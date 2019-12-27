import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from app.main import app
    client = TestClient(app())
    return client