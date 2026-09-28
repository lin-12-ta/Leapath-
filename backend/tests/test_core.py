import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.db import Base, get_db
from app.main import app
from app import rag
from app.rag import retrieve
from app.graph import intake

@pytest.fixture
def client():
    engine=create_engine("sqlite://",connect_args={"check_same_thread":False},poolclass=StaticPool)
    TestingSession=sessionmaker(bind=engine,autoflush=False,expire_on_commit=False)
    Base.metadata.create_all(engine)
    def override():
        db=TestingSession()
        try: yield db
        finally: db.close()
    app.dependency_overrides[get_db]=override
    with TestClient(app) as c: yield c
    app.dependency_overrides.clear(); Base.metadata.drop_all(engine)

def test_retrieval_returns_similarity_ranked_grounding(monkeypatch):
    # Keep the suite deterministic and offline even when an OpenRouter key is in .env.
    monkeypatch.setattr(rag.settings, "openrouter_api_key", "")
    got=retrieve("I enjoy analyzing datasets, SQL, dashboards, and evidence",3)
    assert len(got)==3
    assert got[0]["title"]=="Data Analyst"
    assert got[0]["score"]>got[-1]["score"]

def test_intake_routes_dynamically_only_after_enough_signals():
    one=intake({"message":"I enjoy design and psychology", "profile":{},"history":[]})
    assert one["route"]=="intake"
    two=intake({"message":"I am analytical and like working collaboratively", "profile":one["profile"],"history":[]})
    assert two["route"]=="match"
    assert two["stage"]=="matching"

def test_signup_login_and_protected_session_flow(client):
    created=client.post("/auth/signup",json={"email":"learner@example.com","password":"curious-careers-123"})
    assert created.status_code==201
    access=created.json()["access_token"]
    assert client.get("/sessions").status_code==401
    headers={"Authorization":f"Bearer {access}"}
    made=client.post("/sessions",headers=headers,json={})
    assert made.status_code==201
    assert client.get("/sessions",headers=headers).json()[0]["id"]==made.json()["id"]
    logged=client.post("/auth/login",json={"email":"learner@example.com","password":"curious-careers-123"})
    assert logged.status_code==200
    assert client.post("/auth/login",json={"email":"learner@example.com","password":"wrong-password"}).status_code==401

def test_refresh_rotation_invalidates_previous_token(client):
    tokens=client.post("/auth/signup",json={"email":"refresh@example.com","password":"curious-careers-123"}).json()
    rotated=client.post("/auth/refresh",json={"refresh_token":tokens["refresh_token"]})
    assert rotated.status_code==200
    assert rotated.json()["refresh_token"]!=tokens["refresh_token"]
    assert client.post("/auth/refresh",json={"refresh_token":tokens["refresh_token"]}).status_code==401
