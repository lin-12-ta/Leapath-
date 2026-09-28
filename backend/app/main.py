import hashlib, json, re, secrets, time
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from starlette.requests import Request
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from .config import settings
from .db import Base, engine, get_db
from .models import User, CareerSession, Turn, Recommendation, PasswordReset, RefreshToken
from .auth import hash_password, verify_password, issue_access, issue_refresh, current_user, rotate_refresh
from .graph import career_graph


Base.metadata.create_all(bind=engine)
app = FastAPI(title="Career Navigator API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=[settings.frontend_origin], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def trace_http_requests(request: Request, call_next):
    request_id = uuid4().hex
    started = time.perf_counter()
    route = request.scope.get("route")
    route_name = getattr(route, "path", "unmatched")
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        duration_ms = (time.perf_counter() - started) * 1000
        print(f"[API] {request.method} {route_name} -> {response.status_code} ({duration_ms:.0f} ms) request_id={request_id}", flush=True)
        return response
    except Exception as exc:
        print(f"[API] {request.method} {route_name} failed error_type={type(exc).__name__} request_id={request_id}", flush=True)
        raise

class Credentials(BaseModel): email: EmailStr; password: str = Field(min_length=10, max_length=128)
class RefreshBody(BaseModel): refresh_token: str
class ResetRequest(BaseModel): email: EmailStr
class ResetBody(BaseModel): token: str; new_password: str = Field(min_length=10, max_length=128)
class ChatBody(BaseModel): message: str = Field(min_length=1, max_length=5000)
class SessionCreate(BaseModel): title: str = "Career exploration"

def token_pair(db: Session, uid: int): return {"access_token": issue_access(uid), "refresh_token": issue_refresh(db, uid), "token_type":"bearer"}
def owned_session(db: Session, sid: str, user: User):
    row = db.query(CareerSession).filter_by(id=sid, user_id=user.id).first()
    if not row: raise HTTPException(404, "Session not found")
    return row

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/auth/signup", status_code=201)
def signup(body: Credentials, db: Session = Depends(get_db)):
    email = body.email.lower()
    if db.query(User).filter_by(email=email).first():
        print("[AUTH] signup rejected: duplicate email", flush=True)
        raise HTTPException(409, "Email already registered")
    user = User(email=email, password_hash=hash_password(body.password)); db.add(user); db.commit(); db.refresh(user)
    print(f"[AUTH] signup succeeded user_id={user.id}", flush=True)
    return {"user":{"id":user.id,"email":user.email}, **token_pair(db,user.id)}

@app.post("/auth/login")
def login(body: Credentials, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=body.email.lower()).first()
    if not user or not verify_password(body.password, user.password_hash):
        print("[AUTH] login rejected", flush=True)
        raise HTTPException(401, "Incorrect email or password")
    print(f"[AUTH] login succeeded user_id={user.id}", flush=True)
    return {"user":{"id":user.id,"email":user.email}, **token_pair(db,user.id)}

@app.post("/auth/refresh")
def refresh(body: RefreshBody, db: Session = Depends(get_db)):
    access, new_refresh, uid = rotate_refresh(body.refresh_token, db)
    print(f"[AUTH] refresh succeeded user_id={uid}", flush=True)
    return {"access_token":access,"refresh_token":new_refresh,"token_type":"bearer","user_id":uid}

@app.post("/auth/logout")
def logout(body: RefreshBody, db: Session = Depends(get_db), user: User = Depends(current_user)):
    from jose import jwt, JWTError
    try:
        p = jwt.decode(body.refresh_token, settings.jwt_secret, algorithms=["HS256"])
        if int(p.get("sub", -1)) != user.id: raise ValueError()
        row = db.query(RefreshToken).filter_by(jti=p.get("jti"), user_id=user.id, revoked=False).first()
        if row: row.revoked=True; db.commit()
    except Exception: pass
    print(f"[AUTH] logout completed user_id={user.id}", flush=True)
    return {"ok":True}

@app.post("/auth/forgot-password")
def forgot(body: ResetRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=body.email.lower()).first()
    if user:
        raw = secrets.token_urlsafe(32); digest = hashlib.sha256(raw.encode()).hexdigest()
        db.add(PasswordReset(user_id=user.id, token_hash=digest, expires_at=datetime.now(timezone.utc)+timedelta(minutes=30))); db.commit()
        # Demo flow: token appears only in backend logs. Replace with a mail provider in production.
        print(f"Password reset token for {user.email}: {raw}")
    print(f"[AUTH] password reset requested account_exists={bool(user)}", flush=True)
    return {"message":"If the account exists, reset instructions have been issued."}

@app.post("/auth/reset-password")
def reset_password(body: ResetBody, db: Session = Depends(get_db)):
    digest = hashlib.sha256(body.token.encode()).hexdigest(); row = db.query(PasswordReset).filter_by(token_hash=digest, used=False).first()
    if not row or row.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc): raise HTTPException(400,"Invalid or expired reset token")
    user = db.get(User,row.user_id); user.password_hash=hash_password(body.new_password); row.used=True
    for token in db.query(RefreshToken).filter_by(user_id=user.id,revoked=False): token.revoked=True
    db.commit(); print(f"[AUTH] password reset completed user_id={user.id}", flush=True); return {"ok":True}

@app.get("/auth/me")
def me(user: User = Depends(current_user)): return {"id":user.id,"email":user.email}

@app.post("/sessions", status_code=201)
def create_session(body: SessionCreate, db: Session=Depends(get_db), user: User=Depends(current_user)):
    s=CareerSession(id=str(uuid4()),user_id=user.id,summary=body.title); db.add(s); db.commit(); print(f"[SESSION] created user_id={user.id} session_id={s.id}", flush=True); return {"id":s.id,"stage":s.stage,"title":s.summary}

@app.get("/sessions")
def list_sessions(db: Session=Depends(get_db), user: User=Depends(current_user)):
    rows=db.query(CareerSession).filter_by(user_id=user.id).order_by(CareerSession.updated_at.desc()).all()
    return [{"id":s.id,"stage":s.stage,"title":s.summary,"updated_at":s.updated_at} for s in rows]

@app.get("/sessions/{sid}")
def get_session(sid:str, db: Session=Depends(get_db), user: User=Depends(current_user)):
    s=owned_session(db,sid,user)
    return {"id":s.id,"stage":s.stage,"profile":json.loads(s.profile_json),"turns":[{"role":t.role,"content":t.content,"created_at":t.created_at} for t in s.turns]}

@app.post("/sessions/{sid}/chat")
def chat(sid:str, body:ChatBody, db:Session=Depends(get_db), user:User=Depends(current_user)):
    s=owned_session(db,sid,user)
    stored_history=[{"role":t.role,"content":t.content} for t in s.turns]
    profile=json.loads(s.profile_json)
    history_offset=min(profile.get("history_start_index",0),len(stored_history))
    history=stored_history[history_offset:]
    # Recover the last displayed recommendation set from the persisted assistant
    # turn so the graph can handle post-recommendation follow-up after restarts.
    previous_matches = []
    for turn in reversed(history):
        if turn["role"] != "assistant":
            continue
        titles = re.findall(r"\*\*(.+?)\*\* \(retrieval score", turn["content"])
        if titles:
            previous_matches = [{"title": title} for title in titles]
            break
    graph_input={"message":body.message,"profile":profile,"history":history,"history_offset":history_offset,"stage":s.stage,"matches":previous_matches}
    previous_stage = s.stage
    print(f"[FLOW] turn start session_id={sid} stage={s.stage} history_messages={len(history)} profile_fields={list(graph_input['profile'].keys())}", flush=True)
    try: out=career_graph.invoke(graph_input)
    except Exception as exc:
        print(f"[FLOW] failed user_id={user.id} error_type={type(exc).__name__}", flush=True)
        raise HTTPException(503, "Career service temporarily unavailable. Check the backend console flow output for the request.")
    s.profile_json=json.dumps(out.get("profile",json.loads(s.profile_json))); s.stage=out.get("stage",s.stage); s.summary=out.get("summary",s.summary)
    db.add_all([Turn(session_id=sid,role="user",content=body.message),Turn(session_id=sid,role="assistant",content=out["reply"])])
    for m in out.get("matches",[]):
        db.add(Recommendation(session_id=sid,career_title=m["title"],rationale=m["rationale"],sources_json=json.dumps(out.get("retrieved",[]))))
    db.commit()
    retrieved_count = len(out.get("retrieved", []))
    recommendation_count = len(out.get("matches", []))
    print(f"[FLOW] turn saved session_id={sid} stage={previous_stage}->{s.stage} retrieved={retrieved_count} recommendations={recommendation_count}", flush=True)
    return {"reply":out["reply"],"stage":s.stage,"profile":json.loads(s.profile_json),"matches":out.get("matches",[]),"retrieved":out.get("retrieved",[]),"critique":out.get("critique",{})}
