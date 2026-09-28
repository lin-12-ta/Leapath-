from datetime import datetime, timedelta, timezone
from uuid import uuid4
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from .config import settings
from .db import get_db
from .models import User, RefreshToken

pwd = CryptContext(schemes=["bcrypt_sha256"], deprecated="auto")
bearer = HTTPBearer(auto_error=False)
ALG = "HS256"

def hash_password(raw: str) -> str: return pwd.hash(raw)
def verify_password(raw: str, hashed: str) -> bool: return pwd.verify(raw, hashed)
def issue_access(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode({"sub": str(user_id), "type": "access", "iat": now, "exp": now + timedelta(minutes=settings.access_token_minutes)}, settings.jwt_secret, algorithm=ALG)
def issue_refresh(db: Session, user_id: int) -> str:
    now = datetime.now(timezone.utc); jti = str(uuid4()); exp = now + timedelta(days=settings.refresh_token_days)
    db.add(RefreshToken(user_id=user_id, jti=jti, expires_at=exp)); db.commit()
    return jwt.encode({"sub": str(user_id), "type": "refresh", "jti": jti, "iat": now, "exp": exp}, settings.jwt_secret, algorithm=ALG)

def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer), db: Session = Depends(get_db)) -> User:
    if creds is None: raise HTTPException(401, "Authentication required")
    try:
        payload = jwt.decode(creds.credentials, settings.jwt_secret, algorithms=[ALG])
        if payload.get("type") != "access": raise ValueError()
        user_id = int(payload["sub"])
    except (JWTError, ValueError, KeyError): raise HTTPException(401, "Invalid or expired access token")
    user = db.get(User, user_id)
    if not user: raise HTTPException(401, "User not found")
    return user

def rotate_refresh(token: str, db: Session) -> tuple[str, str, int]:
    try:
        p = jwt.decode(token, settings.jwt_secret, algorithms=[ALG])
        if p.get("type") != "refresh": raise ValueError()
        row = db.query(RefreshToken).filter_by(jti=p["jti"], user_id=int(p["sub"]), revoked=False).first()
        if not row or row.expires_at.replace(tzinfo=timezone.utc) <= datetime.now(timezone.utc): raise ValueError()
    except Exception: raise HTTPException(401, "Invalid or expired refresh token")
    row.revoked = True; db.commit()
    uid = int(p["sub"])
    return issue_access(uid), issue_refresh(db, uid), uid
