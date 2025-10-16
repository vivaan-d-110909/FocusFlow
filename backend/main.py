# backend/main.py
"""
Complete FastAPI backend for FocusFlow.

Features included in this single file:
- SQLite + SQLAlchemy setup (auto-create DB and tables)
- Pydantic schemas
- Task CRUD endpoints
- Timer session start/stop endpoints (records FocusSession)
- Simple stats endpoint (aggregates focus durations)
- AI feedback endpoint that calls Groq model (configurable via env)
- Serves frontend static files from ../frontend
- CORS enabled for local dev
- Safe loading of secrets from .env
- Run with: uvicorn backend.main:app --reload
"""

import os
import datetime
import json
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Interval,
    func,
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from dotenv import load_dotenv
import requests

# Load environment variables from .env (if present)
load_dotenv()

# ---- Configuration ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_URL = os.getenv(
    "DATABASE_URL", f"sqlite:///{os.path.join(DATA_DIR, 'focusflow.db')}"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "gpt-120oss-b")
# Default guessed base URL — change in .env if Groq uses different path
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/v1/models")

# ---- Database setup ----
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---- Models ----
class UserTask(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    description = Column(String(1000), nullable=True)
    status = Column(String(50), default="todo")  # todo, doing, done
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    sessions = relationship("FocusSession", back_populates="task", cascade="all, delete")


class FocusSession(Base):
    __tablename__ = "focus_sessions"
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_secs = Column(Integer, nullable=True)  # store seconds for easy aggregation
    notes = Column(String(1000), nullable=True)
    task = relationship("UserTask", back_populates="sessions")


# Create tables
Base.metadata.create_all(bind=engine)

# ---- Pydantic Schemas ----
class TaskCreate(BaseModel):
    title: str = Field(..., max_length=200)
    description: Optional[str] = None


class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = None
    status: Optional[str] = None


class TaskOut(BaseModel):
    id: int
    title: str
    description: Optional[str]
    status: str
    created_at: datetime.datetime

    class Config:
        orm_mode = True


class StartSessionIn(BaseModel):
    task_id: Optional[int] = None
    notes: Optional[str] = None


class StopSessionIn(BaseModel):
    session_id: int
    notes: Optional[str] = None


class SessionOut(BaseModel):
    id: int
    task_id: Optional[int]
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime]
    duration_secs: Optional[int]
    notes: Optional[str]

    class Config:
        orm_mode = True


class AIRequest(BaseModel):
    prompt: Optional[str] = None  # if not provided, server will synthesize a prompt from stats


# ---- FastAPI app ----
app = FastAPI(title="FocusFlow Backend", version="0.1.0")

# Allow requests from local frontend during development; adjust origin list for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during dev allow all; restrict to your domain(s) in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
else:
    # If the frontend folder doesn't exist, just make sure path is set
    os.makedirs(FRONTEND_DIR, exist_ok=True)


# Helper: load index.html or return simple page
@app.get("/", response_class=HTMLResponse)
def read_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    # fallback simple HTML
    return HTMLResponse(
        """
        <!doctype html>
        <html>
          <head><meta charset="utf-8"><title>FocusFlow</title></head>
          <body>
            <h1>FocusFlow Backend is running</h1>
            <p>Create a frontend file at frontend/index.html to use the UI.</p>
            <p>API root: <a href="/docs">/docs (OpenAPI)</a></p>
          </body>
        </html>
        """
    )


# ---- Task endpoints ----
@app.post("/api/tasks", response_model=TaskOut, status_code=201)
def create_task(payload: TaskCreate, db: Session = Depends(get_db)):
    task = UserTask(title=payload.title.strip(), description=payload.description)
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@app.get("/api/tasks", response_model=List[TaskOut])
def list_tasks(db: Session = Depends(get_db)):
    tasks = db.query(UserTask).order_by(UserTask.created_at.desc()).all()
    return tasks


@app.get("/api/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(UserTask).filter(UserTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.patch("/api/tasks/{task_id}", response_model=TaskOut)
def update_task(task_id: int, payload: TaskUpdate, db: Session = Depends(get_db)):
    task = db.query(UserTask).filter(UserTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if payload.title is not None:
        task.title = payload.title.strip()
    if payload.description is not None:
        task.description = payload.description
    if payload.status is not None:
        task.status = payload.status
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@app.delete("/api/tasks/{task_id}", status_code=204)
def delete_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(UserTask).filter(UserTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    db.delete(task)
    db.commit()
    return JSONResponse(status_code=204, content={})


# ---- Timer session endpoints ----
@app.post("/api/timer/start", response_model=SessionOut)
def start_session(payload: StartSessionIn, db: Session = Depends(get_db)):
    # Optionally verify task exists
    if payload.task_id is not None:
        t = db.query(UserTask).filter(UserTask.id == payload.task_id).first()
        if not t:
            raise HTTPException(status_code=404, detail="Task not found")
    session = FocusSession(
        task_id=payload.task_id,
        start_time=datetime.datetime.utcnow(),
        notes=payload.notes,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.post("/api/timer/stop", response_model=SessionOut)
def stop_session(payload: StopSessionIn, db: Session = Depends(get_db)):
    session = db.query(FocusSession).filter(FocusSession.id == payload.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.end_time is not None:
        raise HTTPException(status_code=400, detail="Session already stopped")
    now = datetime.datetime.utcnow()
    session.end_time = now
    duration = int((session.end_time - session.start_time).total_seconds())
    session.duration_secs = max(0, duration)
    if payload.notes:
        # append notes safely
        existing = session.notes or ""
        session.notes = (existing + "\n" + payload.notes).strip()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.get("/api/sessions", response_model=List[SessionOut])
def list_sessions(limit: int = 100, db: Session = Depends(get_db)):
    sessions = (
        db.query(FocusSession)
        .order_by(FocusSession.start_time.desc())
        .limit(limit)
        .all()
    )
    return sessions


# ---- Stats endpoint ----
@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    # Total focused time (seconds)
    total = db.query(func.sum(FocusSession.duration_secs)).scalar() or 0
    # Number of sessions
    count = db.query(func.count(FocusSession.id)).scalar() or 0
    # Average session length
    avg = db.query(func.avg(FocusSession.duration_secs)).scalar() or 0

    # Top tasks by total focus time
    top_tasks = (
        db.query(
            UserTask.id,
            UserTask.title,
            func.coalesce(func.sum(FocusSession.duration_secs), 0).label("total_secs"),
        )
        .outerjoin(FocusSession, FocusSession.task_id == UserTask.id)
        .group_by(UserTask.id)
        .order_by(func.sum(FocusSession.duration_secs).desc())
        .limit(10)
        .all()
    )
    top_tasks_list = [
        {"task_id": t.id, "title": t.title, "total_secs": int(t.total_secs or 0)}
        for t in top_tasks
    ]

    return {
        "total_focused_secs": int(total),
        "session_count": int(count),
        "avg_session_secs": float(avg) if avg is not None else 0.0,
        "top_tasks": top_tasks_list,
    }


# ---- AI Feedback ----
def call_groq_model(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    Call Groq model API to get text output.
    This is a defensive implementation — the actual Groq API shape may differ.
    Configure GROQ_API_URL and GROQ_MODEL via .env if endpoints differ.
    """
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env (GROQ_API_KEY=...)."
        )

    # Compose a best-guess endpoint; users can override GROQ_API_URL in .env if needed.
    endpoint = f"{GROQ_API_URL}/{GROQ_MODEL}/outputs"

    # payload shape is based on many modern LLM APIs — if Groq uses a different shape,
    # update this function accordingly.
    body = {
        "input": prompt,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
        # Add other params as needed by Groq API
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(endpoint, json=body, headers=headers, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to reach Groq API: {str(e)}")

    if resp.status_code >= 400:
        # Include response text for debugging (but don't log secrets)
        raise RuntimeError(
            f"Groq API error {resp.status_code}: {resp.text[:400]}"
        )

    # Attempt to parse common response patterns
    data = resp.json()
    # attempt a few shapes:
    if isinstance(data, dict):
        # common pattern: {'outputs': [{'content': '...'}]} or {'text': '...'}
        if "outputs" in data and isinstance(data["outputs"], list):
            first = data["outputs"][0]
            # try several keys
            for k in ("content", "text", "output"):
                if isinstance(first, dict) and k in first:
                    return str(first[k])
            # fallback to stringified first item
            return json.dumps(first)
        if "text" in data:
            return str(data["text"])
        if "output" in data and isinstance(data["output"], str):
            return data["output"]
    # fallback: stringify whole response (truncated)
    return json.dumps(data)[:4000]


def synthesize_prompt_from_stats(stats: dict) -> str:
    # Simple prompt builder — you can improve prompt engineering later
    lines = [
        "You are a friendly productivity coach. Analyze the user's focus data and provide:",
        "1) A short summary of the user's current focus behaviour.",
        "2) 3 practical suggestions to improve focus (concise).",
        "3) A suggested schedule (times of day) for their best focus slots if possible.",
        "",
        "Data:",
        json.dumps(stats),
        "",
        "Keep the answer short (<= 300 words), actionable, and friendly.",
    ]
    return "\n".join(lines)


@app.post("/api/ai_feedback")
def ai_feedback(request: AIRequest, db: Session = Depends(get_db)):
    """
    Returns AI-generated feedback. If prompt not provided, server synthesizes one from stats.
    Uses Groq model specified by GROQ_MODEL and GROQ_API_KEY.
    """
    # Gather stats to include if needed
    stats = get_stats(db)

    prompt = request.prompt.strip() if request.prompt else synthesize_prompt_from_stats(stats)

    try:
        result_text = call_groq_model(prompt=prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI request failed: {str(e)}")

    return {"model": GROQ_MODEL, "feedback": result_text, "used_prompt": prompt}


# ---- Health and info endpoints ----
@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat()}


@app.get("/api/info")
def info():
    return {
        "app": "FocusFlow",
        "version": "0.1.0",
        "db": DATABASE_URL,
        "groq_model": GROQ_MODEL,
        "groq_api_url": GROQ_API_URL,
        "groq_available": bool(GROQ_API_KEY),
    }


# ---- Error handlers ----
@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
def generic_exception_handler(request: Request, exc: Exception):
    # For production, replace this with robust logging and less noisy error messages.
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ---- Run if invoked directly ----
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
