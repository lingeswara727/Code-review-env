"""
App.py — CodeReviewEnv FastAPI Server
OpenEnv-compliant endpoints for Meta RL Hackathon Round 1
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from env import CodeReviewEnv, TASKS

app = FastAPI(
    title="CodeReviewEnv",
    description="OpenEnv-compliant AI code review RL environment",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# One env per task (in-memory)
sessions: dict = {}

def get_env(task_id: str) -> CodeReviewEnv:
    if task_id not in TASKS:
        raise HTTPException(404, f"Unknown task '{task_id}'. Use: {list(TASKS.keys())}")
    if task_id not in sessions:
        sessions[task_id] = CodeReviewEnv(task_id=task_id)
    return sessions[task_id]

class ReviewAction(BaseModel):
    has_bug:         bool
    bug_description: str
    fixed_code:      str
    quality_score:   float

@app.get("/")
def root():
    return {
        "env":       "CodeReviewEnv",
        "tasks":     list(TASKS.keys()),
        "endpoints": ["/reset/{task_id}", "/step/{task_id}", "/state/{task_id}", "/health"]
    }

@app.get("/ui")
def serve_ui():
    return FileResponse("static/r1-api-tester.html")

@app.get("/health")
def health():
    return {"status": "ok", "env": "CodeReviewEnv"}

@app.post("/reset/{task_id}")
def reset(task_id: str):
    env = get_env(task_id)
    obs = env.reset()
    return {
        "task_id":      obs.task_id,
        "difficulty":   obs.difficulty,
        "instructions": obs.instructions,
        "code_snippet": obs.code_snippet,
        "step":         obs.step,
    }

@app.post("/step/{task_id}")
def step(task_id: str, action: ReviewAction):
    env = get_env(task_id)
    try:
        obs, reward, done, info = env.step(action.dict())
    except AssertionError as e:
        raise HTTPException(400, str(e))
    return {
        "observation": {
            "task_id":      obs.task_id,
            "difficulty":   obs.difficulty,
            "instructions": obs.instructions,
            "code_snippet": obs.code_snippet,
            "step":         obs.step,
        },
        "reward": reward,
        "done":   done,
        "info":   info,
    }

@app.get("/state/{task_id}")
def state(task_id: str):
    return get_env(task_id).state()