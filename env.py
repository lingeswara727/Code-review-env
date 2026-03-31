"""
env.py — CodeReviewEnv
OpenEnv-compliant RL environment for AI code review.
Tasks: easy, medium, hard
"""

import re
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────
#  TASKS DEFINITION
# ─────────────────────────────────────────────
TASKS = {
    "easy": {
        "difficulty": "easy",
        "instructions": "Review the following Python function. Identify if there is a bug and suggest a fix.",
        "code_snippet": """\
def add_numbers(a, b):
    return a - b  # BUG: should be a + b
""",
        "has_bug": True,
        "bug_description": "subtraction used instead of addition",
        "fixed_code": """\
def add_numbers(a, b):
    return a + b
""",
        "max_steps": 1,
    },

    "medium": {
        "difficulty": "medium",
        "instructions": (
            "Review the following Python function. "
            "It should return the factorial of n recursively. "
            "Find any bugs and fix them."
        ),
        "code_snippet": """\
def factorial(n):
    if n == 0:
        return 0  # BUG: should return 1
    return n * factorial(n - 1)
""",
        "has_bug": True,
        "bug_description": "base case returns 0 instead of 1, making all factorials 0",
        "fixed_code": """\
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
""",
        "max_steps": 2,
    },

    "hard": {
        "difficulty": "hard",
        "instructions": (
            "Review the following Python function. "
            "It should safely divide two numbers and handle division by zero. "
            "Find all bugs and fix them."
        ),
        "code_snippet": """\
def safe_divide(a, b):
    if b = 0:           # BUG 1: assignment instead of ==
        return None
    result = a // b     # BUG 2: integer division, should be float
    return results      # BUG 3: typo — 'results' not defined
""",
        "has_bug": True,
        "bug_description": "three bugs: assignment in condition, integer division, undefined variable typo",
        "fixed_code": """\
def safe_divide(a, b):
    if b == 0:
        return None
    result = a / b
    return result
""",
        "max_steps": 3,
    },
}


# ─────────────────────────────────────────────
#  OBSERVATION
# ─────────────────────────────────────────────
@dataclass
class CodeReviewObservation:
    task_id:      str
    difficulty:   str
    instructions: str
    code_snippet: str
    step:         int


# ─────────────────────────────────────────────
#  ENVIRONMENT
# ─────────────────────────────────────────────
class CodeReviewEnv:
    """
    OpenEnv-compliant RL environment.

    Observation : CodeReviewObservation
    Action      : dict with keys:
                    has_bug         (bool)
                    bug_description (str)
                    fixed_code      (str)
                    quality_score   (float 0-1)

    Reward breakdown (max 1.0 per step):
        +0.30  correct bug detection
        +0.20  correct bug description
        +0.30  correct fixed code
        +0.20  quality score reasonable (0.6-1.0 for buggy, 0-0.5 for clean)
    """

    def __init__(self, task_id: str = "easy"):
        assert task_id in TASKS, f"Unknown task: {task_id}. Choose from {list(TASKS.keys())}"
        self.task_id   = task_id
        self.task      = TASKS[task_id]
        self._step     = 0
        self._done     = False
        self._history  = []
        self._total_reward = 0.0

    # ── OpenEnv API ──────────────────────────

    def reset(self) -> CodeReviewObservation:
        self._step         = 0
        self._done         = False
        self._history      = []
        self._total_reward = 0.0
        return self._observe()

    def step(self, action: dict):
        assert not self._done, "Episode done. Call reset() first."

        reward, breakdown, feedback = self._score(action)
        self._total_reward += reward
        self._step         += 1

        if self._step >= self.task["max_steps"]:
            self._done = True

        info = {
            "feedback":         feedback,
            "reward_breakdown": breakdown,
            "total_reward":     self._total_reward,
            "done":             self._done,
        }

        self._history.append({
            "step":    self._step,
            "action":  action,
            "reward":  reward,
            "info":    info,
        })

        obs = self._observe()
        return obs, reward, self._done, info

    def state(self) -> dict:
        return {
            "task_id":      self.task_id,
            "step":         self._step,
            "done":         self._done,
            "total_reward": self._total_reward,
            "history":      self._history,
        }

    # ── Internal ─────────────────────────────

    def _observe(self) -> CodeReviewObservation:
        return CodeReviewObservation(
            task_id      = self.task_id,
            difficulty   = self.task["difficulty"],
            instructions = self.task["instructions"],
            code_snippet = self.task["code_snippet"],
            step         = self._step,
        )

    def _score(self, action: dict):
        task      = self.task
        breakdown = {}
        feedback  = []

        # ── 1. Bug detection (+0.30) ──
        correct_bug = action.get("has_bug") == task["has_bug"]
        breakdown["bug_detection"] = 0.30 if correct_bug else 0.0
        feedback.append(
            "✓ Correctly identified bug presence"
            if correct_bug else
            "✗ Wrong bug detection"
        )

        # ── 2. Bug description (+0.20) ──
        desc       = str(action.get("bug_description", "")).lower()
        true_desc  = task["bug_description"].lower()
        # Check keyword overlap
        keywords   = [w for w in true_desc.split() if len(w) > 3]
        matches    = sum(1 for k in keywords if k in desc)
        desc_score = 0.20 * (matches / max(len(keywords), 1)) if task["has_bug"] else 0.20
        breakdown["bug_description"] = round(desc_score, 4)
        feedback.append(
            f"✓ Bug description score: {desc_score:.2f}"
            if desc_score > 0.10 else
            "✗ Bug description too vague"
        )

        # ── 3. Fixed code (+0.30) ──
        fixed      = str(action.get("fixed_code", ""))
        true_fixed = task["fixed_code"]
        code_score = self._code_similarity(fixed, true_fixed) * 0.30
        breakdown["fixed_code"] = round(code_score, 4)
        feedback.append(
            f"✓ Fixed code score: {code_score:.2f}"
            if code_score > 0.15 else
            "✗ Fixed code incorrect"
        )

        # ── 4. Quality score (+0.20) ──
        qs = float(action.get("quality_score", 0.5))
        if task["has_bug"]:
            # Buggy code: expect low quality (0.0–0.5)
            qual_score = 0.20 if qs <= 0.5 else max(0, 0.20 - (qs - 0.5) * 0.4)
        else:
            # Clean code: expect high quality (0.6–1.0)
            qual_score = 0.20 if qs >= 0.6 else max(0, 0.20 - (0.6 - qs) * 0.4)
        breakdown["quality_score"] = round(qual_score, 4)
        feedback.append(
            f"✓ Quality score reasonable: {qs:.2f}"
            if qual_score >= 0.15 else
            f"✗ Quality score unreasonable: {qs:.2f}"
        )

        total = sum(breakdown.values())
        breakdown["total"] = round(total, 4)

        return round(total, 4), breakdown, " | ".join(feedback)

    def _code_similarity(self, a: str, b: str) -> float:
        """Simple token overlap similarity."""
        def tokens(s):
            return set(re.findall(r'\w+', s.lower()))
        ta, tb = tokens(a), tokens(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)