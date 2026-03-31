"""
baseline.py — Baseline agent for CodeReviewEnv
Uses Groq LLM to review code across easy / medium / hard tasks.
"""
import os
import json
from dotenv import load_dotenv
from groq import Groq
from env import CodeReviewEnv, TASKS   # ✅ Fixed import

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert Python code reviewer.
Respond ONLY with a valid JSON object — no markdown, no explanation.

Required format:
{
  "has_bug": true or false,
  "bug_description": "description or 'none'",
  "fixed_code": "corrected full function",
  "quality_score": 0.0 to 1.0
}"""


def run_task(task_id: str) -> dict:
    env = CodeReviewEnv(task_id=task_id)
    obs = env.reset()
    total_reward = 0.0
    steps = 0

    print(f"\n{'='*50}")
    print(f"Task: {task_id.upper()} | Difficulty: {obs.difficulty}")
    print(f"{'='*50}")

    while not env.state()["done"]:
        user_prompt = f"""Instructions: {obs.instructions}

Code to review:
```python
{obs.code_snippet}
```
Respond with JSON only."""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()

        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  [WARN] Bad JSON: {raw[:80]}")
            action = {
                "has_bug": False,
                "bug_description": "none",
                "fixed_code": obs.code_snippet,
                "quality_score": 0.5,
            }

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        print(f"  Step {steps}: reward={reward:.4f}")
        print(f"  Feedback: {info['feedback']}")
        print(f"  Breakdown: {info['reward_breakdown']}")

    avg = total_reward / steps if steps > 0 else 0.0
    print(f"\n  → Avg reward: {avg:.4f} over {steps} step(s)")
    return {"task_id": task_id, "avg_reward": avg, "steps": steps}


def main():
    print(f"Baseline agent ({MODEL}) — all tasks\n")
    results = {}

    for task_id in TASKS:
        results[task_id] = run_task(task_id)

    print(f"\n{'='*50}\nBASELINE SUMMARY\n{'='*50}")
    overall = 0.0
    for tid, r in results.items():
        print(f"  {tid:8s}: {r['avg_reward']:.4f}")
        overall += r["avg_reward"]
    print(f"  {'OVERALL':8s}: {overall/len(results):.4f}")

    with open("baseline_scores.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved → baseline_scores.json")


if __name__ == "__main__":
    main()