
## Task Definition

Navigate a 9×9 grid from **Start (0,0)** to **Goal (8,8)**
while avoiding obstacles — in as few steps as possible.

| Element   | Symbol | Description                        |
|-----------|--------|------------------------------------|
| Agent     | 🔵     | Starts at top-left (0,0)           |
| Goal      | ⭐     | Bottom-right corner (8,8)          |
| Obstacle  | ✕      | 10 fixed cells — avoid these       |
| Free cell | ·      | Walkable grid cell                 |

---

## Reward Function

| Event              | Reward  |
|--------------------|---------|
| Reach goal         | +100.0  |
| Hit obstacle       | −10.0   |
| Hit wall boundary  | −5.0    |
| Each step taken    | −1.0    |

The step penalty encourages **efficiency** — shorter paths score higher.

---

## Observation Space

```
Box(low=[0,0], high=[8,8], shape=(2,), dtype=int32)
obs[0] = agent row
obs[1] = agent col
```

## Action Space

```
Discrete(4)
0 = UP
1 = DOWN
2 = LEFT
3 = RIGHT
```

---

## Installation

```bash
pip install gymnasium numpy pygame
```

---

## Usage

```python
from grid_world_env import GridWorldEnv

env = GridWorldEnv(render_mode="ansi")
obs, info = env.reset(seed=42)

for _ in range(200):
    action = env.action_space.sample()          # replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)
    print(env.render())
    if terminated or truncated:
        break

env.close()
```

### Visual Mode
```bash
python grid_world_env.py --visual
```

---

## Run Grader

```python
from grid_world_env import GridWorldEnv, grade_environment

results = grade_environment(GridWorldEnv, num_episodes=10)
print(f"Score: {results['score']} / 100")
```

### Grader checks:
- ✅ Gymnasium API compliance (`reset`, `step`, spaces)
- ✅ Correct observation & reward types
- ✅ Goal is reachable
- ✅ Reward consistency
- ✅ Success rate over 10 episodes

---

## Episode Termination

- ✅ **Success** — agent reaches (8,8)
- ⏱ **Truncated** — max 200 steps exceeded

---

## Obstacles (Fixed)

```
(2,3), (2,4), (3,1), (3,2),
(4,3), (4,4), (4,5), (5,2),
(6,4), (6,5)
```

---

*Built for Meta RL Hackathon · Round 1 · March–April 2025*