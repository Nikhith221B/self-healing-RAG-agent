from __future__ import annotations

import json
import random
from pathlib import Path

from config import settings

# Arms map to retrieval depth (offline learning: which top_k works best).
ARMS: dict[str, int] = {
    "top_k_3": 3,
    "top_k_5": 5,
    "top_k_8": 8,
}


def _state_path() -> Path:
    path = Path(settings.log_dir) / "bandit_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_state() -> dict[str, dict[str, float]]:
    path = _state_path()
    if not path.exists():
        return {arm: {"pulls": 0.0, "reward_sum": 0.0} for arm in ARMS}
    data = json.loads(path.read_text(encoding="utf-8"))
    for arm in ARMS:
        data.setdefault(arm, {"pulls": 0.0, "reward_sum": 0.0})
    return data


def _save_state(state: dict[str, dict[str, float]]) -> None:
    _state_path().write_text(json.dumps(state, indent=2), encoding="utf-8")


def select_top_k(epsilon: float = 0.15) -> tuple[str, int]:
    """Epsilon-greedy arm selection for retrieval top_k."""
    state = _load_state()
    if random.random() < epsilon:
        arm = random.choice(list(ARMS))
        return arm, ARMS[arm]

    best_arm = max(
        ARMS,
        key=lambda arm: _arm_score(state[arm]),
    )
    return best_arm, ARMS[best_arm]


def _arm_score(arm_state: dict[str, float]) -> float:
    pulls = arm_state.get("pulls", 0.0)
    if pulls == 0:
        return float("inf")
    return arm_state.get("reward_sum", 0.0) / pulls


def update(arm: str, reward: float) -> None:
    if arm not in ARMS:
        return
    state = _load_state()
    state[arm]["pulls"] = state[arm].get("pulls", 0.0) + 1.0
    state[arm]["reward_sum"] = state[arm].get("reward_sum", 0.0) + reward
    _save_state(state)


def reward_from_result(result: dict) -> float:
    """Map graph outcome to a bandit reward in [0, 1]."""
    status = result.get("final_status")
    feedback = result.get("critic_feedback") or {}
    score = float(feedback.get("score") or 0.0)

    if status == "accepted":
        return max(0.7, min(1.0, score if score > 0 else 1.0))
    if status == "refused":
        return 0.1
    return max(0.2, min(0.6, score))


def summary() -> dict:
    state = _load_state()
    out: dict[str, dict] = {}
    for arm, counts in state.items():
        pulls = counts.get("pulls", 0.0)
        out[arm] = {
            "top_k": ARMS.get(arm),
            "pulls": int(pulls),
            "avg_reward": round(counts.get("reward_sum", 0.0) / pulls, 3) if pulls else None,
        }
    return out
