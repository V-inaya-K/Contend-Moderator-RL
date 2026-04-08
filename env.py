from pydantic import BaseModel
from typing import Optional
import random


class Observation(BaseModel):
    content: str
    step_count: int


class Action(BaseModel):
    label: str
    severity: str
    action_type: str
    message: Optional[str] = ""


class Reward(BaseModel):
    value: float


class ModerationEnv:
    def __init__(self):
        self.base_data = [
            {"content": "I hate you, you're useless!", "label": "toxic", "severity": "high", "correct_action": "ban"},
            {"content": "You are stupid", "label": "toxic", "severity": "medium", "correct_action": "remove"},
            {"content": "Buy cheap sunglasses here!!!", "label": "spam", "severity": "medium", "correct_action": "remove"},
            {"content": "Limited offer!!! Click now!!!", "label": "spam", "severity": "high", "correct_action": "ban"},
            {"content": "Nice article, very helpful", "label": "safe", "severity": "low", "correct_action": "allow"},
            {"content": "I disagree with your opinion", "label": "safe", "severity": "low", "correct_action": "allow"},
            {"content": "You're not very smart", "label": "toxic", "severity": "low", "correct_action": "warn"},
        ]
        self.data = []
        self.index = 0
        self.done = False
        self.max_steps = 10

    def reset(self):
        self.data = self.base_data.copy()
        random.shuffle(self.data)
        self.index = 0
        self.done = False
        return Observation(content=self.data[0]["content"], step_count=0)

    def state(self):
        return {"index": self.index}

    def step(self, action: Action):
        gt = self.data[self.index]
        reward = 0.0

        # classification
        reward += 0.4 if action.label == gt["label"] else -0.2

        # severity
        if action.severity == gt["severity"]:
            reward += 0.2

        # action correctness
        reward += 0.3 if action.action_type == gt["correct_action"] else -0.3

        # message quality
        if action.message and len(action.message) > 5:
            reward += 0.1

        # over-moderation penalty
        if gt["label"] == "safe" and action.action_type in ["remove", "ban"]:
            reward -= 0.5

        # noise
        reward += random.uniform(-0.05, 0.05)

        reward = max(min(reward, 1.0), -1.0)

        self.index += 1

        if self.index >= len(self.data) or self.index >= self.max_steps:
            self.done = True
            obs = None
        else:
            obs = Observation(
                content=self.data[self.index]["content"],
                step_count=self.index
            )

        return obs, Reward(value=round(reward, 2)), self.done, {}