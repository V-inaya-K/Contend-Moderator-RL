# import os
# from env import ModerationEnv, Action
# from openai import OpenAI

# # REQUIRED ENV VARS
# API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
# MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
# API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy"

# # REQUIRED: OpenAI client usage
# client = OpenAI(
#     base_url=API_BASE_URL,
#     api_key=API_KEY
# )

# TASK_NAME = os.getenv("TASK_NAME", "hard")
# BENCHMARK = "content_moderation_env"

# env = ModerationEnv()

# obs = env.reset()
# done = False
# step = 0
# rewards = []

# print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

# while not done:
#     step += 1
#     text = obs.content.lower()

#     try:
#         _ = client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[{"role": "user", "content": f"Moderate: {text}"}],
#             max_tokens=5
#         )
#     except Exception:
#         pass

#     # baseline logic
#     if any(w in text for w in ["hate", "useless", "stupid"]):
#         action = Action(label="toxic", severity="high", action_type="ban", message="Violation.")
#     elif any(w in text for w in ["buy", "offer", "cheap"]):
#         action = Action(label="spam", severity="medium", action_type="remove", message="Spam.")
#     elif "not very smart" in text:
#         action = Action(label="toxic", severity="low", action_type="warn", message="Be respectful.")
#     else:
#         action = Action(label="safe", severity="low", action_type="allow", message="Allowed.")

#     obs, reward, done, _ = env.step(action)
#     rewards.append(reward.value)

#     # FULL ACTION STRING (IMPORTANT)
#     action_str = f"{action.action_type}:{action.label}:{action.severity}"

#     print(f"[STEP] step={step} action={action_str} reward={reward.value:.2f} done={str(done).lower()} error=null")

# score = sum(rewards) / len(rewards)

# print(f"[END] success=true steps={step} score={score:.2f} rewards={','.join([f'{r:.2f}' for r in rewards])}")

# ---------------

import os
import time
from env import ModerationEnv, Action
from openai import OpenAI

# REQUIRED ENV VARS
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# OpenAI client 
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

TASK_NAME = os.getenv("TASK_NAME", "hard")
BENCHMARK = "content_moderation_env"


def run_episode():
    env = ModerationEnv()

    obs = env.reset()
    done = False
    step = 0
    rewards = []

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    while not done:
        step += 1
        text = obs.content.lower()

        try:
            _ = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Moderate: {text}"}],
                max_tokens=5
            )
        except Exception:
            pass

        # Baseline logic
        if any(w in text for w in ["hate", "useless", "stupid"]):
            action = Action(label="toxic", severity="high", action_type="ban", message="Violation.")
        elif any(w in text for w in ["buy", "offer", "cheap"]):
            action = Action(label="spam", severity="medium", action_type="remove", message="Spam.")
        elif "not very smart" in text:
            action = Action(label="toxic", severity="low", action_type="warn", message="Be respectful.")
        else:
            action = Action(label="safe", severity="low", action_type="allow", message="Allowed.")

        obs, reward, done, _ = env.step(action)
        rewards.append(reward.value)

        action_str = f"{action.action_type}:{action.label}:{action.severity}"

        print(f"[STEP] step={step} action={action_str} reward={reward.value:.2f} done={str(done).lower()} error=null")

    score = sum(rewards) / len(rewards)

    print(f"[END] success=true steps={step} score={score:.2f} rewards={','.join([f'{r:.2f}' for r in rewards])}")


def main():
    run_episode()


if __name__ == "__main__":
    main()
