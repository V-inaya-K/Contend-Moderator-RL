from fastapi import FastAPI
from env import ModerationEnv, Action

app = FastAPI()

env = ModerationEnv()

@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs.model_dump()}

@app.post("/step")
def step(action: dict):
    act = Action(**action)
    obs, reward, done, info = env.step(act)

    return {
        "observation": obs.model_dump() if obs else None,
        "reward": {"value": reward.value},
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return {"state": env.state()}

def main():
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)