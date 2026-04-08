# 🛡️ AI Content Moderation OpenEnv

* AI Content Moderation OpenEnv is a **real-world reinforcement learning environment** that simulates how modern platforms moderate user-generated content.
* It evaluates AI agents on **multi-step decision-making tasks** including classification, severity detection, action selection, and explanation generation.

---

## ✨ Demo

* Log-based evaluation using OpenEnv-compliant `inference.py`
* Produces structured outputs in `[START]`, `[STEP]`, `[END]` format

---

## 🔗 Live Deployment

* Deployed on Hugging Face Spaces (Docker-based environment)
* Fully containerized and reproducible

---

## 🧲 Tech Stack

* Python (Core environment logic)
* Pydantic (Typed models for OpenEnv compliance)
* OpenAI Client (via Hugging Face Router)
* Docker (Containerization)
* Hugging Face Spaces (Deployment)

---

## 🚀 Workflow

1. Environment initializes with randomized moderation scenarios
2. Agent receives content via `reset()`
3. Agent performs moderation decisions:

   * classify content (`safe`, `toxic`, `spam`)
   * assign severity (`low`, `medium`, `high`)
   * choose action (`allow`, `warn`, `remove`, `ban`)
   * generate moderation message
4. Environment evaluates the action using reward function
5. Agent continues until episode ends
6. Final score is computed from cumulative rewards

---

## 🌀 Features

* Multi-Step Moderation Pipeline

  * Content classification
  * Severity detection
  * Action selection
  * Message generation

* Dense Reward Shaping

  * +0.4 → correct classification
  * +0.2 → correct severity
  * +0.3 → correct action
  * +0.1 → meaningful message
  * penalties for incorrect or harmful decisions

* Realistic Simulation

  * Randomized episodes (prevents memorization)
  * Noise injection for real-world uncertainty
  * Edge cases (mild toxicity, ambiguous content)
  * Over-moderation penalties

* Baseline Agent

  * Rule-based agent for reproducible evaluation
  * Integrated OpenAI client (as required by spec)

* OpenEnv Compliance

  * Implements `reset()`, `step()`, `state()`
  * Typed Observation, Action, Reward models

* Structured Logging

  * Strict `[START]`, `[STEP]`, `[END]` format
  * Compatible with automated evaluation pipelines

* Containerized Execution

  * Fully Dockerized
  * Runs in constrained environments (CPU/memory limits)

---

## 🎯 Tasks

* 🟢 Easy

  * Content classification only

* 🟡 Medium

  * Classification + severity detection

* 🔴 Hard

  * Full moderation pipeline:

    * classify
    * assign severity
    * choose action
    * generate message

---

## 🧩 Environment Design

### 🔹 Observation Space

```json
{
  "content": "string",
  "step_count": "int"
}
```

---

### 🔹 Action Space

```json
{
  "label": "safe | toxic | spam",
  "severity": "low | medium | high",
  "action_type": "allow | warn | remove | ban",
  "message": "string"
}
```

---

### 🔹 Reward Design

* Provides **dense feedback across the trajectory**

* Encourages correct decisions at each stage

* Penalizes:

  * incorrect classification
  * wrong actions
  * over-moderation (e.g., banning safe content)

* Reward range: **[-1.0, 1.0]**

---

## 🌊 Setup

1. Clone / Download repo into your local machine.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Create `.env` file:

```bash
HF_TOKEN=your_token
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

4. Run the environment:

```bash
python inference.py
```

---

## 🐳 Docker Setup

```bash
docker build -t moderation-env .
docker run moderation-env
```

---

## 📊 Example Output

```bash
[START] task=hard env=content_moderation_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=allow:safe:low reward=0.99 done=false error=null
[STEP] step=2 action=remove:spam:medium reward=0.13 done=false error=null
...
[END] success=true steps=7 score=0.74 rewards=...
```

---

## 🧠 Key Highlights

* Real-world content moderation simulation
* Multi-step agent evaluation
* Dense reward shaping
* Non-deterministic environment (randomization + noise)
* Robust and reproducible baseline

---

## 🏁 Conclusion

This environment provides a **practical benchmark for evaluating AI moderation systems**, combining structured decision-making with real-world complexity.

It is designed to be:

* realistic
* scalable
* useful for agent evaluation

---

## 📌 Future Improvements

* Larger and more diverse datasets
* LLM-based moderation policies
* Multi-agent moderation workflows
* Adversarial content scenarios

---

## 👨‍💻 Author

Developed as part of OpenEnv benchmark submission.
