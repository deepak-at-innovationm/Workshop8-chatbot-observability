# MLOps Chatbot Demo

A small local chatbot for demonstrating LangSmith tracing, OpenAI tool calling, debugging, and evaluation.

The assistant answers general questions with OpenAI. It exposes one LangChain tool, `get_current_datetime`. The model decides when to call that tool. Conversation history is kept in memory for the life of the server process, separately for each session.

There is no retrieval pipeline, vector database, authentication, or Docker setup.

## Architecture

```text
Browser  -->  GET /  and  POST /api/chat
                 |
                 v
           FastAPI route
                 |
                 v
           ChatService.answer
                 |
                 +--> chat_request          (parent LangSmith trace)
                        |
                        +--> decide_tool_use   (OpenAI, tools bound)
                        +--> get_current_datetime   (only if the model calls it)
                        +--> final_answer      (OpenAI, after the tool result)
```

LangGraph is not used. Tool calling uses `ChatOpenAI.bind_tools`.

## Prerequisites

- Python 3.10 or newer. This project was installed and tested with the Python 3.14 virtual environment in `.venv`.
- An OpenAI API key from the [OpenAI platform](https://platform.openai.com/api-keys).
- A LangSmith API key only if you want traces or uploaded evaluation experiments. The chat still runs with `LANGSMITH_TRACING=false` and no LangSmith key.

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The virtual environment already in this workspace was created without a `pip` module. From that environment, install with `uv`:

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Environment variables

```bash
cp .env.example .env
```

If `.env` already exists, do not replace it. Add any missing keys from `.env.example` instead.

Required for answers:

```env
OPENAI_API_KEY=your-openai-key
```

Tracing:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=mlops-chatbot-demo
```

Defaults used when the variables are unset:

| Variable | Default |
| --- | --- |
| `LLM_MODEL` | `gpt-5-nano` |
| `APP_TIMEZONE` | `Asia/Kolkata` |
| `HISTORY_MAX_MESSAGES` | `10` |
| `ALLOW_DEMO_FAULTS` | `false` |
| `LANGSMITH_PROJECT` | `mlops-chatbot-demo` |

`gpt-5-nano` supports function calling. The app does not set `temperature`, so the API applies the model's own default.

An existing `.env` in this workspace may already set `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`. Those values override the defaults. Check the project name printed at startup before you look for traces. Add `OPENAI_API_KEY` before you expect live answers.

## Run the app

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Useful questions:

- `What is the capital of France?` — answered by the model, no tool call.
- `What is the current date and time?` — the model should call `get_current_datetime`.
- `What time is it now?`
- `Tell me today's date.`

The page shows the session id and the latest request id. Use **New chat** to start a session that does not see the previous one. History is stored in memory and disappears when the server stops.

## Tests

Offline unit tests mock the chat model and do not need API keys:

```bash
source .venv/bin/activate
pytest
```

Live OpenAI checks are skipped unless you opt in:

```bash
RUN_INTEGRATION_TESTS=1 pytest -m integration
```

That command calls OpenAI and consumes API quota. It checks that a knowledge question does not record a tool call and that a clock question does.

## Evaluation

The dataset is `data/evaluation/examples.json` (8 examples): everyday general questions, datetime questions, and one forced model failure.

```bash
source .venv/bin/activate
python scripts/evaluate.py
```

This requires `OPENAI_API_KEY`. When `LANGSMITH_API_KEY` is set, the script syncs a dataset named `mlops-chatbot-demo-eval` and uploads an experiment with the prefix `mlops-chatbot`. The LangSmith URL is printed by the SDK.

Local scoring only:

```bash
python scripts/evaluate.py --no-upload
```

Skip the extra relevance judge (tool and error checks still run):

```bash
python scripts/evaluate.py --no-llm-judge
```

Deterministic checks:

- **tool_selection** — compares `tool_used` recorded by the app with the expected flag. An answer that merely mentions a time does not count as a tool call.
- **error_handling** — the failure example must return the public error text, without a traceback or credential.
- **response_present** — successful answers are non-empty.
- **datetime_grounding** — after a real tool call, the answer should contain today's date in `APP_TIMEZONE`.

The optional **relevance** judge asks the configured OpenAI model whether a general answer addresses the question. That score is a judgment, not proof that the answer is correct.

## Controlled error

Ordinary requests never take this path.

1. Set `ALLOW_DEMO_FAULTS=true` in `.env`.
2. Restart the server.
3. The page shows **Simulate traced error**, or send the header yourself:

```bash
curl -s -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -H 'X-Demo-Fault: model' \
  -d '{"message":"Simulate a model failure.","session_id":"demo-error"}'
```

The API responds with: `The assistant could not complete this request. Please try again.` The LangSmith trace for `chat_request` records `Simulated model failure for the LangSmith demo.` Set the flag back to `false` after the demo. Leaving it `true` does not fail requests that omit the header.

## Tracing

See [docs/LANGSMITH_DEMO.md](docs/LANGSMITH_DEMO.md) for the click-through in the LangSmith UI.

Tracing is optional. With `LANGSMITH_TRACING=false`, chat works without `LANGSMITH_API_KEY`. Upload failures are not returned to the user as the answer.

## Dependency choice

| Package | Version | Role |
| --- | --- | --- |
| `langchain-core` | 1.6.13 | Messages, tools, run config |
| `langchain-openai` | 1.7.2 | `ChatOpenAI` and OpenAI tool calling |
| `langsmith` | 0.14.12 | Traces and `evaluate` |
| `fastapi` | 0.119.0 | HTTP API |
| `uvicorn` | 0.37.0 | Local server |
| `pytest` | 8.4.2 | Tests |

`langchain-openai` 1.7.2 requires `langchain-core` 1.x (`>=1.6.10`). The application code does not depend on the umbrella `langchain` package or on LangGraph.

On Python 3.14, FastAPI may print a warning about Pydantic v1 compatibility. The API still serves requests.

## Troubleshooting

- **`OPENAI_API_KEY is not configured`** — add the key to `.env` and restart uvicorn. The process reads the file at startup.
- **No traces** — startup logs `tracing=` and `project=`. Those are the values LangSmith will use. A key in `.env` is ignored while `LANGSMITH_TRACING=false`.
- **Traces appear in an unexpected project** — `LANGSMITH_PROJECT` in the existing `.env` overrides the example value.
- **The clock tool was not called** — open the `decide_tool_use` child run. The model is instructed to call the tool only for date and time questions. Retry with `What is the current date and time?`.
- **Tool called for a general question** — the same child run shows the tool decision. Tighten nothing in code for a one-off miss; it is the model's choice. The evaluation `tool_selection` score records how often it happens.
- **Evaluation cannot start** — `python scripts/evaluate.py` exits with code 2 until `OPENAI_API_KEY` is set. `pytest` still checks the scoring functions offline.

## Limitations

- History is in memory. Restarting the server clears every session, and two server processes do not share sessions.
- One tool is available. The model can decline to call it, or call it when it should not. The trace is the place to see which happened.
- Answers to general questions come from OpenAI, not from a document index.
- Evaluation relevance is an extra model opinion. Do not treat the experiment as a proof of correctness.
- The live OpenAI path and the LangSmith UI were not exercised in the offline test run. Run the integration command and the evaluation command after adding keys.
