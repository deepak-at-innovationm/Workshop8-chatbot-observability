# LangSmith walkthrough

This demo uses one parent trace per chat request. LangChain creates the model and tool runs underneath that parent. The app does not wrap those calls a second time, so each OpenAI request and each tool execution should appear once.

## Before you start

1. Put `OPENAI_API_KEY` and `LANGSMITH_API_KEY` in `.env`.
2. Set `LANGSMITH_TRACING=true`.
3. Note `LANGSMITH_PROJECT`. The server prints it on startup:

```text
INFO app: MLOps Chatbot Demo starting. tracing=True project=<your project> model=gpt-5-nano
```

4. Start the app:

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

5. Open LangSmith, select the project from that log line, and keep the Tracing page visible.

If this workspace's existing `.env` is unchanged, the project name is `chatbot-mlops-demo`. `.env.example` uses `mlops-chatbot-demo`. The startup line is the source of truth.

## What a normal trace contains

Each UI request creates a root run named `chat_request`.

| Run | What it is |
| --- | --- |
| `chat_request` | Parent chain. Inputs are the question, session id, request id, and history. |
| `decide_tool_use` | OpenAI call with `get_current_datetime` bound. This is where the model chooses whether to call the tool. |
| `get_current_datetime` | Present only when the model actually calls the tool. Output is JSON with `date`, `time`, `timezone`, and `iso8601`. |
| `final_answer` | Second OpenAI call. It sees the tool JSON and writes the user-facing answer. |

Metadata on these runs:

- `thread_id` and `session_id` — the same value as the session id in the web page. LangSmith groups those runs into a thread.
- `request_id` — matches the request id under the chat transcript.
- `application` — `MLOps Chatbot Demo`
- `model_provider` — `openai`
- `model_name` — the configured `LLM_MODEL`
- `workflow_type` — `chat` or `tool`

Filter the tracing table by `session_id` or `thread_id` when several people share a project. Open a run to see latency. Token counts are on the OpenAI runs when the provider returns usage.

The trace inputs do not include API keys. The service object is removed before the run is recorded.

## Demo sequence

### 1. Ask a normal question

In the chat UI, send:

```text
What is MLOps?
```

In LangSmith, open the new `chat_request`. You should see `decide_tool_use` and no `get_current_datetime` child. The assistant reply is the `chat_request` output. The web page shows `tool_used` only when a tool ran, so this reply has no tool note.

### 2. Ask for the clock

Send:

```text
What is the current date and time?
```

Open that `chat_request`. The order should be:

1. `decide_tool_use` with a tool call in the model output.
2. `get_current_datetime` with the JSON clock.
3. `final_answer`, whose input includes that JSON.

The answer in the UI should match the tool's date, time, and `Asia/Kolkata` timezone (or `APP_TIMEZONE` if you changed it). The bubble shows `Tool used: get_current_datetime`.

Other prompts that should take the same path: `What time is it now?` and `Tell me today's date.`

A question such as `What is CI/CD for machine learning?` should not add a tool run.

### 3. Follow a thread

Ask a second question in the same chat, without clicking **New chat**. Both root runs share `thread_id`. In LangSmith, open Threads (or filter runs by that metadata value) and confirm the second trace's inputs include the earlier turn.

Click **New chat** and ask another question. The new session id must not appear on the previous thread.

### 4. Trigger a failed trace

1. Stop the server.
2. Set `ALLOW_DEMO_FAULTS=true`.
3. Start the server again. The page shows **Simulate traced error**.
4. Click that button, or run:

```bash
curl -s -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -H 'X-Demo-Fault: model' \
  -d '{"message":"Simulate a model failure.","session_id":"demo-error"}'
```

The browser shows: `The assistant could not complete this request. Please try again.`

In LangSmith, the `chat_request` run is marked as an error. The error text is `Simulated model failure for the LangSmith demo.` There is no model child, because the failure is raised before OpenAI is called. The HTTP body does not include that internal sentence or a stack trace.

Turn `ALLOW_DEMO_FAULTS` back to `false` and restart. Requests without the header never hit this path, including when the flag is left on by mistake.

### 5. Run evaluation

```bash
source .venv/bin/activate
python scripts/evaluate.py
```

The SDK prints an experiment URL. Open it from the dataset `mlops-chatbot-demo-eval`.

Columns to review:

- `tool_selection` — 1 when the recorded tool decision matches the example. Datetime rows must have `tool_used=true` and `tool_name=get_current_datetime`. General rows must have `tool_used=false`.
- `error_handling` — the simulated failure must be the safe public message.
- `datetime_grounding` — scored only on datetime rows; the answer should contain today's date.
- `relevance` — scored only on general questions, and only if you did not pass `--no-llm-judge`.

A relevance score of 1 means the judge model thought the answer addressed the question. It does not certify the answer.

Compare two runs by executing the script again after a prompt or model change. LangSmith's experiment compare view uses the same dataset.

Local-only scoring, with no experiment upload:

```bash
python scripts/evaluate.py --no-upload
```

## Tracing off

Set `LANGSMITH_TRACING=false` and restart. Chat still answers. New requests do not need `LANGSMITH_API_KEY`. The startup log shows `tracing=False`.

## Latency and tokens

Open `decide_tool_use` or `final_answer` and read the run duration and token usage in the run details. The parent `chat_request` duration includes the model call and, when used, the tool. The tool itself is a local clock read and should be a small fraction of the parent time.
