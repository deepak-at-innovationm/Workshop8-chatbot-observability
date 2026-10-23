const messagesEl = document.querySelector("#messages");
const form = document.querySelector("#form");
const input = document.querySelector("#input");
const send = document.querySelector("#send");
const loading = document.querySelector("#loading");
const errorEl = document.querySelector("#error");
const meta = document.querySelector("#meta");
const newChat = document.querySelector("#new-chat");
const faultRow = document.querySelector("#fault-row");
const faultButton = document.querySelector("#fault");

let sessionId = sessionStorage.getItem("session_id") || "";

function setMeta(requestId) {
  const sessionText = sessionId || "new";
  meta.textContent = requestId
    ? `Session: ${sessionText} · Request: ${requestId}`
    : `Session: ${sessionText}`;
}

function showError(text) {
  errorEl.hidden = !text;
  errorEl.textContent = text || "";
}

function addMessage(role, text, toolName) {
  const item = document.createElement("article");
  item.className = `bubble ${role}`;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = role === "user" ? "You" : "Assistant";
  const body = document.createElement("div");
  body.textContent = text;
  item.append(who, body);
  if (toolName) {
    const note = document.createElement("span");
    note.className = "tool-note";
    note.textContent = `Tool used: ${toolName}`;
    item.append(note);
  }
  messagesEl.append(item);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setBusy(busy) {
  send.disabled = busy;
  input.disabled = busy;
  faultButton.disabled = busy;
  loading.hidden = !busy;
}

async function sendMessage(message, headers) {
  showError("");
  addMessage("user", message);
  setBusy(true);
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(headers || {}) },
      body: JSON.stringify({ message, session_id: sessionId || null }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = payload.detail;
      const text = typeof detail === "string" ? detail : "The assistant could not complete this request.";
      showError(text);
      return;
    }
    sessionId = payload.session_id;
    sessionStorage.setItem("session_id", sessionId);
    setMeta(payload.request_id);
    addMessage("assistant", payload.answer, payload.tool_used ? payload.tool_name : "");
  } catch (_err) {
    showError("The server could not be reached.");
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) {
    return;
  }
  input.value = "";
  sendMessage(message);
});

newChat.addEventListener("click", () => {
  sessionId = "";
  sessionStorage.removeItem("session_id");
  messagesEl.replaceChildren();
  setMeta("");
  showError("");
  input.focus();
});

faultButton.addEventListener("click", () => {
  sendMessage("Simulate a model failure.", { "X-Demo-Fault": "model" });
});

fetch("/health")
  .then((response) => response.json())
  .then((payload) => {
    if (payload.demo_faults_enabled) {
      faultRow.hidden = false;
    }
  })
  .catch(() => {});

setMeta("");
input.focus();
