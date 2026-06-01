const API = "";

const fileInput = document.getElementById("file-input");
const ingestBtn = document.getElementById("ingest-btn");
const ingestStatus = document.getElementById("ingest-status");
const questionInput = document.getElementById("question-input");
const topKInput = document.getElementById("top-k");
const maxRetriesInput = document.getElementById("max-retries");
const askBtn = document.getElementById("ask-btn");
const askStatus = document.getElementById("ask-status");
const answerPanel = document.getElementById("answer-panel");
const statusBadge = document.getElementById("status-badge");
const retryMeta = document.getElementById("retry-meta");
const answerText = document.getElementById("answer-text");
const criticJson = document.getElementById("critic-json");
const queryHistory = document.getElementById("query-history");
const sourcesEl = document.getElementById("sources");
const runIdEl = document.getElementById("run-id");
const feedbackHelpful = document.getElementById("feedback-helpful");
const feedbackBad = document.getElementById("feedback-bad");
const feedbackStatus = document.getElementById("feedback-status");

let lastRunId = null;

function setStatus(el, message, kind = "") {
  el.textContent = message;
  el.className = `status ${kind}`.trim();
}

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(String(value), 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function formatApiError(data, fallback) {
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    return data.detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
  }
  return fallback;
}

ingestBtn.addEventListener("click", async () => {
  const files = fileInput.files;
  if (!files.length) {
    setStatus(ingestStatus, "Select at least one file.", "error");
    return;
  }

  ingestBtn.disabled = true;
  setStatus(ingestStatus, "Uploading and indexing…");

  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }

  try {
    const res = await fetch(`${API}/ingest`, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(formatApiError(data, res.statusText));
    }
    setStatus(
      ingestStatus,
      `Indexed ${data.processed_files} file(s), ${data.chunks_created} chunk(s).`,
      "ok"
    );
  } catch (err) {
    setStatus(ingestStatus, err.message, "error");
  } finally {
    ingestBtn.disabled = false;
  }
});

askBtn.addEventListener("click", async () => {
  const question = questionInput.value.trim();
  if (!question) {
    setStatus(askStatus, "Enter a question.", "error");
    return;
  }

  askBtn.disabled = true;
  answerPanel.hidden = true;
  setStatus(askStatus, "Running self-healing RAG…");

  try {
    const res = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: Math.max(1, parsePositiveInt(topKInput.value, 5)),
        max_retries: parsePositiveInt(maxRetriesInput.value, 2),
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(formatApiError(data, res.statusText));
    }

    answerPanel.hidden = false;
    statusBadge.textContent = data.final_status;
    statusBadge.className = `badge ${data.final_status}`;
    retryMeta.textContent = `Retries: ${data.retry_count}`;
    answerText.textContent = data.answer;
    criticJson.textContent = JSON.stringify(data.critic_feedback, null, 2);

    queryHistory.innerHTML = "";
    for (const q of data.query_history || []) {
      const li = document.createElement("li");
      li.textContent = q;
      queryHistory.appendChild(li);
    }

    sourcesEl.innerHTML = "";
    for (const src of data.sources || []) {
      const card = document.createElement("div");
      card.className = "source-card";
      card.innerHTML = `<strong>${src.source}</strong> · ${src.chunk_id}<br>${escapeHtml(src.text)}`;
      sourcesEl.appendChild(card);
    }

    lastRunId = data.run_id;
    runIdEl.textContent = data.run_id
      ? `Run ID: ${data.run_id}${data.bandit_arm ? ` · bandit: ${data.bandit_arm} (top_k=${data.top_k_used})` : ""}`
      : "";
    setStatus(feedbackStatus, "");
    setStatus(askStatus, "Done.", "ok");
  } catch (err) {
    setStatus(askStatus, err.message, "error");
  } finally {
    askBtn.disabled = false;
  }
});

async function sendFeedback(rating) {
  if (!lastRunId) {
    setStatus(feedbackStatus, "Ask a question first.", "error");
    return;
  }
  try {
    const res = await fetch(`${API}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: lastRunId, rating }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(formatApiError(data, res.statusText));
    setStatus(feedbackStatus, data.message, "ok");
  } catch (err) {
    setStatus(feedbackStatus, err.message, "error");
  }
}

feedbackHelpful.addEventListener("click", () => sendFeedback("helpful"));
feedbackBad.addEventListener("click", () => sendFeedback("not_helpful"));

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
