from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


class SmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_app_ui(self) -> None:
        response = self.client.get("/app/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Self-Healing RAG", response.text)

    def test_root_redirects_to_app(self) -> None:
        response = self.client.get("/", follow_redirects=False)
        self.assertIn(response.status_code, {307, 302})
        self.assertIn("/app/", response.headers.get("location", ""))

    def test_ingest_no_files(self) -> None:
        response = self.client.post("/ingest")
        self.assertEqual(response.status_code, 400)

    def test_ingest_unsupported_type(self) -> None:
        response = self.client.post(
            "/ingest",
            files=[("files", ("bad.exe", b"data", "application/octet-stream"))],
        )
        self.assertEqual(response.status_code, 400)

    def test_ask_empty_question(self) -> None:
        response = self.client.post("/ask", json={"question": "   "})
        self.assertEqual(response.status_code, 400)

    @patch("api.routes.run_self_healing_rag")
    def test_ask_mocked_returns_run_id(self, mock_run) -> None:
        mock_run.return_value = {
            "answer": "18 paid leaves.",
            "final_status": "accepted",
            "retry_count": 0,
            "original_query": "How many leaves?",
            "query_history": ["How many leaves?"],
            "critic_feedback": {
                "grounded": True,
                "score": 0.9,
                "decision": "accept",
                "supported_chunks": [],
                "unsupported_claims": [],
                "reason": "ok",
            },
            "critic_history": [],
            "sources": [],
        }
        response = self.client.post(
            "/ask",
            json={"question": "How many leaves?", "top_k": 5, "max_retries": 2},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("run_id", body)
        self.assertEqual(body["final_status"], "accepted")

    def test_feedback_unknown_run(self) -> None:
        response = self.client.post(
            "/feedback",
            json={"run_id": "00000000-0000-0000-0000-000000000000", "rating": "helpful"},
        )
        self.assertEqual(response.status_code, 404)

    @patch("api.routes.run_self_healing_rag")
    def test_feedback_after_ask(self, mock_run) -> None:
        mock_run.return_value = {
            "answer": "ok",
            "final_status": "accepted",
            "retry_count": 0,
            "original_query": "q",
            "query_history": ["q"],
            "critic_feedback": {"decision": "accept", "score": 1.0, "grounded": True},
            "critic_history": [],
            "sources": [],
        }
        ask_response = self.client.post("/ask", json={"question": "test question"})
        run_id = ask_response.json()["run_id"]
        feedback_response = self.client.post(
            "/feedback",
            json={"run_id": run_id, "rating": "helpful"},
        )
        self.assertEqual(feedback_response.status_code, 200)

    def test_gitignore_blocks_env(self) -> None:
        gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
        self.assertTrue(gitignore.exists())
        content = gitignore.read_text(encoding="utf-8")
        self.assertIn(".env", content)
        self.assertIn("logs/", content)


if __name__ == "__main__":
    unittest.main()
