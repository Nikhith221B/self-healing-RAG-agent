# Screenshots

Add 1–2 UI screenshots here for the GitHub README:

1. **Upload + ask** — `/app/` with a question and an accepted answer (show sources + critic).
2. **Refusal** — unanswerable question showing the standard refusal message.

Capture after running:

```powershell
python ingest.py
.\scripts\run_dev.ps1
```

Open http://127.0.0.1:8000/app/

Suggested filenames:

- `ui-accepted.png`
- `ui-refused.png`

Then link them in the root `README.md`:

```markdown
![Accepted answer](screenshots/ui-accepted.png)
```
