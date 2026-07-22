# ClientDat

A client intake + AI case analysis tool for any legal practice. One Flask
app (`app.py`) — routes, database, encryption, AI calls, and every page's
HTML/CSS all live in that single file.

**Flow:** lawyer generates a branded intake page (their own colour + light/dark
theme) → shares it like a business card → client describes their situation,
uploads documents, and books an appointment time → lawyer opens a private
dashboard and sees an AI-generated summary, urgency flag, missing documents,
and suggested first-meeting questions for every case.

Everything sensitive — case descriptions, uploaded files, AI results, and
appointment notes — is encrypted before it's written to disk or the database,
and only decrypted in memory when a lawyer opens their own dashboard.

---

## 1. Run it locally

```bash
cd clientdat
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000**. It works immediately using an offline
fallback for the "AI analysis" step, so you can try the whole flow before
wiring up a real model.

---

## 2. Set an encryption key

```bash
export ENCRYPTION_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
```

Without this set, ClientDat generates a temporary key each time it starts —
fine for a quick local try, but anything encrypted with it becomes unreadable
after a restart. Set `ENCRYPTION_KEY` once and keep it somewhere safe; losing
it means losing access to every stored case.

---

## 3. Add a real AI model (free tier)

The app calls any OpenAI-compatible `/chat/completions` endpoint. The
easiest free option is **Groq**:

1. Go to **https://console.groq.com** and sign up.
2. **API Keys** → **Create API Key**. Copy it.
3. Set it as an environment variable before starting the app:

   ```bash
   export GROQ_API_KEY="gsk_...your key..."
   python app.py
   ```

4. Submit a new case through a lawyer's public page — you'll get a real
   AI-generated summary, urgency rating, key issues, missing documents, and
   suggested questions instead of the offline fallback.

Swap the model without touching code:

```bash
export AI_MODEL="llama-3.3-70b-versatile"
```

**Using a different provider instead of Groq:** any OpenAI-compatible
provider works (e.g. OpenRouter, which also has free models):

```bash
export AI_API_URL="https://openrouter.ai/api/v1/chat/completions"
export AI_API_KEY="sk-or-...your key..."
export AI_MODEL="meta-llama/llama-3.1-8b-instruct:free"
```

If no key is set, or the API call fails for any reason, ClientDat falls back
to a deterministic offline heuristic so the flow never breaks.

---

## 4. Deploy to Render

1. Push this folder to a GitHub repo.
2. In Render: **New +** → **Web Service**, connect the repo.
3. Settings:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app`
4. Under **Environment**, add:
   - `ENCRYPTION_KEY` — Render generates one automatically from `render.yaml`
   - `GROQ_API_KEY` — optional, needed for real AI output
   - `SECRET_KEY` — any random string
5. Click **Create Web Service**.

**Note on storage:** Render's free web service disk is ephemeral — it resets
on redeploy, which resets the SQLite database and uploaded files too. For
anything persistent, add a Render **Disk** and set:

```bash
export DB_PATH=/data/clientdat.db
export UPLOAD_DIR=/data/uploads
```

---

## What's real vs. what's simplified in this prototype

- **Real:** branded intake page generation with a colour + theme picker, a
  private tokenized dashboard link, encryption at rest for case
  descriptions, uploaded files, AI results, and appointment notes, and live
  AI analysis via any OpenAI-compatible API.
- **Simplified:** no lawyer login/password (the dashboard link itself is the
  credential — add real auth before real client data touches this), no
  email/SMS notifications on new bookings, no document content parsing (the
  AI sees filenames + the client's written description, not file contents).

---

## File structure

```
clientdat/
  app.py             # everything: routes, DB, encryption, AI calls, templates
  requirements.txt
  README.md
  render.yaml
  uploads/            # created automatically, gitignored, files encrypted
  clientdat.db         # created automatically, gitignored, fields encrypted
```
