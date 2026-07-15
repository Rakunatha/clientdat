# ClientDat

A working prototype of a client intake + AI case analysis tool for family law
practices. One Flask app (`app.py`) — routes, database, AI calls, and every
page's HTML/CSS/JS all live in that single file.

**Flow:** lawyer generates a branded intake page → shares it like a business
card → client fills in their situation + uploads documents → client books an
appointment time → lawyer opens a private dashboard and sees an AI-generated
summary, urgency flag, missing documents, and suggested first-meeting
questions for every case.

---

## 1. Run it locally

```bash
cd clientdat
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000**. That's it — no signup, no external service
required. It works immediately using an offline fallback for the "AI
analysis" step (see below), so you can demo the whole flow before wiring up
a real model.

---

## 2. Add a real AI model (free tier)

The app calls any OpenAI-compatible `/chat/completions` endpoint. The
easiest free option is **Groq** (fast, generous free tier, no credit card
required to start):

1. Go to **https://console.groq.com** and sign up.
2. Go to **API Keys** → **Create API Key**. Copy it.
3. Set it as an environment variable before starting the app:

   ```bash
   export GROQ_API_KEY="gsk_...your key..."
   python app.py
   ```

   On Windows (PowerShell): `$env:GROQ_API_KEY="gsk_..."`

4. Submit a new case through a lawyer's public page — you'll now get a real
   AI-generated summary, urgency rating, key issues, missing documents, and
   suggested questions instead of the offline fallback. The dashboard shows
   a banner telling you which mode you're in.

The default model is `llama-3.1-8b-instant`. You can swap it via env vars
without touching code:

```bash
export AI_MODEL="llama-3.3-70b-versatile"      # a larger free Groq model
```

**Using a different provider instead of Groq:** any OpenAI-compatible
provider works (e.g. OpenRouter, which also has free models). Just set:

```bash
export AI_API_URL="https://openrouter.ai/api/v1/chat/completions"
export AI_API_KEY="sk-or-...your key..."
export AI_MODEL="meta-llama/llama-3.1-8b-instruct:free"
```

If no key is set, or the API call fails for any reason (rate limit, network
hiccup, bad key), ClientDat automatically falls back to a deterministic
offline heuristic so the demo never breaks mid-flow.

---

## 3. Deploy to Render

1. Push this folder to a GitHub repo (make sure `app.py`, `requirements.txt`
   are at the root, or note the subfolder in Render's settings).
2. In Render, click **New +** → **Web Service**, connect the repo.
3. Settings:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app`
4. Under **Environment**, add:
   - `GROQ_API_KEY` = your key (optional, but needed for real AI output)
   - `SECRET_KEY` = any random string
5. Click **Create Web Service**. Render will give you a URL like
   `https://clientdat.onrender.com` — that's your live landing page where
   lawyers generate their pages.

**Note on storage (important for a demo, not a limitation of the code):**
Render's free web service disk is ephemeral — it resets on every redeploy
or restart, which means the SQLite database and uploaded files reset too.
That's fine for a live demo. For anything persistent, add a Render **Disk**
(Render dashboard → your service → Disks) mounted at, say, `/data`, and set:

```bash
export DB_PATH=/data/clientdat.db
export UPLOAD_DIR=/data/uploads
```

*(Both are read from environment variables at startup in `app.py` — set
them to paths on your mounted disk and everything else works unchanged.)*

---

## What's real vs. what's simplified in this prototype

- **Real:** lawyer page generation with unique shareable slugs, a private
  tokenized dashboard link (not guessable, not a login system), file
  uploads saved to disk, full case + appointment persistence in SQLite, and
  live AI analysis via any OpenAI-compatible API.
- **Simplified for the demo:** no lawyer login/password (the dashboard link
  itself is the credential — swap in real auth before any real client data
  touches this), no email/SMS notifications on new bookings, no document
  content parsing (the AI sees filenames + the client's written
  description, not the file contents), no encryption at rest.

---

## File structure

```
clientdat/
  app.py             # everything: routes, DB, AI calls, HTML templates
  requirements.txt
  README.md
  uploads/            # created automatically, gitignored
  clientdat.db         # created automatically, gitignored
```
