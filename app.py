"""
ClientDat — client intake + AI case analysis for family law practices.

Single-file Flask app: routes, SQLite persistence, AI analysis, and all
HTML/CSS/JS templates (inlined as strings) live in this one module by design,
so the whole product is one code base to read, deploy, and hack on.

Run locally:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000

See README.md for the free AI API key setup and Render deploy steps.
"""

import os
import json
import sqlite3
import uuid
import secrets
from datetime import datetime
from flask import Flask, request, redirect, url_for, g, render_template_string, abort
import urllib.request
import urllib.error

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "clientdat.db"))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Free-tier friendly default. Groq's OpenAI-compatible endpoint, no cost to start.
# Swap MODEL / API_URL / API_KEY env vars to point at any OpenAI-compatible provider.
AI_API_KEY = os.environ.get("GROQ_API_KEY", "")
AI_API_URL = os.environ.get("AI_API_URL", "https://api.groq.com/openai/v1/chat/completions")
AI_MODEL = os.environ.get("AI_MODEL", "llama-3.1-8b-instant")

ALLOWED_EXT = {"pdf", "doc", "docx", "txt", "png", "jpg", "jpeg"}
CASE_TYPES = ["Divorce", "Child Custody", "Child Support", "Property Division",
              "Adoption", "Restraining Order", "Prenup / Postnup", "Other"]

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")


# --------------------------------------------------------------------------
# DB helpers
# --------------------------------------------------------------------------

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS lawyers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE NOT NULL,
            token TEXT NOT NULL,
            name TEXT NOT NULL,
            firm_name TEXT NOT NULL,
            tagline TEXT,
            email TEXT,
            accent TEXT DEFAULT 'FF5500',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lawyer_id INTEGER NOT NULL,
            client_name TEXT NOT NULL,
            client_email TEXT,
            client_phone TEXT,
            case_type TEXT,
            pain_points TEXT,
            files TEXT,
            ai_status TEXT DEFAULT 'pending',
            ai_result TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (lawyer_id) REFERENCES lawyers(id)
        );

        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submission_id INTEGER NOT NULL,
            pref_date TEXT,
            pref_time TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (submission_id) REFERENCES submissions(id)
        );
        """
    )
    db.commit()
    db.close()


# --------------------------------------------------------------------------
# AI analysis
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a family-law intake analyst helping a lawyer triage a new \
client submission before the first meeting. You are not giving legal advice, you are \
summarizing and flagging what matters for the lawyer's prep.

Return ONLY valid JSON, no markdown fences, no commentary, matching exactly this shape:
{
  "summary": "2-3 sentence plain-language summary of the situation",
  "case_type_guess": "one of: Divorce, Child Custody, Child Support, Property Division, Adoption, Restraining Order, Prenup/Postnup, Other",
  "urgency": "Low, Medium, or High",
  "urgency_reason": "one short sentence",
  "key_issues": ["short issue", "short issue", "..."],
  "missing_documents": ["document the lawyer should ask for", "..."],
  "suggested_questions": ["question to ask at the first meeting", "..."]
}
Keep every list to at most 5 short items. Base everything only on what the client wrote."""


def run_ai_analysis(case_type, pain_points, filenames):
    """Call an OpenAI-compatible chat completion endpoint. Falls back to a
    clearly-labeled offline heuristic result if no API key is configured or
    the call fails, so the demo always works end to end."""

    file_list = ", ".join(filenames) if filenames else "none uploaded"
    user_content = (
        f"Client-selected case type: {case_type}\n"
        f"Documents uploaded: {file_list}\n"
        f"Client's own description of their situation:\n{pain_points}"
    )

    if not AI_API_KEY:
        return _fallback_analysis(case_type, pain_points, filenames, reason="no_api_key")

    payload = {
        "model": AI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "max_tokens": 700,
    }
    req = urllib.request.Request(
        AI_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AI_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip()
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
        result = json.loads(text)
        result["_source"] = "ai"
        return result
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError, TimeoutError) as e:
        fb = _fallback_analysis(case_type, pain_points, filenames, reason="ai_error")
        fb["_error"] = str(e)
        return fb


def _fallback_analysis(case_type, pain_points, filenames, reason):
    """Deterministic offline stand-in so the product works before an API key
    is added, and stays working if the AI call ever fails during a demo."""
    text = (pain_points or "").lower()
    urgent_words = ["urgent", "emergency", "abuse", "unsafe", "threat", "danger", "police", "restraining"]
    urgency = "High" if any(w in text for w in urgent_words) else ("Medium" if len(text) > 200 else "Low")
    return {
        "summary": (pain_points[:220] + ("..." if len(pain_points) > 220 else "")) if pain_points else
                   "Client did not provide a written description.",
        "case_type_guess": case_type or "Other",
        "urgency": urgency,
        "urgency_reason": "Flagged by keyword scan — connect an AI key for a real assessment."
                           if reason == "no_api_key" else "AI service unavailable — fallback heuristic used.",
        "key_issues": ["Add an AI API key to generate real case insights — see README.md"],
        "missing_documents": ["Photo ID", "Any prior court orders", "Financial statements (if relevant)"],
        "suggested_questions": ["Walk me through the timeline in your own words.",
                                 "Are there any safety concerns for you or your children?"],
        "_source": "offline_fallback",
    }


# --------------------------------------------------------------------------
# Small utils
# --------------------------------------------------------------------------

def slugify(name):
    base = "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")
    while "--" in base:
        base = base.replace("--", "-")
    return base or "lawyer"


def unique_slug(name):
    db = get_db()
    base = slugify(name)
    slug = base
    n = 1
    while db.execute("SELECT 1 FROM lawyers WHERE slug=?", (slug,)).fetchone():
        n += 1
        slug = f"{base}-{n}"
    return slug


def render(tpl, **kw):
    """Safe templating: plain .replace() per [[placeholder]], so CSS/JS curly
    braces in the surrounding HTML never collide with str.format()."""
    out = tpl
    for k, v in kw.items():
        out = out.replace(f"[[{k}]]", str(v))
    return out


def get_lawyer_by_slug(slug):
    db = get_db()
    row = db.execute("SELECT * FROM lawyers WHERE slug=?", (slug,)).fetchone()
    if not row:
        abort(404)
    return row


# --------------------------------------------------------------------------
# Shared CSS — "instrument panel" system. Putty / ink / signal-orange.
# --------------------------------------------------------------------------

BASE_CSS = """
:root{
  --putty:#ECEAE2; --surface:#F8F7F3; --ink:#16140F; --ink-60:#615D52;
  --line:#CFC9B8; --accent:#FF5500; --accent-ink:#FFFFFF; --ok:#2F6B3A;
  --radius:3px; --mono:'Space Mono',ui-monospace,'SF Mono',monospace;
  --sans:'Inter',system-ui,-apple-system,sans-serif;
}
*{box-sizing:border-box;}
body{margin:0;background:var(--putty);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased;}
h1,h2,h3{font-family:var(--mono);font-weight:700;letter-spacing:-0.01em;margin:0;}
.label{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-60);}
.wrap{max-width:960px;margin:0 auto;padding:0 20px;}
.tag{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono);font-size:11px;
  text-transform:uppercase;letter-spacing:.08em;border:1px solid var(--line);padding:3px 8px;border-radius:var(--radius);background:var(--surface);}
.dot{width:6px;height:6px;border-radius:50%;background:var(--accent);display:inline-block;}
.topbar{border-bottom:1px solid var(--line);padding:14px 0;}
.topbar .wrap{display:flex;align-items:center;justify-content:space-between;}
.brandmark{font-family:var(--mono);font-weight:700;font-size:15px;letter-spacing:-.02em;}
.brandmark b{color:var(--accent);}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:22px;}
.module-num{font-family:var(--mono);font-size:11px;color:var(--accent);font-weight:700;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
@media(max-width:640px){.grid2{grid-template-columns:1fr;}}
.field{margin-bottom:16px;}
.field label{display:block;margin-bottom:6px;}
input[type=text],input[type=email],input[type=tel],input[type=date],input[type=time],select,textarea{
  width:100%;font-family:var(--sans);font-size:14px;padding:11px 12px;border:1px solid var(--line);
  border-radius:var(--radius);background:#fff;color:var(--ink);}
textarea{resize:vertical;min-height:110px;}
input:focus,select:focus,textarea:focus{outline:2px solid var(--accent);outline-offset:1px;}
.file-drop{border:1px dashed var(--line);border-radius:var(--radius);padding:18px;text-align:center;
  font-family:var(--mono);font-size:12px;color:var(--ink-60);background:#fff;cursor:pointer;}
.file-drop.has-files{border-color:var(--accent);color:var(--ink);}
.btn{font-family:var(--mono);text-transform:uppercase;letter-spacing:.06em;font-size:13px;font-weight:700;
  border:1px solid var(--ink);background:var(--ink);color:#fff;padding:13px 20px;border-radius:var(--radius);
  cursor:pointer;box-shadow:3px 3px 0 var(--line);transition:transform .08s, box-shadow .08s;}
.btn:hover{transform:translate(1px,1px);box-shadow:2px 2px 0 var(--line);}
.btn:active{transform:translate(3px,3px);box-shadow:0 0 0 var(--line);}
.btn.accent{background:var(--accent);border-color:var(--accent);box-shadow:3px 3px 0 #C94100;}
.btn.accent:hover{box-shadow:2px 2px 0 #C94100;}
.btn.ghost{background:transparent;color:var(--ink);box-shadow:none;}
.btn:disabled{opacity:.5;cursor:not-allowed;}
.chain{display:flex;align-items:center;gap:0;margin:26px 0;font-family:var(--mono);font-size:11px;
  text-transform:uppercase;letter-spacing:.06em;flex-wrap:wrap;}
.chain .step{border:1px solid var(--line);background:var(--surface);padding:8px 12px;border-radius:var(--radius);}
.chain .step.active{border-color:var(--accent);color:var(--accent);}
.chain .arrow{padding:0 8px;color:var(--ink-60);}
footer{border-top:1px solid var(--line);margin-top:60px;padding:18px 0;font-family:var(--mono);
  font-size:11px;color:var(--ink-60);text-transform:uppercase;letter-spacing:.06em;}
a{color:var(--ink);}
.small{font-size:12px;color:var(--ink-60);}
.err{color:#B3261E;font-size:13px;font-family:var(--mono);}
.copybox{display:flex;gap:8px;align-items:center;border:1px solid var(--line);background:#fff;
  border-radius:var(--radius);padding:10px 12px;font-family:var(--mono);font-size:13px;overflow-x:auto;white-space:nowrap;}
.copybox button{margin-left:auto;flex-shrink:0;}
.badge{display:inline-block;font-family:var(--mono);font-size:11px;text-transform:uppercase;
  letter-spacing:.06em;padding:3px 8px;border-radius:var(--radius);border:1px solid;}
.badge.High{color:#B3261E;border-color:#B3261E;background:#FCEBEA;}
.badge.Medium{color:#8A5A00;border-color:#8A5A00;background:#FBF1DE;}
.badge.Low{color:var(--ok);border-color:var(--ok);background:#EAF3EB;}
"""

HEAD = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{css}</style>"""


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------

LANDING_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ClientDat — intake &amp; case analysis for family law</title>""" + HEAD.replace("{css}", BASE_CSS) + """
</head><body>
<div class="topbar"><div class="wrap">
  <div class="brandmark">Client<b>Dat</b></div>
  <span class="tag"><span class="dot"></span> v0.1 prototype</span>
</div></div>

<div class="wrap" style="padding-top:56px;padding-bottom:40px;">
  <span class="module-num">MOD 00 — SETUP</span>
  <h1 style="font-size:38px;margin-top:10px;max-width:640px;">Intake &amp; case analysis, built for family law.</h1>
  <p style="max-width:520px;color:var(--ink-60);margin-top:14px;">
    Generate a branded intake page in one minute, share it like a business card,
    and let ClientDat pull the signal out of every client submission before you
    walk into the first meeting.
  </p>

  <div class="chain">
    <div class="step active">01 · Set up page</div><div class="arrow">→</div>
    <div class="step">02 · Client submits</div><div class="arrow">→</div>
    <div class="step">03 · AI analysis</div><div class="arrow">→</div>
    <div class="step">04 · Appointment booked</div>
  </div>

  <div class="card" style="max-width:560px;margin-top:10px;">
    <span class="module-num">MOD 01 — YOUR PAGE</span>
    <h2 style="font-size:18px;margin-top:8px;margin-bottom:18px;">Create your intake page</h2>
    <form method="post" action="/api/lawyers">
      <div class="grid2">
        <div class="field"><label>Your name</label><input type="text" name="name" required placeholder="Dana Whitfield"></div>
        <div class="field"><label>Firm name</label><input type="text" name="firm_name" required placeholder="Whitfield Family Law"></div>
      </div>
      <div class="field"><label>Tagline (shown on your page)</label>
        <input type="text" name="tagline" placeholder="Calm, direct counsel for families in transition."></div>
      <div class="field"><label>Email for booking notifications</label>
        <input type="email" name="email" required placeholder="dana@whitfieldlaw.com"></div>
      <button class="btn accent" type="submit">Generate my page →</button>
    </form>
  </div>

  <p class="small" style="max-width:560px;margin-top:18px;">
    This is a working prototype — data is stored in a local database on this server,
    file uploads are kept on disk, and case analysis calls an AI model if one is configured
    (see README.md). No payment, no account system, nothing external required to try it.
  </p>
</div>
<footer><div class="wrap">ClientDat — demo build</div></footer>
</body></html>"""


LAWYER_CREATED_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Your page is live — ClientDat</title>""" + HEAD.replace("{css}", BASE_CSS) + """
</head><body>
<div class="topbar"><div class="wrap"><div class="brandmark">Client<b>Dat</b></div></div></div>
<div class="wrap" style="padding-top:56px;max-width:640px;">
  <span class="module-num">MOD 01 — READY</span>
  <h1 style="font-size:30px;margin-top:10px;">Your page is live.</h1>
  <p style="color:var(--ink-60);">Share the first link with clients. Keep the second one for yourself — it opens your private case dashboard.</p>

  <div class="field" style="margin-top:26px;">
    <label>Client-facing intake page (share this)</label>
    <div class="copybox"><span id="pub">[[public_url]]</span>
      <button class="btn ghost" type="button" onclick="copyIt('pub')">Copy</button></div>
  </div>
  <div class="field">
    <label>Your private dashboard (bookmark this — it is not guessable)</label>
    <div class="copybox"><span id="dash">[[dash_url]]</span>
      <button class="btn ghost" type="button" onclick="copyIt('dash')">Copy</button></div>
  </div>

  <a href="[[dash_url]]"><button class="btn accent" style="margin-top:8px;">Open my dashboard →</button></a>
</div>
<script>
function copyIt(id){navigator.clipboard.writeText(document.getElementById(id).textContent);}
</script>
<footer><div class="wrap">ClientDat — demo build</div></footer>
</body></html>"""


PUBLIC_PAGE_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{firm_name} — Case Intake</title>""" + HEAD.replace("{css}", BASE_CSS) + """
</head><body>
<div class="topbar"><div class="wrap">
  <div class="brandmark">[[firm_name]]</div>
  <span class="tag">via Client<b style="color:var(--accent)">Dat</b></span>
</div></div>

<div class="wrap" style="padding-top:44px;padding-bottom:50px;">
  <span class="module-num">MOD 02 — INTAKE</span>
  <h1 style="font-size:28px;margin-top:8px;">[[name]]</h1>
  <p style="color:var(--ink-60);max-width:520px;">[[tagline]]</p>

  <div class="chain">
    <div class="step active">01 · Tell us what's going on</div><div class="arrow">→</div>
    <div class="step">02 · We review</div><div class="arrow">→</div>
    <div class="step">03 · Pick a time to talk</div>
  </div>

  <div class="card" style="max-width:640px;">
    <span class="module-num">STEP 1 / 2</span>
    <h2 style="font-size:18px;margin-top:8px;margin-bottom:18px;">Your situation</h2>
    <form method="post" action="/submit/[[slug]]" enctype="multipart/form-data" id="intakeForm">
      <div class="grid2">
        <div class="field"><label>Your name</label><input type="text" name="client_name" required></div>
        <div class="field"><label>Email</label><input type="email" name="client_email" required></div>
      </div>
      <div class="grid2">
        <div class="field"><label>Phone</label><input type="tel" name="client_phone"></div>
        <div class="field"><label>Case type</label>
          <select name="case_type">[[case_type_options]]</select>
        </div>
      </div>
      <div class="field">
        <label>What's happening? Be as detailed as you're comfortable with.</label>
        <textarea name="pain_points" required placeholder="Timeline, main concerns, anything urgent..."></textarea>
      </div>
      <div class="field">
        <label>Relevant documents (optional)</label>
        <label class="file-drop" id="dropLabel">
          <input type="file" name="files" multiple style="display:none" id="fileInput">
          <span id="dropText">Click to attach PDFs, images, or docs</span>
        </label>
      </div>
      <button class="btn accent" type="submit" id="submitBtn">Submit &amp; continue →</button>
      <p class="small" style="margin-top:10px;">Sent directly and privately to [[name]]. Not encrypted in this prototype — do not upload highly sensitive originals in a live demo.</p>
    </form>
  </div>
</div>

<script>
const fi = document.getElementById('fileInput');
const dl = document.getElementById('dropLabel');
fi.addEventListener('change', () => {
  if(fi.files.length){
    dl.classList.add('has-files');
    document.getElementById('dropText').textContent = fi.files.length + ' file(s) selected';
  }
});
document.getElementById('intakeForm').addEventListener('submit', () => {
  document.getElementById('submitBtn').disabled = true;
  document.getElementById('submitBtn').textContent = 'Submitting…';
});
</script>
<footer><div class="wrap">Powered by Client<b style="color:var(--accent)">Dat</b></div></footer>
</body></html>"""


BOOKING_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Book a time — [[firm_name]]</title>""" + HEAD.replace("{css}", BASE_CSS) + """
</head><body>
<div class="topbar"><div class="wrap"><div class="brandmark">[[firm_name]]</div></div></div>
<div class="wrap" style="padding-top:44px;padding-bottom:50px;">
  <span class="module-num">MOD 02 — INTAKE</span>
  <h1 style="font-size:26px;margin-top:8px;">Thanks, [[client_name]]. One more step.</h1>
  <div class="chain">
    <div class="step">01 · Tell us what's going on</div><div class="arrow">→</div>
    <div class="step active">02 · We received it</div><div class="arrow">→</div>
    <div class="step active">03 · Pick a time to talk</div>
  </div>

  <div class="card" style="max-width:520px;">
    <span class="module-num">STEP 2 / 2</span>
    <h2 style="font-size:18px;margin-top:8px;margin-bottom:18px;">Suggest a time for your call with [[name]]</h2>
    <form method="post" action="/book/[[submission_id]]">
      <div class="grid2">
        <div class="field"><label>Preferred date</label><input type="date" name="pref_date" required></div>
        <div class="field"><label>Preferred time</label><input type="time" name="pref_time" required></div>
      </div>
      <div class="field"><label>Anything else? (optional)</label><textarea name="notes" placeholder="e.g. I can only do mornings"></textarea></div>
      <button class="btn accent" type="submit">Request appointment →</button>
    </form>
  </div>
</div>
<footer><div class="wrap">Powered by Client<b style="color:var(--accent)">Dat</b></div></footer>
</body></html>"""


CONFIRM_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Request sent — [[firm_name]]</title>""" + HEAD.replace("{css}", BASE_CSS) + """
</head><body>
<div class="topbar"><div class="wrap"><div class="brandmark">[[firm_name]]</div></div></div>
<div class="wrap" style="padding-top:44px;">
  <span class="module-num">MOD 02 — COMPLETE</span>
  <h1 style="font-size:26px;margin-top:8px;">You're all set.</h1>
  <div class="chain">
    <div class="step">01 · Tell us what's going on</div><div class="arrow">→</div>
    <div class="step">02 · We received it</div><div class="arrow">→</div>
    <div class="step active">03 · Time requested</div>
  </div>
  <div class="card" style="max-width:480px;">
    <p>[[name]] will confirm your appointment on <b>[[pref_date]] at [[pref_time]]</b>. You'll be contacted at the email you provided.</p>
  </div>
</div>
<footer><div class="wrap">Powered by Client<b style="color:var(--accent)">Dat</b></div></footer>
</body></html>"""


DASHBOARD_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard — [[firm_name]] — ClientDat</title>""" + HEAD.replace("{css}", BASE_CSS) + """
</head><body>
<div class="topbar"><div class="wrap">
  <div class="brandmark">Client<b>Dat</b></div>
  <span class="tag"><span class="dot"></span> [[firm_name]]</span>
</div></div>

<div class="wrap" style="padding-top:44px;padding-bottom:60px;">
  <span class="module-num">MOD 03 — DASHBOARD</span>
  <h1 style="font-size:28px;margin-top:8px;">Cases</h1>
  <p class="small">Your public intake page: <a href="/l/[[slug]]">/l/[[slug]]</a></p>

  [[ai_notice]]

  [[rows]]
</div>
<footer><div class="wrap">ClientDat — demo build</div></footer>
</body></html>"""


CASE_ROW_HTML = """
<div class="card" style="margin-top:16px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;">
    <div>
      <span class="module-num">SUB-[[id]]</span>
      <h3 style="font-size:17px;margin-top:6px;">[[client_name]] <span class="small">· [[case_type]]</span></h3>
      <p class="small">[[client_email]] [[phone_part]] · submitted [[created_at]]</p>
    </div>
    <span class="badge [[urgency]]">[[urgency]] urgency</span>
  </div>

  <div class="grid2" style="margin-top:16px;">
    <div>
      <span class="label">AI summary [[source_tag]]</span>
      <p style="margin-top:6px;">[[summary]]</p>
      <span class="label">Key issues</span>
      <ul style="margin:6px 0 0 18px;padding:0;">[[key_issues]]</ul>
    </div>
    <div>
      <span class="label">Missing documents</span>
      <ul style="margin:6px 0 14px 18px;padding:0;">[[missing_docs]]</ul>
      <span class="label">Suggested first-meeting questions</span>
      <ul style="margin:6px 0 0 18px;padding:0;">[[questions]]</ul>
    </div>
  </div>

  <div style="margin-top:16px;border-top:1px solid var(--line);padding-top:14px;">
    <span class="label">Client's own words</span>
    <p class="small" style="margin-top:6px;white-space:pre-wrap;">[[pain_points]]</p>
  </div>

  [[files_block]]
  [[appt_block]]
</div>
"""


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def landing():
    return LANDING_HTML


@app.route("/api/lawyers", methods=["POST"])
def create_lawyer():
    name = request.form.get("name", "").strip()
    firm_name = request.form.get("firm_name", "").strip()
    tagline = request.form.get("tagline", "").strip()
    email = request.form.get("email", "").strip()
    if not name or not firm_name or not email:
        abort(400, "Missing required fields")

    db = get_db()
    slug = unique_slug(firm_name)
    token = secrets.token_urlsafe(16)
    db.execute(
        "INSERT INTO lawyers (slug, token, name, firm_name, tagline, email, created_at) VALUES (?,?,?,?,?,?,?)",
        (slug, token, name, firm_name, tagline, email, datetime.utcnow().isoformat()),
    )
    db.commit()

    public_url = url_for("public_page", slug=slug, _external=True)
    dash_url = url_for("dashboard", slug=slug, token=token, _external=True)
    return render(LAWYER_CREATED_HTML, public_url=public_url, dash_url=dash_url)


@app.route("/l/<slug>")
def public_page(slug):
    lawyer = get_lawyer_by_slug(slug)
    options = "".join(f'<option value="{c}">{c}</option>' for c in CASE_TYPES)
    return render(
        PUBLIC_PAGE_HTML,
        firm_name=lawyer["firm_name"], name=lawyer["name"],
        tagline=lawyer["tagline"] or "Family law counsel.",
        slug=slug, case_type_options=options,
    )


@app.route("/submit/<slug>", methods=["POST"])
def submit_case(slug):
    lawyer = get_lawyer_by_slug(slug)
    client_name = request.form.get("client_name", "").strip()
    client_email = request.form.get("client_email", "").strip()
    client_phone = request.form.get("client_phone", "").strip()
    case_type = request.form.get("case_type", "Other")
    pain_points = request.form.get("pain_points", "").strip()
    if not client_name or not client_email or not pain_points:
        abort(400, "Missing required fields")

    sub_dir = os.path.join(UPLOAD_DIR, slug, str(uuid.uuid4()))
    saved_names = []
    files = request.files.getlist("files")
    if any(f.filename for f in files):
        os.makedirs(sub_dir, exist_ok=True)
        for f in files:
            if not f.filename:
                continue
            ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
            if ext not in ALLOWED_EXT:
                continue
            safe_name = f"{uuid.uuid4().hex[:8]}_{f.filename.replace('/', '_')}"
            f.save(os.path.join(sub_dir, safe_name))
            saved_names.append(f.filename)

    ai_result = run_ai_analysis(case_type, pain_points, saved_names)

    db = get_db()
    cur = db.execute(
        "INSERT INTO submissions (lawyer_id, client_name, client_email, client_phone, case_type, "
        "pain_points, files, ai_status, ai_result, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (lawyer["id"], client_name, client_email, client_phone, case_type, pain_points,
         json.dumps(saved_names), "done", json.dumps(ai_result), datetime.utcnow().isoformat()),
    )
    db.commit()
    submission_id = cur.lastrowid

    return render(
        BOOKING_HTML,
        firm_name=lawyer["firm_name"], name=lawyer["name"],
        client_name=client_name, submission_id=submission_id,
    )


@app.route("/book/<int:submission_id>", methods=["POST"])
def book_appointment(submission_id):
    db = get_db()
    sub = db.execute("SELECT * FROM submissions WHERE id=?", (submission_id,)).fetchone()
    if not sub:
        abort(404)
    lawyer = db.execute("SELECT * FROM lawyers WHERE id=?", (sub["lawyer_id"],)).fetchone()

    pref_date = request.form.get("pref_date", "")
    pref_time = request.form.get("pref_time", "")
    notes = request.form.get("notes", "")
    db.execute(
        "INSERT INTO appointments (submission_id, pref_date, pref_time, notes, created_at) VALUES (?,?,?,?,?)",
        (submission_id, pref_date, pref_time, notes, datetime.utcnow().isoformat()),
    )
    db.commit()

    return render(
        CONFIRM_HTML,
        firm_name=lawyer["firm_name"], name=lawyer["name"],
        pref_date=pref_date, pref_time=pref_time,
    )


@app.route("/dashboard/<slug>/<token>")
def dashboard(slug, token):
    lawyer = get_lawyer_by_slug(slug)
    if not secrets.compare_digest(token, lawyer["token"]):
        abort(403)

    db = get_db()
    subs = db.execute(
        "SELECT * FROM submissions WHERE lawyer_id=? ORDER BY id DESC", (lawyer["id"],)
    ).fetchall()

    ai_notice = ""
    if not AI_API_KEY:
        ai_notice = ('<div class="card" style="border-color:var(--accent);margin-bottom:20px;">'
                     '<span class="label">No AI key configured</span>'
                     '<p style="margin-top:6px;">Showing offline heuristic summaries. '
                     'Add <code>GROQ_API_KEY</code> as an environment variable and resubmit a case '
                     'to see real AI analysis. See README.md.</p></div>')

    if not subs:
        rows = ('<div class="card" style="margin-top:20px;"><p class="small">No submissions yet. '
                'Share your intake page with a client to see a case appear here.</p></div>')
    else:
        rows = ""
        for sub in subs:
            ai = json.loads(sub["ai_result"]) if sub["ai_result"] else {}
            appt = db.execute(
                "SELECT * FROM appointments WHERE submission_id=? ORDER BY id DESC LIMIT 1", (sub["id"],)
            ).fetchone()
            files = json.loads(sub["files"]) if sub["files"] else []

            key_issues = "".join(f"<li>{i}</li>" for i in ai.get("key_issues", [])) or "<li>—</li>"
            missing_docs = "".join(f"<li>{i}</li>" for i in ai.get("missing_documents", [])) or "<li>—</li>"
            questions = "".join(f"<li>{i}</li>" for i in ai.get("suggested_questions", [])) or "<li>—</li>"
            source_tag = "· offline heuristic" if ai.get("_source") != "ai" else "· AI-generated"

            files_block = ""
            if files:
                files_block = ('<div style="margin-top:12px;"><span class="label">Attached files</span>'
                               '<p class="small" style="margin-top:6px;">' + ", ".join(files) + "</p></div>")

            appt_block = ""
            if appt:
                appt_block = (f'<div style="margin-top:12px;border-top:1px solid var(--line);padding-top:12px;">'
                              f'<span class="label">Requested appointment</span>'
                              f'<p style="margin-top:6px;">{appt["pref_date"]} at {appt["pref_time"]}'
                              f'{" — " + appt["notes"] if appt["notes"] else ""}</p></div>')
            else:
                appt_block = ('<div style="margin-top:12px;border-top:1px solid var(--line);padding-top:12px;">'
                              '<span class="label">Appointment</span><p class="small" style="margin-top:6px;">Not yet booked.</p></div>')

            rows += render(
                CASE_ROW_HTML,
                id=f'{sub["id"]:04d}', client_name=sub["client_name"], case_type=sub["case_type"],
                client_email=sub["client_email"],
                phone_part=("· " + sub["client_phone"]) if sub["client_phone"] else "",
                created_at=sub["created_at"][:16].replace("T", " "),
                urgency=ai.get("urgency", "Low"),
                summary=ai.get("summary", "—"), source_tag=source_tag,
                key_issues=key_issues, missing_docs=missing_docs, questions=questions,
                pain_points=sub["pain_points"], files_block=files_block, appt_block=appt_block,
            )

    return render(DASHBOARD_HTML, firm_name=lawyer["firm_name"], slug=slug, ai_notice=ai_notice, rows=rows)


@app.errorhandler(404)
def not_found(e):
    return "<body style='font-family:monospace;padding:40px;'>404 — not found. <a href='/'>Back to ClientDat</a></body>", 404


@app.errorhandler(403)
def forbidden(e):
    return "<body style='font-family:monospace;padding:40px;'>403 — wrong dashboard link.</body>", 403


init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
