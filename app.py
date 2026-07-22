"""
ClientDat — branded client intake + AI case analysis for any legal practice.

Four things, done well:
  1. A lawyer generates a branded intake page (their own colour + theme).
  2. A client describes their situation, uploads documents, and books a time.
  3. The lawyer gets an AI-generated read on the case before the first call.
  4. Everything sensitive is encrypted at rest — case descriptions, uploaded
     files, and AI results are never stored as plain text or plain bytes.

Single-file Flask app by design: routes, SQLite persistence, encryption,
AI analysis, and every page's HTML/CSS all live in this one module.

Run locally:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000 — see README.md for AI + encryption setup.
"""

import os
import io
import json
import sqlite3
import uuid
import secrets
import base64
from datetime import datetime
from flask import Flask, request, url_for, g, abort
import urllib.request
import urllib.error
from cryptography.fernet import Fernet, InvalidToken

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "clientdat.db"))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

AI_API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("AI_API_KEY", "")
AI_API_URL = os.environ.get("AI_API_URL", "https://api.groq.com/openai/v1/chat/completions")
AI_MODEL = os.environ.get("AI_MODEL", "llama-3.1-8b-instant")

ALLOWED_EXT = {"pdf", "doc", "docx", "txt", "png", "jpg", "jpeg"}
CASE_TYPES = ["Family Law", "Criminal Defense", "Personal Injury", "Estate Planning",
              "Business / Corporate", "Real Estate", "Employment", "Immigration",
              "Civil Litigation", "Contract Dispute", "Intellectual Property", "Other"]

# Curated brand colours a lawyer can pick for their intake page.
ACCENTS = [
    ("FF5500", "Signal"),
    ("2F6B3A", "Forest"),
    ("3B4CCA", "Indigo"),
    ("8A2E3B", "Burgundy"),
    ("1E7F79", "Teal"),
    ("45516B", "Slate"),
]
DEFAULT_ACCENT = ACCENTS[0][0]

THEMES = {
    "light": {"bg": "#ECEAE2", "surface": "#F8F7F3", "ink": "#16140F",
              "ink60": "#615D52", "line": "#CFC9B8", "field": "#FFFFFF"},
    "dark": {"bg": "#15140F", "surface": "#1D1B15", "ink": "#F3F1E9",
             "ink60": "#A19C8C", "line": "#3A362B", "field": "#232017"},
}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(16))

# --------------------------------------------------------------------------
# Encryption at rest
# --------------------------------------------------------------------------
# Every case description, uploaded file, AI result, and appointment note is
# encrypted with Fernet (AES-128 + HMAC) before it touches the database or
# disk, and only decrypted in memory when a lawyer opens their own dashboard.

_env_key = os.environ.get("ENCRYPTION_KEY", "")
if _env_key:
    try:
        _fernet_key = _env_key.encode()
        Fernet(_fernet_key)  # validate
    except Exception:
        _fernet_key = base64.urlsafe_b64encode(_env_key.encode().ljust(32, b"0")[:32])
else:
    _fernet_key = Fernet.generate_key()
    print("[ClientDat] No ENCRYPTION_KEY set — generated a temporary key for this "
          "process only. Set ENCRYPTION_KEY in your environment so encrypted data "
          "is still readable after a restart.")

_fernet = Fernet(_fernet_key)


def enc_text(s):
    return _fernet.encrypt((s or "").encode("utf-8")).decode("ascii")


def dec_text(token):
    if not token:
        return ""
    try:
        return _fernet.decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken:
        return "[unable to decrypt]"


def enc_bytes(b):
    return _fernet.encrypt(b)


def dec_bytes(b):
    try:
        return _fernet.decrypt(b)
    except InvalidToken:
        return b""


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
            theme TEXT DEFAULT 'light',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lawyer_id INTEGER NOT NULL,
            client_name TEXT NOT NULL,
            client_email TEXT,
            client_phone TEXT,
            case_type TEXT,
            pain_points_enc TEXT,
            files TEXT,
            ai_result_enc TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (lawyer_id) REFERENCES lawyers(id)
        );

        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submission_id INTEGER NOT NULL,
            pref_date TEXT,
            pref_time TEXT,
            notes_enc TEXT,
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

SYSTEM_PROMPT = """You are a legal intake analyst helping a lawyer triage a new client \
submission, in any practice area, before the first meeting. You are not giving legal \
advice, you are summarizing and flagging what matters for the lawyer's prep.

Return ONLY valid JSON, no markdown fences, no commentary, matching exactly this shape:
{
  "summary": "2-3 sentence plain-language summary of the situation",
  "urgency": "Low, Medium, or High",
  "urgency_reason": "one short sentence",
  "key_issues": ["short issue", "..."],
  "missing_documents": ["document the lawyer should ask for", "..."],
  "suggested_questions": ["question to ask at the first meeting", "..."]
}
Keep every list to at most 5 short items. Base everything only on what the client wrote."""


def run_ai_analysis(case_type, pain_points, filenames):
    """Call an OpenAI-compatible chat completion endpoint. Falls back to a
    deterministic offline heuristic if no key is configured or the call
    fails, so the product always returns a result."""

    file_list = ", ".join(filenames) if filenames else "none uploaded"
    user_content = (
        f"Client-selected case type: {case_type}\n"
        f"Documents uploaded: {file_list}\n"
        f"Client's own description of their situation:\n{pain_points}"
    )

    if not AI_API_KEY:
        return _fallback_analysis(pain_points, reason="no_api_key")

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
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {AI_API_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip().strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
        result = json.loads(text)
        result["_source"] = "ai"
        return result
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
            json.JSONDecodeError, TimeoutError):
        return _fallback_analysis(pain_points, reason="ai_error")


def _fallback_analysis(pain_points, reason):
    text = (pain_points or "").lower()
    urgent_words = ["urgent", "emergency", "abuse", "unsafe", "threat", "danger", "police", "restraining"]
    urgency = "High" if any(w in text for w in urgent_words) else ("Medium" if len(text) > 200 else "Low")
    return {
        "summary": (pain_points[:220] + ("..." if len(pain_points) > 220 else "")) if pain_points else
                   "Client did not provide a written description.",
        "urgency": urgency,
        "urgency_reason": "Flagged by keyword scan pending a connected AI model."
                           if reason == "no_api_key" else "AI service unavailable — fallback used.",
        "key_issues": ["Connect an AI model to generate real case insights."],
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
    """Plain [[placeholder]] substitution — avoids collisions with the CSS/JS
    curly braces sitting in the surrounding template."""
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


def hexify(h):
    h = (h or DEFAULT_ACCENT).lstrip("#")
    return h if len(h) == 6 else DEFAULT_ACCENT


def shade(hex_color, factor):
    """Darken a hex colour by `factor` (0-1) for hover/shadow states."""
    h = hex_color.lstrip("#")
    r, gg, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r, gg, b = (max(0, int(c * (1 - factor))) for c in (r, gg, b))
    return f"{r:02X}{gg:02X}{b:02X}"


# --------------------------------------------------------------------------
# Theming — CSS is generated per-lawyer from their accent + theme choice
# --------------------------------------------------------------------------

def page_css(accent="FF5500", theme="light"):
    t = THEMES.get(theme, THEMES["light"])
    accent = hexify(accent)
    accent_shadow = shade(accent, 0.25)
    return f"""
:root{{
  --bg:{t['bg']}; --surface:{t['surface']}; --ink:{t['ink']}; --ink-60:{t['ink60']};
  --line:{t['line']}; --field:{t['field']}; --accent:#{accent}; --accent-shadow:#{accent_shadow};
  --accent-ink:#FFFFFF; --ok:#2F6B3A; --radius:6px;
  --mono:'Space Mono',ui-monospace,'SF Mono',monospace;
  --sans:'Inter',system-ui,-apple-system,sans-serif;
}}
*{{box-sizing:border-box;}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased;}}
h1,h2,h3{{font-family:var(--mono);font-weight:700;letter-spacing:-0.01em;margin:0;}}
.label{{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-60);}}
.wrap{{max-width:920px;margin:0 auto;padding:0 20px;}}
.narrow{{max-width:620px;}}
.topbar{{border-bottom:1px solid var(--line);padding:16px 0;}}
.topbar .wrap{{display:flex;align-items:center;justify-content:space-between;}}
.brandmark{{font-family:var(--mono);font-weight:700;font-size:15px;letter-spacing:-.02em;}}
.brandmark b{{color:var(--accent);}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:26px;}}
.steps{{display:flex;align-items:center;gap:0;margin:24px 0 32px;font-family:var(--mono);font-size:11px;
  text-transform:uppercase;letter-spacing:.06em;flex-wrap:wrap;}}
.steps .step{{border:1px solid var(--line);background:var(--surface);padding:8px 12px;border-radius:var(--radius);color:var(--ink-60);}}
.steps .step.active{{border-color:var(--accent);color:var(--accent);}}
.steps .arrow{{padding:0 8px;color:var(--ink-60);}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
@media(max-width:640px){{.grid2{{grid-template-columns:1fr;}}}}
.field{{margin-bottom:18px;}}
.field label{{display:block;margin-bottom:7px;font-size:13px;font-weight:600;}}
input[type=text],input[type=email],input[type=tel],input[type=date],input[type=time],select,textarea{{
  width:100%;font-family:var(--sans);font-size:14px;padding:12px 13px;border:1px solid var(--line);
  border-radius:var(--radius);background:var(--field);color:var(--ink);}}
textarea{{resize:vertical;min-height:120px;}}
input:focus,select:focus,textarea:focus{{outline:2px solid var(--accent);outline-offset:1px;}}
.file-drop{{border:1px dashed var(--line);border-radius:var(--radius);padding:20px;text-align:center;
  font-size:13px;color:var(--ink-60);background:var(--field);cursor:pointer;}}
.file-drop.has-files{{border-color:var(--accent);border-style:solid;color:var(--ink);}}
.btn{{font-family:var(--mono);text-transform:uppercase;letter-spacing:.06em;font-size:13px;font-weight:700;
  border:1px solid var(--accent);background:var(--accent);color:var(--accent-ink);padding:14px 22px;
  border-radius:var(--radius);cursor:pointer;box-shadow:3px 3px 0 var(--accent-shadow);
  transition:transform .08s, box-shadow .08s;}}
.btn:hover{{transform:translate(1px,1px);box-shadow:2px 2px 0 var(--accent-shadow);}}
.btn:active{{transform:translate(3px,3px);box-shadow:0 0 0 var(--accent-shadow);}}
.btn.ghost{{background:transparent;color:var(--ink);border-color:var(--line);box-shadow:none;}}
.btn:disabled{{opacity:.5;cursor:not-allowed;}}
footer{{border-top:1px solid var(--line);margin-top:56px;padding:20px 0;font-family:var(--mono);
  font-size:11px;color:var(--ink-60);text-transform:uppercase;letter-spacing:.06em;}}
a{{color:var(--ink);}}
.small{{font-size:12px;color:var(--ink-60);}}
.lock{{font-size:12px;color:var(--ink-60);display:flex;align-items:center;gap:6px;margin-top:12px;}}
.copybox{{display:flex;gap:8px;align-items:center;border:1px solid var(--line);background:var(--field);
  border-radius:var(--radius);padding:11px 13px;font-family:var(--mono);font-size:13px;overflow-x:auto;white-space:nowrap;}}
.copybox button{{margin-left:auto;flex-shrink:0;}}
.badge{{display:inline-block;font-family:var(--mono);font-size:11px;text-transform:uppercase;
  letter-spacing:.06em;padding:4px 9px;border-radius:var(--radius);border:1px solid;}}
.badge.High{{color:#B3261E;border-color:#B3261E;background:#FCEBEA;}}
.badge.Medium{{color:#8A5A00;border-color:#8A5A00;background:#FBF1DE;}}
.badge.Low{{color:var(--ok);border-color:var(--ok);background:#EAF3EB;}}
.swatches{{display:flex;gap:10px;flex-wrap:wrap;}}
.swatch{{position:relative;}}
.swatch input{{position:absolute;opacity:0;width:100%;height:100%;margin:0;cursor:pointer;}}
.swatch span{{display:block;width:34px;height:34px;border-radius:50%;border:2px solid transparent;box-shadow:0 0 0 1px var(--line);}}
.swatch input:checked + span{{border-color:var(--ink);box-shadow:0 0 0 2px var(--surface),0 0 0 4px var(--ink);}}
.theme-toggle{{display:flex;gap:8px;}}
.theme-toggle label{{flex:1;position:relative;}}
.theme-toggle input{{position:absolute;opacity:0;width:100%;height:100%;margin:0;cursor:pointer;}}
.theme-toggle span{{display:block;text-align:center;padding:10px;border:1px solid var(--line);border-radius:var(--radius);
  font-family:var(--mono);font-size:12px;text-transform:uppercase;letter-spacing:.06em;background:var(--field);}}
.theme-toggle input:checked + span{{border-color:var(--ink);background:var(--ink);color:var(--surface);}}
"""


HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&'
        'family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">'
        '<style>[[css]]</style>')


def page(title, body, accent=DEFAULT_ACCENT, theme="light"):
    head = render(HEAD, css=page_css(accent, theme))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{title}</title>{head}</head><body>{body}</body></html>')


# --------------------------------------------------------------------------
# Templates (page bodies)
# --------------------------------------------------------------------------

ACCENT_SWATCHES = "".join(
    f'<label class="swatch"><input type="radio" name="accent" value="{hexv}"'
    f'{" checked" if hexv == DEFAULT_ACCENT else ""}>'
    f'<span style="background:#{hexv}"></span></label>'
    for hexv, _label in ACCENTS
)

CASE_TYPE_OPTIONS = "".join(f'<option value="{c}">{c}</option>' for c in CASE_TYPES)


def landing_body():
    return f"""
<div class="topbar"><div class="wrap"><div class="brandmark">Client<b>Dat</b></div></div></div>
<div class="wrap" style="padding-top:56px;padding-bottom:50px;">
  <h1 style="font-size:36px;max-width:600px;">Intake and case analysis, for any practice.</h1>
  <p style="max-width:500px;color:var(--ink-60);margin-top:14px;">
    Create your branded intake page, share it like a business card, and get an
    AI-prepared read on every case before you walk into the first meeting.
  </p>

  <div class="card narrow" style="margin-top:34px;">
    <h2 style="font-size:18px;margin-bottom:20px;">Create your intake page</h2>
    <form method="post" action="/api/lawyers">
      <div class="grid2">
        <div class="field"><label>Your name</label><input type="text" name="name" required placeholder="Dana Whitfield"></div>
        <div class="field"><label>Firm name</label><input type="text" name="firm_name" required placeholder="Whitfield & Cole Law"></div>
      </div>
      <div class="field"><label>Tagline shown on your page</label>
        <input type="text" name="tagline" placeholder="Clear counsel, from first call through resolution."></div>
      <div class="field"><label>Email for booking notifications</label>
        <input type="email" name="email" required placeholder="dana@whitfieldlaw.com"></div>

      <div class="grid2">
        <div class="field">
          <label>Accent colour</label>
          <div class="swatches">{ACCENT_SWATCHES}</div>
        </div>
        <div class="field">
          <label>Theme</label>
          <div class="theme-toggle">
            <label><input type="radio" name="theme" value="light" checked><span>Light</span></label>
            <label><input type="radio" name="theme" value="dark"><span>Dark</span></label>
          </div>
        </div>
      </div>

      <button class="btn" type="submit">Generate my page →</button>
    </form>
  </div>
</div>
<footer><div class="wrap">Client<b style="color:var(--accent)">Dat</b></div></footer>
"""


def lawyer_created_body(public_url, dash_url):
    return f"""
<div class="topbar"><div class="wrap"><div class="brandmark">Client<b>Dat</b></div></div></div>
<div class="wrap narrow" style="padding-top:56px;">
  <h1 style="font-size:28px;">Your page is live.</h1>
  <p style="color:var(--ink-60);margin-top:8px;">
    Share the first link with clients. Keep the second one for yourself — it opens
    your private case dashboard and cannot be guessed.
  </p>

  <div class="field" style="margin-top:28px;">
    <label>Client-facing intake page</label>
    <div class="copybox"><span id="pub">[[public_url]]</span>
      <button class="btn ghost" type="button" onclick="copyIt('pub')">Copy</button></div>
  </div>
  <div class="field">
    <label>Your private dashboard</label>
    <div class="copybox"><span id="dash">[[dash_url]]</span>
      <button class="btn ghost" type="button" onclick="copyIt('dash')">Copy</button></div>
  </div>

  <a href="[[dash_url]]"><button class="btn" style="margin-top:6px;">Open my dashboard →</button></a>
</div>
<script>function copyIt(id){{navigator.clipboard.writeText(document.getElementById(id).textContent);}}</script>
<footer><div class="wrap">Client<b style="color:var(--accent)">Dat</b></div></footer>
""".replace("[[public_url]]", public_url).replace("[[dash_url]]", dash_url)


def public_page_body(lawyer, slug):
    return f"""
<div class="topbar"><div class="wrap">
  <div class="brandmark">[[firm_name]]</div>
  <span class="small">via Client<b style="color:var(--accent)">Dat</b></span>
</div></div>

<div class="wrap narrow" style="padding-top:44px;padding-bottom:50px;">
  <h1 style="font-size:26px;">[[name]]</h1>
  <p style="color:var(--ink-60);margin-top:8px;">[[tagline]]</p>

  <div class="steps">
    <div class="step active">1 · Your situation</div><div class="arrow">→</div>
    <div class="step">2 · We review</div><div class="arrow">→</div>
    <div class="step">3 · Pick a time</div>
  </div>

  <div class="card">
    <form method="post" action="/submit/[[slug]]" enctype="multipart/form-data" id="intakeForm">
      <div class="grid2">
        <div class="field"><label>Your name</label><input type="text" name="client_name" required></div>
        <div class="field"><label>Email</label><input type="email" name="client_email" required></div>
      </div>
      <div class="grid2">
        <div class="field"><label>Phone</label><input type="tel" name="client_phone"></div>
        <div class="field"><label>Case type</label><select name="case_type">{CASE_TYPE_OPTIONS}</select></div>
      </div>
      <div class="field">
        <label>What's happening?</label>
        <textarea name="pain_points" required placeholder="Timeline, main concerns, anything urgent..."></textarea>
      </div>
      <div class="field">
        <label>Relevant documents (optional)</label>
        <label class="file-drop" id="dropLabel">
          <input type="file" name="files" multiple style="display:none" id="fileInput">
          <span id="dropText">Click to attach PDFs, images, or docs</span>
        </label>
      </div>
      <button class="btn" type="submit" id="submitBtn">Submit &amp; continue →</button>
      <p class="lock">🔒 Your description and files are encrypted before they're stored.</p>
    </form>
  </div>
</div>
<script>
const fi = document.getElementById('fileInput');
const dl = document.getElementById('dropLabel');
fi.addEventListener('change', () => {{
  if(fi.files.length){{
    dl.classList.add('has-files');
    document.getElementById('dropText').textContent = fi.files.length + ' file(s) selected';
  }}
}});
document.getElementById('intakeForm').addEventListener('submit', () => {{
  document.getElementById('submitBtn').disabled = true;
  document.getElementById('submitBtn').textContent = 'Submitting…';
}});
</script>
<footer><div class="wrap">Client<b style="color:var(--accent)">Dat</b></div></footer>
""".replace("[[firm_name]]", lawyer["firm_name"]).replace("[[name]]", lawyer["name"]) \
   .replace("[[tagline]]", lawyer["tagline"] or "Legal counsel.").replace("[[slug]]", slug)


def booking_body(firm_name, name, client_name, submission_id):
    return f"""
<div class="topbar"><div class="wrap"><div class="brandmark">[[firm_name]]</div></div></div>
<div class="wrap narrow" style="padding-top:44px;padding-bottom:50px;">
  <h1 style="font-size:24px;">Thanks, [[client_name]]. One more step.</h1>
  <div class="steps">
    <div class="step">1 · Your situation</div><div class="arrow">→</div>
    <div class="step active">2 · Received</div><div class="arrow">→</div>
    <div class="step active">3 · Pick a time</div>
  </div>
  <div class="card">
    <h2 style="font-size:17px;margin-bottom:18px;">Suggest a time for your call with [[name]]</h2>
    <form method="post" action="/book/[[submission_id]]">
      <div class="grid2">
        <div class="field"><label>Preferred date</label><input type="date" name="pref_date" required></div>
        <div class="field"><label>Preferred time</label><input type="time" name="pref_time" required></div>
      </div>
      <div class="field"><label>Anything else? (optional)</label><textarea name="notes" placeholder="e.g. I can only do mornings"></textarea></div>
      <button class="btn" type="submit">Request appointment →</button>
    </form>
  </div>
</div>
<footer><div class="wrap">Client<b style="color:var(--accent)">Dat</b></div></footer>
""".replace("[[firm_name]]", firm_name).replace("[[client_name]]", client_name) \
   .replace("[[name]]", name).replace("[[submission_id]]", str(submission_id))


def confirm_body(firm_name, name, pref_date, pref_time):
    return f"""
<div class="topbar"><div class="wrap"><div class="brandmark">[[firm_name]]</div></div></div>
<div class="wrap narrow" style="padding-top:44px;padding-bottom:50px;">
  <h1 style="font-size:24px;">You're all set.</h1>
  <div class="steps">
    <div class="step">1 · Your situation</div><div class="arrow">→</div>
    <div class="step">2 · Received</div><div class="arrow">→</div>
    <div class="step active">3 · Time requested</div>
  </div>
  <div class="card">
    <p>[[name]] will confirm your appointment on <b>[[pref_date]] at [[pref_time]]</b>.
    You'll be contacted at the email you provided.</p>
  </div>
</div>
<footer><div class="wrap">Client<b style="color:var(--accent)">Dat</b></div></footer>
""".replace("[[firm_name]]", firm_name).replace("[[name]]", name) \
   .replace("[[pref_date]]", pref_date).replace("[[pref_time]]", pref_time)


def case_row_html(sub, ai, appt, files):
    key_issues = "".join(f"<li>{i}</li>" for i in ai.get("key_issues", [])) or "<li>—</li>"
    missing_docs = "".join(f"<li>{i}</li>" for i in ai.get("missing_documents", [])) or "<li>—</li>"
    questions = "".join(f"<li>{i}</li>" for i in ai.get("suggested_questions", [])) or "<li>—</li>"
    source_tag = "· preliminary read" if ai.get("_source") != "ai" else "· AI-generated"

    files_block = ""
    if files:
        files_block = ('<div style="margin-top:14px;"><span class="label">Attached files (encrypted)</span>'
                        '<p class="small" style="margin-top:6px;">' + ", ".join(files) + "</p></div>")

    if appt:
        notes = dec_text(appt["notes_enc"])
        appt_block = (f'<div style="margin-top:14px;border-top:1px solid var(--line);padding-top:14px;">'
                      f'<span class="label">Requested appointment</span>'
                      f'<p style="margin-top:6px;">{appt["pref_date"]} at {appt["pref_time"]}'
                      f'{" — " + notes if notes else ""}</p></div>')
    else:
        appt_block = ('<div style="margin-top:14px;border-top:1px solid var(--line);padding-top:14px;">'
                      '<span class="label">Appointment</span><p class="small" style="margin-top:6px;">Not yet booked.</p></div>')

    phone_part = f'· {sub["client_phone"]}' if sub["client_phone"] else ""
    pain_points = dec_text(sub["pain_points_enc"])

    return f"""
<div class="card" style="margin-top:18px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;">
    <div>
      <h3 style="font-size:17px;">{sub["client_name"]} <span class="small">· {sub["case_type"]}</span></h3>
      <p class="small" style="margin-top:4px;">{sub["client_email"]} {phone_part} · submitted {sub["created_at"][:16].replace("T", " ")}</p>
    </div>
    <span class="badge {ai.get("urgency", "Low")}">{ai.get("urgency", "Low")} urgency</span>
  </div>

  <div class="grid2" style="margin-top:18px;">
    <div>
      <span class="label">Summary {source_tag}</span>
      <p style="margin-top:6px;">{ai.get("summary", "—")}</p>
      <span class="label">Key issues</span>
      <ul style="margin:6px 0 0 18px;padding:0;">{key_issues}</ul>
    </div>
    <div>
      <span class="label">Missing documents</span>
      <ul style="margin:6px 0 14px 18px;padding:0;">{missing_docs}</ul>
      <span class="label">Suggested first-meeting questions</span>
      <ul style="margin:6px 0 0 18px;padding:0;">{questions}</ul>
    </div>
  </div>

  <div style="margin-top:16px;border-top:1px solid var(--line);padding-top:14px;">
    <span class="label">Client's own words</span>
    <p class="small" style="margin-top:6px;white-space:pre-wrap;">{pain_points}</p>
  </div>

  {files_block}
  {appt_block}
</div>
"""


def dashboard_body(lawyer, rows_html):
    ai_notice = ""
    if not AI_API_KEY:
        ai_notice = ('<div class="card" style="margin-bottom:20px;">'
                     '<span class="label">Preliminary read mode</span>'
                     '<p style="margin-top:6px;" class="small">Connect an AI model to turn on full case analysis — see README.md.</p></div>')

    return f"""
<div class="topbar"><div class="wrap">
  <div class="brandmark">[[firm_name]]</div>
  <span class="small">Client<b style="color:var(--accent)">Dat</b></span>
</div></div>

<div class="wrap" style="padding-top:44px;padding-bottom:60px;">
  <h1 style="font-size:26px;">Cases</h1>
  <p class="small" style="margin-top:6px;">Your intake page: <a href="/l/[[slug]]">/l/[[slug]]</a></p>

  {ai_notice}
  {rows_html}
</div>
<footer><div class="wrap">Client<b style="color:var(--accent)">Dat</b></div></footer>
""".replace("[[firm_name]]", lawyer["firm_name"]).replace("[[slug]]", lawyer["slug"])


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def landing():
    return page("ClientDat — intake & case analysis for any legal practice", landing_body())


@app.route("/api/lawyers", methods=["POST"])
def create_lawyer():
    name = request.form.get("name", "").strip()
    firm_name = request.form.get("firm_name", "").strip()
    tagline = request.form.get("tagline", "").strip()
    email = request.form.get("email", "").strip()
    accent = hexify(request.form.get("accent", DEFAULT_ACCENT))
    theme = request.form.get("theme", "light")
    if theme not in THEMES:
        theme = "light"
    if not name or not firm_name or not email:
        abort(400, "Missing required fields")

    db = get_db()
    slug = unique_slug(firm_name)
    token = secrets.token_urlsafe(16)
    db.execute(
        "INSERT INTO lawyers (slug, token, name, firm_name, tagline, email, accent, theme, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (slug, token, name, firm_name, tagline, email, accent, theme, datetime.utcnow().isoformat()),
    )
    db.commit()

    public_url = url_for("public_page", slug=slug, _external=True)
    dash_url = url_for("dashboard", slug=slug, token=token, _external=True)
    return page("Your page is live — ClientDat", lawyer_created_body(public_url, dash_url))


@app.route("/l/<slug>")
def public_page(slug):
    lawyer = get_lawyer_by_slug(slug)
    return page(f'{lawyer["firm_name"]} — Case Intake', public_page_body(lawyer, slug),
                accent=lawyer["accent"], theme=lawyer["theme"])


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
            raw = f.read()
            encrypted = enc_bytes(raw)
            safe_name = f"{uuid.uuid4().hex[:8]}_{f.filename.replace('/', '_')}.enc"
            with open(os.path.join(sub_dir, safe_name), "wb") as out:
                out.write(encrypted)
            saved_names.append(f.filename)

    ai_result = run_ai_analysis(case_type, pain_points, saved_names)

    db = get_db()
    cur = db.execute(
        "INSERT INTO submissions (lawyer_id, client_name, client_email, client_phone, case_type, "
        "pain_points_enc, files, ai_result_enc, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (lawyer["id"], client_name, client_email, client_phone, case_type,
         enc_text(pain_points), json.dumps(saved_names), enc_text(json.dumps(ai_result)),
         datetime.utcnow().isoformat()),
    )
    db.commit()
    submission_id = cur.lastrowid

    return page(f'Book a time — {lawyer["firm_name"]}',
                booking_body(lawyer["firm_name"], lawyer["name"], client_name, submission_id),
                accent=lawyer["accent"], theme=lawyer["theme"])


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
        "INSERT INTO appointments (submission_id, pref_date, pref_time, notes_enc, created_at) VALUES (?,?,?,?,?)",
        (submission_id, pref_date, pref_time, enc_text(notes), datetime.utcnow().isoformat()),
    )
    db.commit()

    return page(f'Request sent — {lawyer["firm_name"]}',
                confirm_body(lawyer["firm_name"], lawyer["name"], pref_date, pref_time),
                accent=lawyer["accent"], theme=lawyer["theme"])


@app.route("/dashboard/<slug>/<token>")
def dashboard(slug, token):
    lawyer = get_lawyer_by_slug(slug)
    if not secrets.compare_digest(token, lawyer["token"]):
        abort(403)

    db = get_db()
    subs = db.execute("SELECT * FROM submissions WHERE lawyer_id=? ORDER BY id DESC", (lawyer["id"],)).fetchall()

    if not subs:
        rows_html = '<div class="card"><p class="small">No submissions yet. Share your intake page to see a case appear here.</p></div>'
    else:
        rows_html = ""
        for sub in subs:
            ai_json = dec_text(sub["ai_result_enc"])
            try:
                ai = json.loads(ai_json) if ai_json else {}
            except json.JSONDecodeError:
                ai = {}
            appt = db.execute(
                "SELECT * FROM appointments WHERE submission_id=? ORDER BY id DESC LIMIT 1", (sub["id"],)
            ).fetchone()
            files = json.loads(sub["files"]) if sub["files"] else []
            rows_html += case_row_html(sub, ai, appt, files)

    return page(f'Dashboard — {lawyer["firm_name"]} — ClientDat', dashboard_body(lawyer, rows_html),
                accent=lawyer["accent"], theme=lawyer["theme"])


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
