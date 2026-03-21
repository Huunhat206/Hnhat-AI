#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║              H N H A T   A I   v3.0                          ║
║                                                              ║
║  Text  → Groq API  (llama / deepseek / mixtral / qwen…)      ║
║  Image → Gemini API (gemini-2.5-flash / pro)                 ║
║                                                              ║
║  10 Models · 20MB Code Upload · Chunked Analysis             ║
║  Streaming · Vision · Export · Dark/Light · 3000+ lines      ║
╚══════════════════════════════════════════════════════════════╝

  Install  :  pip install flask openai google-generativeai pillow
  Run      :  python Hnhatai.py
  Open     :  http://localhost:5000

  Groq key   → https://console.groq.com/keys
  Gemini key → https://aistudio.google.com/app/apikey
"""

# ─────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────
import os
import sys
import json
import uuid
import base64
import datetime
import threading
import webbrowser
import mimetypes
import traceback
from io import BytesIO
from pathlib import Path
from flask import (
    Flask, render_template_string, request,
    jsonify, Response, stream_with_context
)

# Groq / OpenAI-compatible client
try:
    from openai import OpenAI
    GROQ_LIB = True
except ImportError:
    GROQ_LIB = False
    print("  [WARN] openai not installed. Run: pip install openai")

# Gemini
try:
    import google.generativeai as genai
    GEMINI_LIB = True
except ImportError:
    GEMINI_LIB = False
    genai = None
    print("  [WARN] google-generativeai not installed. Run: pip install google-generativeai")

# PIL for image handling
try:
    from PIL import Image as PILImage
    PIL_LIB = True
except ImportError:
    PIL_LIB = False
    PILImage = None

if not GROQ_LIB:
    print("\n  ERROR: pip install flask openai\n")
    sys.exit(1)

def get_data_dir():
    """ Lấy thư mục chứa file .exe đang chạy để lưu file dữ liệu (JSON, Uploads) """
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).parent

def resource_path(relative_path):
    """ Lấy đường dẫn tới các file tĩnh được gói bên trong file .exe (Ảnh) """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return Path(base_path) / relative_path

# ─────────────────────────────────────────────────────────────
#  APP + PATHS
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "hnhat-v3-2025"

DATA_DIR = get_data_dir()
CONFIG_FILE  = DATA_DIR / "hnhat_config.json"
CHATS_FILE   = DATA_DIR / "hnhat_chats.json"
UPLOADS_DIR  = DATA_DIR / "hnhat_uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

MAX_FILE_BYTES = 20 * 1024 * 1024          # 20 MB hard cap
DIRECT_LIMIT   = 1_000_000                 # 1 MB — inject full content
CHUNK_CHARS    = 80_000                    # ~20k tokens per analysis chunk

# ─────────────────────────────────────────────────────────────
#  SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────
_BASE = (
    "Bạn là Hnhat AI — trợ lý cá nhân thông minh, được tạo bởi Hnhat.\n"
    "Chuyên sâu: Python · Web · Desktop · Automation · AI/ML · DevOps.\n"
    "Luôn trả lời bằng tiếng Việt trừ khi được yêu cầu khác.\n"
    "Câu trả lời rõ ràng, chính xác, có ví dụ thực tế khi cần."
)

SYS_CODE = (
    "Bạn là Hnhat Code — AI chuyên gia lập trình đỉnh cao, được tạo bởi Hnhat.\n"
    "Năng lực cốt lõi:\n"
    "• Đọc & phân tích toàn bộ codebase lớn (hàng ngàn dòng), tìm bug & lỗ hổng\n"
    "• Viết code sạch, tối ưu hiệu suất, có test đầy đủ, theo best practices\n"
    "• Refactor, code review chuyên sâu, giải thích từng dòng\n"
    "• Thiết kế kiến trúc: microservices, REST API, database schema\n"
    "• Build: .exe (PyInstaller) · Docker · CI/CD · VPS deploy\n"
    "• Stack: Python · JS/TS/Node · Go · Rust · C/C++ · Java · SQL · Bash\n"
    "Luôn dùng code blocks chuẩn, giải thích logic, đề xuất cải tiến cụ thể."
)

SYS_VISION = (
    "Bạn là Hnhat Vision — AI phân tích hình ảnh thông minh, được tạo bởi Hnhat.\n"
    "Khả năng: Mô tả chi tiết ảnh, đọc text trong ảnh (OCR), debug UI từ screenshot,\n"
    "phân tích biểu đồ/sơ đồ kiến trúc, nhận diện code từ ảnh chụp màn hình,\n"
    "đánh giá design, nhận diện lỗi từ error screenshot."
)

SYS_REASON = (
    "Bạn là Hnhat Reason — AI suy luận chuyên sâu, được tạo bởi Hnhat.\n"
    "Bạn suy nghĩ từng bước trước khi trả lời.\n"
    "Chuyên về: toán học, logic, lập kế hoạch phức tạp, debug khó, phân tích hệ thống.\n"
    "Luôn trình bày quá trình suy luận rõ ràng."
)

# ─────────────────────────────────────────────────────────────
#  MODELS REGISTRY
# ─────────────────────────────────────────────────────────────
GROQ_BASE = "https://api.groq.com/openai/v1"

MODELS = {
    # ── Groq models ────────────────────────────────────────
    "Hnhat Fast": {
        "id": "llama-3.1-8b-instant",
        "api": "groq",
        "icon": "⚡",
        "color": "#f59e0b",
        "badge": "FAST",
        "desc": "Siêu nhanh · Tiết kiệm",
        "ctx": "128K",
        "max_tokens": 4096,
        "sys": _BASE,
    },
    "Hnhat Pro": {
        "id": "llama-3.3-70b-versatile",
        "api": "groq",
        "icon": "🔥",
        "color": "#8b5cf6",
        "badge": "PRO",
        "desc": "Cân bằng · Thông minh nhất",
        "ctx": "128K",
        "max_tokens": 8192,
        "sys": _BASE,
    },
    "Hnhat Master": {
        "id": "llama-3.3-70b-versatile",
        "api": "groq",
        "icon": "👑",
        "color": "#06b6d4",
        "badge": "MASTER",
        "desc": "LLaMA 3.3 70B · Toàn năng nhất",
        "ctx": "128K",
        "max_tokens": 8192,
        "sys": SYS_REASON,
    },
    "Hnhat Code": {
        "id": "llama-3.3-70b-versatile",
        "api": "groq",
        "icon": "💻",
        "color": "#10b981",
        "badge": "CODE",
        "desc": "Chuyên code · Upload 20MB",
        "ctx": "128K",
        "max_tokens": 8192,
        "sys": SYS_CODE,
    },
    "Hnhat Reason": {
        "id": "qwen-qwq-32b",
        "api": "groq",
        "icon": "🧠",
        "color": "#f97316",
        "badge": "REASON",
        "desc": "QwQ 32B · Tư duy chuỗi",
        "ctx": "128K",
        "max_tokens": 8192,
        "sys": SYS_REASON,
    },
    "Hnhat Vision": {
        "id": "gemini-2.5-flash",
        "api": "gemini",
        "icon": "👁",
        "color": "#4285f4",
        "badge": "VISION",
        "desc": "Gemini Flash · Phân tích ảnh",
        "ctx": "1M",
        "max_tokens": 4096,
        "sys": SYS_VISION,
    },
    "Hnhat Vision+": {
        "id": "gemini-2.5-pro",
        "api": "gemini",
        "icon": "🔭",
        "color": "#1a73e8",
        "badge": "VIS+",
        "desc": "Gemini Pro · Ảnh chất lượng cao",
        "ctx": "1M",
        "max_tokens": 4096,
        "sys": SYS_VISION,
    },
    "Kimi K2": {
        "id": "moonshotai/kimi-k2-instruct",
        "api": "groq",
        "icon": "🌀",
        "color": "#0ea5e9",
        "badge": "MIXTRAL",
        "desc": "MoE · Context 32K",
        "ctx": "32K",
        "max_tokens": 4096,
        "sys": _BASE,
    },
    "Gemma 2": {
        "id": "gemma2-9b-it",
        "api": "groq",
        "icon": "💎",
        "color": "#14b8a6",
        "badge": "GEMMA",
        "desc": "Gemma 2 · Nhỏ gọn & nhanh",
        "ctx": "8K",
        "max_tokens": 2048,
        "sys": _BASE,
    },
    "Compound": {
        "id": "compound-beta",
        "api": "groq",
        "icon": "⚗️",
        "color": "#a855f7",
        "badge": "COMP",
        "desc": "Compound AI · Đa năng",
        "ctx": "128K",
        "max_tokens": 4096,
        "sys": _BASE,
    },
}

# Code-only text extensions
TEXT_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".scss",
    ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".env",
    ".txt", ".md", ".rst", ".sql", ".sh", ".bash", ".zsh", ".ps1",
    ".java", ".kt", ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".php", ".rb", ".swift", ".dart", ".r", ".lua",
    ".vue", ".svelte", ".graphql", ".proto", ".tf", ".bicep",
    ".dockerfile", ".makefile", ".gradle", ".cmake", ".nim",
}

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text("utf-8"))
        except Exception:
            pass
    return {
        "groq_key": "",
        "gemini_key": "",
        "default_model": "Hnhat Pro",
        "theme": "dark",
        "system_prompt": "",
    }


def save_config(c: dict):
    CONFIG_FILE.write_text(json.dumps(c, ensure_ascii=False, indent=2), "utf-8")


def load_chats() -> dict:
    if CHATS_FILE.exists():
        try:
            return json.loads(CHATS_FILE.read_text("utf-8"))
        except Exception:
            pass
    return {}


def save_chats(c: dict):
    CHATS_FILE.write_text(json.dumps(c, ensure_ascii=False, indent=2), "utf-8")


def groq_client(key: str) -> OpenAI:
    return OpenAI(api_key=key, base_url=GROQ_BASE)


def now_iso() -> str:
    return datetime.datetime.now().isoformat()


def chunk_text(text: str, chunk_size: int = CHUNK_CHARS) -> list:
    """
    Split text into overlapping chunks, trying to break at blank lines.
    Returns list of strings.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    lines = text.splitlines(keepends=True)
    current: list = []
    size = 0
    overlap_lines = 15  # lines kept from previous chunk for context

    for line in lines:
        current.append(line)
        size += len(line)
        if size >= chunk_size:
            chunks.append("".join(current))
            # keep tail for overlap
            current = current[-overlap_lines:] if len(current) > overlap_lines else []
            size = sum(len(l) for l in current)

    if current:
        chunks.append("".join(current))

    return chunks


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_error(msg: str) -> str:
    return sse({"error": msg})


# ─────────────────────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


# ── Config ────────────────────────────────────────────────────

@app.route("/api/config", methods=["GET"])
def r_config_get():
    c = load_config()
    return jsonify({
        "default_model":  c.get("default_model", "Hnhat Pro"),
        "theme":          c.get("theme", "dark"),
        "has_groq":       bool(c.get("groq_key", "").strip()),
        "has_gemini":     bool(c.get("gemini_key", "").strip()),
        "bg_image":       c.get("bg_image", ""),
        "bg_blur":        c.get("bg_blur", 0),
        "bg_opacity":     c.get("bg_opacity", 0.15),
        "font_size":      c.get("font_size", "md"),
        "chat_bubble":    c.get("chat_bubble", "default"),
        "send_on_enter":  c.get("send_on_enter", True),
    })


@app.route("/api/config", methods=["POST"])
def r_config_set():
    d = request.json or {}
    c = load_config()
    for k in ("groq_key", "gemini_key", "default_model", "theme", "system_prompt",
              "bg_image", "bg_blur", "bg_opacity", "font_size", "chat_bubble", "send_on_enter"):
        if k in d:
            v = d[k]
            c[k] = v.strip() if isinstance(v, str) else v
    save_config(c)
    return jsonify({"ok": True})


# ── Chats ─────────────────────────────────────────────────────

@app.route("/api/chats", methods=["GET"])
def r_chats_list():
    chats = load_chats()
    out = [
        {
            "id":      v["id"],
            "title":   v.get("title", "Chat"),
            "model":   v.get("model", "Hnhat Pro"),
            "updated": v.get("updated", ""),
            "count":   len(v.get("messages", [])),
        }
        for v in sorted(chats.values(), key=lambda x: x.get("updated", ""), reverse=True)
    ]
    return jsonify(out)


@app.route("/api/chats/<cid>", methods=["GET"])
def r_chat_get(cid):
    chats = load_chats()
    if cid not in chats:
        return jsonify({"error": "not found"}), 404
    return jsonify(chats[cid])


@app.route("/api/chats", methods=["POST"])
def r_chat_new():
    d    = request.json or {}
    cid  = str(uuid.uuid4())
    now_ = now_iso()
    chat = {
        "id":       cid,
        "title":    "Cuộc trò chuyện mới",
        "model":    d.get("model", "Hnhat Pro"),
        "messages": [],
        "created":  now_,
        "updated":  now_,
    }
    chats = load_chats()
    chats[cid] = chat
    save_chats(chats)
    return jsonify(chat)


@app.route("/api/chats/<cid>", methods=["DELETE"])
def r_chat_del(cid):
    chats = load_chats()
    chats.pop(cid, None)
    save_chats(chats)
    # Clean up associated upload files
    for f in UPLOADS_DIR.glob(f"{cid}_*"):
        try:
            f.unlink()
        except Exception:
            pass
    return jsonify({"ok": True})


@app.route("/api/chats/<cid>/rename", methods=["POST"])
def r_chat_rename(cid):
    chats = load_chats()
    if cid in chats:
        chats[cid]["title"] = (request.json or {}).get("title", "Chat")[:80]
        save_chats(chats)
    return jsonify({"ok": True})


@app.route("/api/chats/<cid>/clear", methods=["POST"])
def r_chat_clear(cid):
    chats = load_chats()
    if cid in chats:
        chats[cid]["messages"] = []
        chats[cid]["title"]    = "Cuộc trò chuyện mới"
        chats[cid]["updated"]  = now_iso()
        save_chats(chats)
    return jsonify({"ok": True})


# ── File Upload ───────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def r_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f    = request.files["file"]
    data = f.read()
    size = len(data)

    if size > MAX_FILE_BYTES:
        return jsonify({
            "error": f"File quá lớn ({size // (1024*1024):.1f}MB). Tối đa 20MB."
        }), 400

    ext  = Path(f.filename).suffix.lower()
    mime = f.content_type or mimetypes.guess_type(f.filename)[0] or "application/octet-stream"

    # ── Image ──
    if mime.startswith("image/"):
        b64 = base64.b64encode(data).decode()
        return jsonify({
            "type":  "image",
            "name":  f.filename,
            "mime":  mime,
            "b64":   b64,
            "size":  size,
        })

    # ── Text / Code ──
    if ext in TEXT_EXTS or mime.startswith("text/"):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="replace")

        file_id = str(uuid.uuid4())
        (UPLOADS_DIR / file_id).write_text(text, "utf-8")

        return jsonify({
            "type":    "text",
            "name":    f.filename,
            "file_id": file_id,
            "size":    size,
            "lines":   text.count("\n") + 1,
            "tokens":  est_tokens(text),
            "large":   size > DIRECT_LIMIT,
            "preview": text[:500],
        })

    return jsonify({
        "error": "Định dạng không hỗ trợ. Hãy dùng file code/text hoặc ảnh."
    }), 400


# ── Regular Chat Stream (Groq) ────────────────────────────────

@app.route("/api/chat/stream", methods=["POST"])
def r_chat_stream():
    d          = request.json or {}
    cid        = d.get("chat_id")
    user_msg   = d.get("message", "").strip()
    model_name = d.get("model", "Hnhat Pro")
    file_meta  = d.get("file")          # {type, name, file_id, content, b64, mime, size}

    cfg = load_config()
    key = cfg.get("groq_key", "").strip()

    if not key:
        def _e():
            yield sse_error("⚠️ Chưa cài Groq API Key! Vào ⚙️ Settings để nhập.")
        return Response(stream_with_context(_e()), mimetype="text/event-stream")

    m         = MODELS.get(model_name, MODELS["Hnhat Pro"])
    model_id  = m["id"]
    max_tok   = m["max_tokens"]
    sys_text  = cfg.get("system_prompt") or m["sys"]

    # ── Build user message content ──────────────────────────
    user_content = user_msg

    if file_meta and file_meta.get("type") == "text":
        fname   = file_meta.get("name", "file")
        fid     = file_meta.get("file_id", "")
        fsize   = file_meta.get("size", 0)

        # Load file content
        fpath = UPLOADS_DIR / fid
        if fid and fpath.exists():
            try:
                fc = fpath.read_text("utf-8")
            except Exception:
                fc = fpath.read_text("latin-1", errors="replace")
        else:
            fc = file_meta.get("content", "")

        ext_ = Path(fname).suffix.lstrip(".")
        size_kb = fsize // 1024

        if fsize <= DIRECT_LIMIT:
            user_content = (
                f"📎 **{fname}** ({size_kb}KB · {file_meta.get('lines', 0)} dòng)\n\n"
                f"```{ext_}\n{fc}\n```\n\n"
                f"{user_msg}"
            )
        else:
            # Large file: truncate with warning
            limit = DIRECT_LIMIT
            user_content = (
                f"📎 **{fname}** ({size_kb}KB · {file_meta.get('lines', 0)} dòng)\n"
                f"⚠️ File lớn — hiển thị {limit//1024}KB đầu tiên. Dùng **Hnhat Code** để phân tích toàn bộ.\n\n"
                f"```{ext_}\n{fc[:limit]}\n```\n\n"
                f"{user_msg}"
            )

    # ── Persist to chat ──────────────────────────────────────
    chats = load_chats()
    if not cid or cid not in chats:
        cid  = str(uuid.uuid4())
        now_ = now_iso()
        chats[cid] = {
            "id":       cid,
            "title":    "Cuộc trò chuyện mới",
            "model":    model_name,
            "messages": [],
            "created":  now_,
            "updated":  now_,
        }

    chat = chats[cid]
    chat["messages"].append({"role": "user",      "content": user_content, "model": model_name})
    chat["model"]   = model_name
    chat["updated"] = now_iso()

    # ── Build API messages ───────────────────────────────────
    api_msgs = [{"role": "system", "content": sys_text}]
    # Keep last 40 messages to avoid context overflow
    for msg in chat["messages"][-40:]:
        api_msgs.append({"role": msg["role"], "content": msg["content"]})

    def generate():
        collected = []
        try:
            client = groq_client(key)
            stream = client.chat.completions.create(
                model=model_id,
                messages=api_msgs,
                stream=True,
                max_tokens=max_tok,
                temperature=0.7,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    collected.append(delta)
                    yield sse({"content": delta})

        except Exception as ex:
            yield sse_error(f"Groq API lỗi: {ex}")
            return

        reply = "".join(collected)
        chat["messages"].append({"role": "assistant", "content": reply, "model": model_name})

        # Auto-title from first user turn
        if len(chat["messages"]) <= 4 and "mới" in chat["title"]:
            raw = user_msg or (file_meta or {}).get("name", "Chat")
            chat["title"] = raw[:60] + ("…" if len(raw) > 60 else "")

        chats[cid] = chat
        save_chats(chats)
        yield sse({"done": True, "chat_id": cid})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── Code Chunked Analysis (20 MB) ────────────────────────────

@app.route("/api/code/analyze", methods=["POST"])
def r_code_analyze():
    """
    Stream chunked analysis for files > DIRECT_LIMIT.
    Steps: per-chunk summary → final synthesis answer.
    """
    d          = request.json or {}
    file_id    = d.get("file_id", "")
    filename   = d.get("filename", "code")
    question   = d.get("question", "Phân tích và tóm tắt code này")
    chat_id    = d.get("chat_id")
    model_name = d.get("model", "Hnhat Code")

    cfg = load_config()
    key = cfg.get("groq_key", "").strip()

    if not key:
        def _e():
            yield sse_error("Chưa cài Groq API Key!")
        return Response(stream_with_context(_e()), mimetype="text/event-stream")

    fpath = UPLOADS_DIR / file_id
    if not fpath.exists():
        def _e():
            yield sse_error("File không tìm thấy. Hãy upload lại.")
        return Response(stream_with_context(_e()), mimetype="text/event-stream")

    try:
        content = fpath.read_text("utf-8")
    except Exception:
        content = fpath.read_text("latin-1", errors="replace")

    chunks      = chunk_text(content)
    total_size  = len(content)
    ext_        = Path(filename).suffix.lstrip(".")

    def generate():
        client = groq_client(key)

        # ── Single chunk (≤ 80k chars) ───────────────────────
        if len(chunks) == 1:
            yield sse({"info": f"📄 Phân tích {total_size // 1024}KB code…"})
            msgs = [
                {"role": "system",  "content": SYS_CODE},
                {"role": "user",
                 "content": (
                     f"File: **{filename}** ({total_size//1024}KB)\n\n"
                     f"```{ext_}\n{content}\n```\n\n"
                     f"**Câu hỏi:** {question}"
                 )},
            ]
            try:
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile", messages=msgs,
                    stream=True, max_tokens=8192,
                )
                for chunk_ in stream:
                    delta = chunk_.choices[0].delta.content
                    if delta:
                        yield sse({"content": delta})
            except Exception as ex:
                yield sse_error(f"Lỗi: {ex}")
                return

        # ── Multiple chunks ───────────────────────────────────
        else:
            yield sse({
                "info": (
                    f"📂 File lớn: {total_size//1024}KB · "
                    f"{len(chunks)} phần · ~{est_tokens(content):,} tokens\n"
                    f"Đang phân tích từng phần…"
                )
            })

            summaries: list = []
            for i, chunk_ in enumerate(chunks, 1):
                pct = int(i / len(chunks) * 100)
                yield sse({
                    "progress": {
                        "current": i,
                        "total":   len(chunks),
                        "pct":     pct,
                        "label":   f"Phần {i}/{len(chunks)} ({len(chunk_)//1024}KB)…",
                    }
                })

                summary_msgs = [
                    {"role": "system",
                     "content": (
                         "Tóm tắt ngắn đoạn code sau: nêu các hàm/class quan trọng, "
                         "logic chính, patterns sử dụng. Tối đa 300 từ, tiếng Việt."
                     )},
                    {"role": "user",
                     "content": (
                         f"Phần {i}/{len(chunks)} của `{filename}`:\n\n"
                         f"```{ext_}\n{chunk_}\n```"
                     )},
                ]
                try:
                    resp = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=summary_msgs,
                        max_tokens=400,
                    )
                    summary = resp.choices[0].message.content or ""
                except Exception as ex:
                    summary = f"(Lỗi phân tích phần {i}: {ex})"

                summaries.append(f"### Phần {i}/{len(chunks)}\n{summary}")

            # Final synthesis
            yield sse({"info": "✨ Đang tổng hợp và trả lời…"})

            combined_summary = "\n\n".join(summaries)
            final_msgs = [
                {"role": "system", "content": SYS_CODE},
                {"role": "user",
                 "content": (
                     f"Tôi vừa phân tích file `{filename}` ({total_size//1024}KB, "
                     f"{len(chunks)} phần).\n\n"
                     f"**Tóm tắt từng phần:**\n\n{combined_summary}\n\n"
                     f"---\n**Câu hỏi của tôi:** {question}\n\n"
                     f"Hãy trả lời câu hỏi dựa trên phân tích trên, "
                     f"và đưa ra nhận xét tổng thể về codebase."
                 )},
            ]

            try:
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=final_msgs,
                    stream=True,
                    max_tokens=8192,
                )
                full_reply = []
                for chunk_ in stream:
                    delta = chunk_.choices[0].delta.content
                    if delta:
                        full_reply.append(delta)
                        yield sse({"content": delta})

                # Save to chat
                if chat_id:
                    chats = load_chats()
                    if chat_id in chats:
                        reply_text = "".join(full_reply)
                        chats[chat_id]["messages"].append({
                            "role":    "user",
                            "content": f"[📎 {filename}] {question}",
                            "model":   model_name,
                        })
                        chats[chat_id]["messages"].append({
                            "role":    "assistant",
                            "content": reply_text,
                            "model":   model_name,
                        })
                        chats[chat_id]["updated"] = now_iso()
                        save_chats(chats)

            except Exception as ex:
                yield sse_error(f"Lỗi tổng hợp: {ex}")
                return

        yield sse({"done": True, "chat_id": chat_id})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── Vision (Gemini) ───────────────────────────────────────────

@app.route("/api/vision", methods=["POST"])
def r_vision():
    d          = request.json or {}
    b64_data   = d.get("b64", "")
    mime_type  = d.get("mime", "image/jpeg")
    prompt_txt = d.get("prompt", "Mô tả chi tiết hình ảnh này")
    model_name = d.get("model", "Hnhat Vision")
    chat_id    = d.get("chat_id")
    user_msg   = d.get("user_msg", prompt_txt)

    cfg = load_config()
    gem_key = cfg.get("gemini_key", "").strip()

    if not GEMINI_LIB:
        def _e():
            yield sse_error("Thư viện google-generativeai chưa cài. Chạy: pip install google-generativeai")
        return Response(stream_with_context(_e()), mimetype="text/event-stream")

    if not gem_key:
        def _e():
            yield sse_error("Chưa cài Gemini API Key! Vào ⚙️ Settings để nhập.")
        return Response(stream_with_context(_e()), mimetype="text/event-stream")

    m_info   = MODELS.get(model_name, MODELS["Hnhat Vision"])
    model_id = m_info["id"]

    def generate():
        try:
            genai.configure(api_key=gem_key)
            img_bytes = base64.b64decode(b64_data)

            if PIL_LIB:
                img_obj = PILImage.open(BytesIO(img_bytes))
                content_parts = [prompt_txt, img_obj]
            else:
                content_parts = [
                    {"mime_type": mime_type, "data": img_bytes},
                    prompt_txt,
                ]

            g_model  = genai.GenerativeModel(model_id)
            response = g_model.generate_content(content_parts, stream=True)

            full_reply = []
            for chunk_ in response:
                text_part = getattr(chunk_, "text", None)
                if text_part:
                    full_reply.append(text_part)
                    yield sse({"content": text_part})

            # Save to chat
            if chat_id:
                chats = load_chats()
                if chat_id in chats:
                    chats[chat_id]["messages"].append({
                        "role":    "user",
                        "content": f"[🖼 Ảnh] {user_msg}",
                        "model":   model_name,
                    })
                    chats[chat_id]["messages"].append({
                        "role":    "assistant",
                        "content": "".join(full_reply),
                        "model":   model_name,
                    })
                    chats[chat_id]["updated"] = now_iso()
                    save_chats(chats)

        except Exception as ex:
            yield sse_error(f"Gemini API lỗi: {ex}")
            return

        yield sse({"done": True, "chat_id": chat_id})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )






# ── Serve PNG icon from same folder as script ─────────────────
ICON_PATH = resource_path("hnhatai.png")

@app.route("/favicon.ico")
@app.route("/icon.png")
def r_icon():
    if ICON_PATH.exists():
        return app.response_class(ICON_PATH.read_bytes(), mimetype="image/png")
    # Fallback: 1x1 transparent PNG
    import base64
    px = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    return app.response_class(px, mimetype="image/png")

# ── Prompt Library ────────────────────────────────────────────

@app.route("/api/prompts", methods=["GET"])
def r_prompts_get():
    c = load_config()
    return jsonify(c.get("custom_prompts", []))

@app.route("/api/prompts", methods=["POST"])
def r_prompts_add():
    d = request.json or {}
    c = load_config()
    prompts = c.get("custom_prompts", [])
    prompts.append({
        "id":   str(uuid.uuid4()),
        "name": d.get("name", "Prompt"),
        "text": d.get("text", ""),
        "icon": d.get("icon", "📝"),
    })
    c["custom_prompts"] = prompts
    save_config(c)
    return jsonify({"ok": True})

@app.route("/api/prompts/<pid>", methods=["DELETE"])
def r_prompt_del(pid):
    c = load_config()
    c["custom_prompts"] = [p for p in c.get("custom_prompts", []) if p.get("id") != pid]
    save_config(c)
    return jsonify({"ok": True})

# ── Favorite chats ────────────────────────────────────────────

@app.route("/api/chats/<cid>/favorite", methods=["POST"])
def r_favorite(cid):
    chats = load_chats()
    if cid not in chats:
        return jsonify({"error": "not found"}), 404
    chats[cid]["favorite"] = not chats[cid].get("favorite", False)
    save_chats(chats)
    return jsonify({"favorite": chats[cid]["favorite"]})

# ─────────────────────────────────────────────────────────────
#  HTML TEMPLATE  —  "Stellar Dark" UI  v3.0
# ─────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi" data-theme="dark">
<head>
<meta charset="UTF-8">
<link rel="icon" type="image/png" href="/icon.png">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hnhat AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark-dimmed.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInDocument()"></script>
<style>
/* ═══ CSS VARIABLES ═══════════════════════════════════════════ */
:root {
  /* Backgrounds */
  --bg:         #030309;
  --bg-s:       #07071a;
  --bg-c:       #0c0c22;
  --bg-i:       #10102e;
  --bg-h:       #141440;
  --bg-a:       #1a1a50;

  /* Glass layers */
  --g1: rgba(255,255,255,.03);
  --g2: rgba(255,255,255,.06);
  --g3: rgba(255,255,255,.09);

  /* Borders */
  --b1: rgba(255,255,255,.05);
  --b2: rgba(255,255,255,.09);
  --b3: rgba(255,255,255,.14);

  /* Text */
  --t1: #e8eaf8;
  --t2: #7878a8;
  --t3: #3c3c60;

  /* Brand accents */
  --ac1: #7c3aed;   /* purple  */
  --ac2: #0891b2;   /* cyan    */
  --ac3: #059669;   /* green   */

  /* Groq teal glow */
  --groq: #00d4aa;
  /* Gemini blue */
  --gem:  #4285f4;

  /* Status */
  --ok:  #16a34a;
  --err: #dc2626;
  --wrn: #d97706;

  /* Sizes */
  --sw: 276px;
  --r:  12px;
  --rs: 8px;

  /* Fonts */
  --fui: 'Outfit', sans-serif;
  --fco: 'Fira Code', monospace;
}

[data-theme="light"] {
  /* Soft warm-gray — không chói, dễ nhìn như Notion */
  --bg:   #f5f4f0;
  --bg-s: #eeedea;
  --bg-c: #e8e7e3;
  --bg-i: #deddda;
  --bg-h: #d4d3cf;
  --bg-a: #c9c8c4;
  --g1: rgba(0,0,0,.03);
  --g2: rgba(0,0,0,.055);
  --g3: rgba(0,0,0,.09);
  --b1: rgba(0,0,0,.09);
  --b2: rgba(0,0,0,.14);
  --b3: rgba(0,0,0,.20);
  --t1: #1c1b18;
  --t2: #5a5850;
  --t3: #9a9890;
  --ac1: #4f46e5;
  --ac2: #0284c7;
  --ac3: #059669;
}
/* Light: sidebar và topbar giữ màu riêng */
[data-theme="light"] #sidebar {
  background: #eae9e5 !important;
  border-right-color: rgba(0,0,0,.1) !important;
}
[data-theme="light"] #topbar {
  background: rgba(245,244,240,.92) !important;
  backdrop-filter: blur(20px);
  border-bottom-color: rgba(0,0,0,.09) !important;
}
[data-theme="light"] #input-box {
  background: #fff !important;
  border-color: rgba(0,0,0,.14) !important;
  box-shadow: 0 1px 4px rgba(0,0,0,.08) !important;
}
[data-theme="light"] #input-box:focus-within {
  border-color: rgba(79,70,229,.45) !important;
  box-shadow: 0 0 0 3px rgba(79,70,229,.1), 0 2px 8px rgba(0,0,0,.1) !important;
}
[data-theme="light"] .bubble-user {
  background: linear-gradient(135deg,rgba(79,70,229,.14),rgba(2,132,199,.10)) !important;
  border-color: rgba(79,70,229,.22) !important;
}
[data-theme="light"] .msg-content { color: #1c1b18 !important; }
[data-theme="light"] .msg-content strong { color: #111 !important; }
[data-theme="light"] .msg-content code:not(pre code) {
  background: rgba(79,70,229,.08) !important;
  border-color: rgba(79,70,229,.18) !important;
  color: #4f46e5 !important;
}
[data-theme="light"] .msg-content pre {
  border-color: rgba(0,0,0,.12) !important;
  background: #1e1e2e !important; /* dark code block even in light mode */
}
[data-theme="light"] .code-header {
  background: rgba(0,0,0,.06) !important;
  border-bottom-color: rgba(0,0,0,.08) !important;
}
[data-theme="light"] .model-dd {
  background: #eae9e5 !important;
  border-color: rgba(0,0,0,.14) !important;
  box-shadow: 0 12px 40px rgba(0,0,0,.15), 0 2px 8px rgba(0,0,0,.08) !important;
}
[data-theme="light"] .model-opt:hover  { background: rgba(79,70,229,.08) !important; }
[data-theme="light"] .model-opt.selected{ background: rgba(79,70,229,.14) !important; }
[data-theme="light"] .model-pill {
  background: rgba(79,70,229,.08) !important;
  border-color: rgba(79,70,229,.22) !important;
}
[data-theme="light"] .model-pill:hover {
  background: rgba(79,70,229,.14) !important;
}
[data-theme="light"] .chat-item:hover  { background: rgba(0,0,0,.05) !important; }
[data-theme="light"] .chat-item.active { background: rgba(79,70,229,.12) !important; }
[data-theme="light"] .new-btn {
  background: rgba(79,70,229,.08) !important;
  border-color: rgba(79,70,229,.28) !important;
}
[data-theme="light"] .new-btn:hover {
  background: rgba(79,70,229,.16) !important;
  border-color: rgba(79,70,229,.5) !important;
}
[data-theme="light"] .sf-btn:hover {
  background: rgba(79,70,229,.1) !important;
  border-color: rgba(79,70,229,.3) !important;
}
[data-theme="light"] .chip {
  background: rgba(255,255,255,.7) !important;
  border-color: rgba(0,0,0,.12) !important;
  color: #3a3830 !important;
  backdrop-filter: blur(8px);
}
[data-theme="light"] .chip:hover {
  background: rgba(79,70,229,.12) !important;
  border-color: rgba(79,70,229,.35) !important;
  color: #1c1b18 !important;
}
[data-theme="light"] .modal-card,
[data-theme="light"] .pm-card,
[data-theme="light"] .stats-card,
[data-theme="light"] .shortcuts-card {
  background: #f0efe9 !important;
  border-color: rgba(0,0,0,.14) !important;
  box-shadow: 0 16px 50px rgba(0,0,0,.18) !important;
}
[data-theme="light"] .form-input,
[data-theme="light"] .form-textarea,
[data-theme="light"] .form-select {
  background: #fff !important;
  border-color: rgba(0,0,0,.14) !important;
  color: #1c1b18 !important;
}
[data-theme="light"] .form-input:focus,
[data-theme="light"] .form-textarea:focus {
  border-color: rgba(79,70,229,.5) !important;
  box-shadow: 0 0 0 3px rgba(79,70,229,.1) !important;
}
[data-theme="light"] .search-box {
  background: rgba(255,255,255,.8) !important;
  border-color: rgba(0,0,0,.12) !important;
}
[data-theme="light"] #send-btn {
  background: linear-gradient(135deg,#4f46e5,#0284c7) !important;
}
[data-theme="light"] .save-btn {
  background: linear-gradient(135deg,#4f46e5,#0284c7) !important;
}
[data-theme="light"] .w-title {
  background: linear-gradient(110deg,#1c1b18 10%,#4f46e5 55%,#0284c7 100%) !important;
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
}
[data-theme="light"] .w-sub { color: #5a5850 !important; }
[data-theme="light"] body::before {
  background: radial-gradient(circle,rgba(79,70,229,.25),transparent 70%) !important;
  opacity: .18 !important;
}
[data-theme="light"] body::after {
  background: radial-gradient(circle,rgba(2,132,199,.2),transparent 70%) !important;
  opacity: .14 !important;
}
[data-theme="light"] .katex { color: #1c1b18 !important; }
[data-theme="light"] .katex-display {
  background: rgba(79,70,229,.05) !important;
  border-color: rgba(0,0,0,.1) !important;
  border-left-color: #4f46e5 !important;
}
[data-theme="light"] .progress-wrap {
  background: #fff !important;
  border-color: rgba(0,0,0,.1) !important;
}
[data-theme="light"] .stat-box {
  background: #fff !important;
  border-color: rgba(0,0,0,.1) !important;
}
[data-theme="light"] .kbd {
  background: #fff !important;
  border-color: rgba(0,0,0,.18) !important;
  color: #1c1b18 !important;
}
[data-theme="light"] #toast {
  background: #f0efe9 !important;
  color: #1c1b18 !important;
}
/* scrollbar in light mode */
[data-theme="light"] ::-webkit-scrollbar-thumb {
  background: rgba(0,0,0,.18) !important;
}
[data-theme="light"] ::-webkit-scrollbar-thumb:hover {
  background: rgba(79,70,229,.4) !important;
}

/* ═══ RESET + BASE ═══════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 15px; -webkit-font-smoothing: antialiased; }
body {
  font-family: var(--fui);
  background: var(--bg);
  color: var(--t1);
  display: flex;
  height: 100vh;
  overflow: hidden;
  position: relative;
}

/* Ambient orbs */
body::before, body::after {
  content: '';
  position: fixed;
  border-radius: 50%;
  filter: blur(140px);
  pointer-events: none;
  z-index: 0;
  opacity: .12;
}
body::before {
  width: 700px; height: 700px;
  top: -300px; left: -200px;
  background: radial-gradient(circle, var(--ac1), transparent 70%);
  animation: orbA 22s ease-in-out infinite alternate;
}
body::after {
  width: 600px; height: 600px;
  bottom: -250px; right: -150px;
  background: radial-gradient(circle, var(--ac2), transparent 70%);
  animation: orbB 28s ease-in-out infinite alternate;
}
@keyframes orbA { from { transform: translate(0,0); } to { transform: translate(60px,40px); } }
@keyframes orbB { from { transform: translate(0,0); } to { transform: translate(-50px,-30px); } }

/* Scrollbar */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg-h); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--t3); }

/* ═══ SIDEBAR ═══════════════════════════════════════════════ */
#sidebar {
  width: var(--sw);
  min-width: var(--sw);
  background: var(--bg-s);
  border-right: 1px solid var(--b1);
  display: flex;
  flex-direction: column;
  position: relative;
  z-index: 10;
  transition: transform .3s cubic-bezier(.4,0,.2,1), min-width .3s, width .3s;
  overflow: hidden;
}
#sidebar.collapsed {
  transform: translateX(calc(-1 * var(--sw)));
  min-width: 0;
  width: 0;
  border: none;
}

/* Sidebar top */
.s-top {
  padding: 18px 14px 12px;
  border-bottom: 1px solid var(--b1);
  flex-shrink: 0;
}
.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
  user-select: none;
}
.logo-gem {
  width: 40px; height: 40px;
  border-radius: 12px;
  flex-shrink: 0;
  background: linear-gradient(135deg, var(--ac1), var(--ac2));
  display: flex; align-items: center; justify-content: center;
  font-size: 20px;
  box-shadow: 0 0 24px rgba(124,58,237,.5), 0 0 50px rgba(8,145,178,.2);
  animation: logoGlow 4s ease-in-out infinite;
}
@keyframes logoGlow {
  0%, 100% { box-shadow: 0 0 24px rgba(124,58,237,.5), 0 0 50px rgba(8,145,178,.2); }
  50%       { box-shadow: 0 0 38px rgba(124,58,237,.72), 0 0 75px rgba(8,145,178,.35); }
}
.logo-text { flex: 1; }
.logo-name {
  font-weight: 800;
  font-size: 1.1rem;
  background: linear-gradient(110deg, #fff 20%, var(--ac2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  line-height: 1.1;
}
.logo-ver {
  font-size: .62rem;
  color: var(--t3);
  letter-spacing: .04em;
  margin-top: 2px;
  display: block;
}

/* New chat button */
.new-btn {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 13px;
  background: var(--g2);
  border: 1px solid var(--b2);
  border-radius: var(--rs);
  color: var(--t1);
  font: 500 .82rem var(--fui);
  cursor: pointer;
  transition: .2s;
  position: relative;
  overflow: hidden;
}
.new-btn::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(135deg, rgba(124,58,237,.15), rgba(8,145,178,.1));
  opacity: 0;
  transition: opacity .2s;
}
.new-btn:hover { border-color: rgba(124,58,237,.42); transform: translateY(-1px); box-shadow: 0 4px 18px rgba(124,58,237,.2); }
.new-btn:hover::before { opacity: 1; }
.new-ic {
  width: 20px; height: 20px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--ac1), var(--ac2));
  display: flex; align-items: center; justify-content: center;
  font-size: 13px;
  flex-shrink: 0;
}

/* Search */
.s-search { padding: 10px 14px 5px; flex-shrink: 0; }
.search-box {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  background: var(--bg-c);
  border: 1px solid var(--b1);
  border-radius: var(--rs);
  transition: .2s;
}
.search-box:focus-within { border-color: rgba(124,58,237,.4); }
.search-box span { font-size: 11px; color: var(--t3); }
.search-input {
  flex: 1;
  background: none;
  border: none;
  outline: none;
  color: var(--t1);
  font: .79rem var(--fui);
}
.search-input::placeholder { color: var(--t3); }

/* Chat list */
.cl-label {
  font: 600 .66rem var(--fui);
  color: var(--t3);
  text-transform: uppercase;
  letter-spacing: .09em;
  padding: 10px 14px 4px;
  flex-shrink: 0;
}
#chat-list { flex: 1; overflow-y: auto; padding: 2px 8px 8px; }

.chat-item {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 8px 9px;
  border-radius: var(--rs);
  cursor: pointer;
  transition: .14s;
  position: relative;
}
.chat-item:hover { background: var(--bg-c); }
.chat-item.active { background: var(--bg-i); }
.chat-item.active::before {
  content: '';
  position: absolute;
  left: 0; top: 5px; bottom: 5px;
  width: 2px;
  border-radius: 0 2px 2px 0;
  background: linear-gradient(to bottom, var(--ac1), var(--ac2));
}
.ci-icon { font-size: 12px; flex-shrink: 0; opacity: .45; }
.ci-body { flex: 1; min-width: 0; }
.ci-title {
  font: .78rem var(--fui);
  color: var(--t1);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ci-meta { font: .67rem var(--fui); color: var(--t3); margin-top: 2px; }
.ci-del {
  opacity: 0;
  background: none;
  border: none;
  color: var(--t3);
  cursor: pointer;
  padding: 3px 5px;
  border-radius: 4px;
  font-size: 10px;
  transition: .13s;
  flex-shrink: 0;
}
.chat-item:hover .ci-del { opacity: 1; }
.ci-del:hover { color: var(--err); background: rgba(220,38,38,.12); }

/* Sidebar footer */
.s-foot {
  padding: 9px;
  border-top: 1px solid var(--b1);
  flex-shrink: 0;
  display: flex;
  gap: 5px;
}
.sf-btn {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  padding: 7px 6px;
  background: var(--g1);
  border: 1px solid var(--b1);
  border-radius: var(--rs);
  color: var(--t2);
  font: .72rem var(--fui);
  cursor: pointer;
  transition: .14s;
  white-space: nowrap;
}
.sf-btn:hover { background: var(--g2); color: var(--t1); border-color: var(--b2); }

/* ═══ MAIN ══════════════════════════════════════════════════ */
#main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: relative;
  z-index: 1;
}

/* Topbar */
#topbar {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 11px 16px;
  border-bottom: 1px solid var(--b1);
  flex-shrink: 0;
  background: rgba(3,3,9,.8);
  backdrop-filter: blur(20px);
  position: relative;
  z-index: 5;
}
.tb-btn {
  width: 32px; height: 32px;
  display: flex; align-items: center; justify-content: center;
  background: var(--g1);
  border: 1px solid var(--b1);
  border-radius: var(--rs);
  color: var(--t2);
  cursor: pointer;
  font-size: 13px;
  transition: .14s;
  flex-shrink: 0;
}
.tb-btn:hover { background: var(--g2); color: var(--t1); border-color: var(--b2); }

#chat-title {
  font: 600 .87rem var(--fui);
  color: var(--t1);
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  border-radius: 5px;
  padding: 2px 5px;
  transition: .15s;
  cursor: text;
}
#chat-title:hover { background: var(--g1); }
#chat-title[contenteditable="true"] {
  outline: 1px solid rgba(124,58,237,.4);
  background: var(--bg-c);
  padding: 2px 7px;
}

.tb-actions { display: flex; gap: 5px; flex-shrink: 0; }

/* ── Model pill ── */
.model-pill {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 11px;
  background: var(--bg-c);
  border: 1px solid var(--b2);
  border-radius: 99px;
  cursor: pointer;
  transition: .2s;
  position: relative;
  flex-shrink: 0;
  user-select: none;
}
.model-pill:hover { border-color: var(--b3); background: var(--bg-i); }
.model-pill.open { border-color: rgba(124,58,237,.45); box-shadow: 0 0 16px rgba(124,58,237,.18); }
.pill-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  transition: .2s;
}
.pill-name { font: 500 .79rem var(--fui); color: var(--t1); }
.pill-arrow { font-size: .52rem; color: var(--t3); transition: transform .2s; }
.model-pill.open .pill-arrow { transform: rotate(180deg); }

/* Model dropdown */
.model-dd {
  position: absolute;
  top: calc(100% + 10px);
  right: 0;
  width: 282px;
  background: var(--bg-c);
  border: 1px solid var(--b3);
  border-radius: var(--r);
  padding: 6px;
  box-shadow: 0 24px 70px rgba(0,0,0,.7), 0 0 36px rgba(124,58,237,.14);
  display: none;
  z-index: 300;
}
.model-dd.visible { display: block; animation: ddIn .18s ease; }
@keyframes ddIn {
  from { opacity: 0; transform: translateY(-6px) scale(.97); }
  to   { opacity: 1; transform: none; }
}
.dd-section { display: none; }
.dd-divider { height: 1px; background: var(--b1); margin: 5px 0; }

.model-opt {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 8px 10px;
  border-radius: var(--rs);
  cursor: pointer;
  transition: .12s;
}
.model-opt:hover { background: var(--bg-i); }
.model-opt.selected { background: rgba(124,58,237,.1); }
.mopt-icon { font-size: 16px; width: 24px; text-align: center; flex-shrink: 0; }
.mopt-body { flex: 1; }
.mopt-name { font: 600 .82rem var(--fui); color: var(--t1); }
.mopt-desc { font: .68rem var(--fui); color: var(--t3); margin-top: 1px; }
.mopt-meta { display: flex; align-items: center; gap: 5px; margin-top: 2px; }
.mopt-badge {
  font: 500 .58rem var(--fco);
  padding: 1px 6px;
  border-radius: 99px;
}
.mopt-ctx { font: .6rem var(--fco); color: var(--t3); }
.api-tag {
  font: 500 .58rem var(--fco);
  padding: 1px 5px;
  border-radius: 4px;
}
.api-groq  { display: none; }
.api-gemini{ display: none; }

/* ═══ CHAT AREA ═════════════════════════════════════════════ */
#chat-area { flex: 1; overflow-y: auto; position: relative; }

/* Welcome screen */
#welcome {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100%;
  padding: 40px 20px;
  text-align: center;
}
.w-orb {
  width: 74px; height: 74px;
  border-radius: 22px;
  background: linear-gradient(135deg, var(--ac1), var(--ac2));
  display: flex; align-items: center; justify-content: center;
  font-size: 35px;
  margin-bottom: 22px;
  box-shadow: 0 0 55px rgba(124,58,237,.5), 0 0 100px rgba(8,145,178,.25);
  animation: wOrbFloat 5s ease-in-out infinite;
}
@keyframes wOrbFloat {
  0%, 100% { transform: translateY(0) rotate(0deg); }
  50%       { transform: translateY(-10px) rotate(2deg); }
}
.w-title {
  font: 800 2.4rem var(--fui);
  line-height: 1.1;
  background: linear-gradient(110deg, #fff 10%, var(--ac1) 50%, var(--ac2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 10px;
}
.w-sub {
  color: var(--t2);
  font: .9rem/1.65 var(--fui);
  max-width: 450px;
  margin-bottom: 36px;
}
.w-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  max-width: 700px;
}
.chip {
  padding: 7px 15px;
  background: var(--g2);
  border: 1px solid var(--b1);
  border-radius: 99px;
  font: .79rem var(--fui);
  color: var(--t2);
  cursor: pointer;
  transition: .2s;
  white-space: nowrap;
  backdrop-filter: blur(10px);
}
.chip:hover {
  background: var(--bg-h);
  border-color: var(--b3);
  color: var(--t1);
  transform: translateY(-2px);
  box-shadow: 0 5px 20px rgba(124,58,237,.18);
}

/* Messages container */
#messages {
  padding: 16px 0 120px;
  max-width: 860px;
  margin: 0 auto;
  width: 100%;
}

/* Message group */
.msg-group {
  padding: 6px 20px;
  animation: msgIn .27s ease;
}
@keyframes msgIn {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: none; }
}
.msg-group.user { display: flex; justify-content: flex-end; }

/* User bubble */
.bubble-user {
  max-width: 70%;
  padding: 11px 15px;
  border-radius: 18px 18px 4px 18px;
  background: linear-gradient(135deg, rgba(124,58,237,.24), rgba(8,145,178,.15));
  border: 1px solid rgba(124,58,237,.3);
  font: .87rem/1.7 var(--fui);
  word-break: break-word;
}
.bubble-ts {
  font: .62rem var(--fco);
  color: var(--t3);
  margin-top: 5px;
  text-align: right;
}

/* AI response */
.bubble-ai { max-width: 100%; }
.ai-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 9px;
}
.ai-avatar {
  width: 28px; height: 28px;
  border-radius: 9px;
  flex-shrink: 0;
  font-size: 13px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, var(--ac1), var(--ac2));
}
.ai-who { font: 600 .77rem var(--fui); color: var(--t2); }
.ai-model-tag {
  font: .62rem var(--fco);
  padding: 2px 8px;
  background: var(--bg-c);
  border: 1px solid var(--b1);
  border-radius: 99px;
  color: var(--t3);
}
.ai-timestamp { font: .62rem var(--fco); color: var(--t3); margin-left: auto; }

/* Markdown content */
.msg-content { font: .87rem/1.78 var(--fui); color: var(--t1); }
.msg-content p { margin: 0 0 .68em; }
.msg-content p:last-child { margin: 0; }
.msg-content h1, .msg-content h2, .msg-content h3 {
  font: 700 var(--fui);
  margin: 1.1em 0 .44em;
  color: #fff;
}
.msg-content h1 { font-size: 1.22rem; }
.msg-content h2 { font-size: 1.03rem; }
.msg-content h3 { font-size: .92rem; }
.msg-content ul, .msg-content ol {
  padding-left: 1.35em;
  margin: .4em 0;
}
.msg-content li { margin: .22em 0; }
.msg-content strong { font-weight: 700; color: #fff; }
.msg-content em { color: var(--t2); font-style: italic; }
.msg-content a { color: var(--ac2); text-decoration: none; }
.msg-content a:hover { text-decoration: underline; }
.msg-content hr { border: none; border-top: 1px solid var(--b1); margin: .8em 0; }
.msg-content blockquote {
  border-left: 3px solid var(--ac1);
  padding-left: 12px;
  color: var(--t2);
  margin: .55em 0;
  font-style: italic;
}
.msg-content table { width: 100%; border-collapse: collapse; margin: .65em 0; font-size: .82rem; }
.msg-content th, .msg-content td { padding: 7px 11px; border: 1px solid var(--b2); text-align: left; }
.msg-content th { background: var(--bg-c); font-weight: 600; }
.msg-content tr:nth-child(even) td { background: rgba(255,255,255,.02); }

/* Code blocks */
.msg-content pre {
  margin: .7em 0;
  border-radius: var(--r);
  overflow: hidden;
  border: 1px solid var(--b2);
  background: #0d1117 !important;
}
.code-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 7px 14px;
  background: rgba(255,255,255,.04);
  border-bottom: 1px solid var(--b1);
}
.code-lang { font: .67rem var(--fco); color: var(--t3); text-transform: uppercase; letter-spacing: .07em; }
.code-copy {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 3px 9px;
  background: rgba(255,255,255,.05);
  border: 1px solid var(--b1);
  border-radius: 5px;
  color: var(--t2);
  font: .67rem var(--fui);
  cursor: pointer;
  transition: .13s;
}
.code-copy:hover { background: rgba(255,255,255,.1); color: var(--t1); }
.code-copy.copied { color: var(--ok); border-color: rgba(22,163,74,.3); }
.msg-content pre code {
  padding: 14px 16px !important;
  font: .8rem/1.62 var(--fco) !important;
  display: block;
  overflow-x: auto;
}
.msg-content code:not(pre code) {
  background: var(--bg-a);
  border: 1px solid var(--b1);
  padding: 1px 5px;
  border-radius: 4px;
  font: .8em var(--fco);
  color: #c4b5fd;
}

/* Message actions */
.msg-actions {
  display: flex;
  gap: 4px;
  margin-top: 7px;
  opacity: 0;
  transition: opacity .2s;
  flex-wrap: wrap;
}
.msg-group:hover .msg-actions { opacity: 1; }
.act-btn {
  display: flex;
  align-items: center;
  gap: 3px;
  padding: 4px 9px;
  background: none;
  border: 1px solid var(--b1);
  border-radius: 6px;
  color: var(--t3);
  font: .69rem var(--fui);
  cursor: pointer;
  transition: .13s;
}
.act-btn:hover { background: var(--bg-c); color: var(--t1); border-color: var(--b2); }

/* Thinking indicator */
.thinking-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 9px;
}
.thinking-dots { display: flex; gap: 4px; }
.thinking-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--ac1);
  animation: tdot 1.3s ease-in-out infinite;
}
.thinking-dot:nth-child(2) { animation-delay: .22s; background: color-mix(in srgb, var(--ac1) 60%, var(--ac2)); }
.thinking-dot:nth-child(3) { animation-delay: .44s; background: var(--ac2); }
@keyframes tdot {
  0%, 80%, 100% { transform: scale(.55); opacity: .3; }
  40%           { transform: scale(1); opacity: 1; }
}
.thinking-text {
  font: .82rem var(--fui);
  color: var(--t3);
  font-style: italic;
}

/* Progress bar (chunked analysis) */
.progress-wrap {
  margin: 8px 0;
  padding: 10px 14px;
  background: var(--bg-c);
  border: 1px solid var(--b1);
  border-radius: var(--rs);
}
.progress-label { font: .79rem var(--fui); color: var(--t2); margin-bottom: 7px; }
.progress-bar-bg {
  height: 4px;
  background: var(--bg-i);
  border-radius: 4px;
  overflow: hidden;
}
.progress-bar-fill {
  height: 100%;
  border-radius: 4px;
  background: linear-gradient(90deg, var(--ac1), var(--ac2));
  transition: width .4s ease;
}
.progress-pct { font: .69rem var(--fco); color: var(--t3); margin-top: 4px; text-align: right; }

/* Info / status line */
.info-line {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: rgba(8,145,178,.08);
  border: 1px solid rgba(8,145,178,.18);
  border-radius: var(--rs);
  font: .78rem var(--fui);
  color: var(--ac2);
  margin-bottom: 8px;
}

/* ═══ INPUT AREA ════════════════════════════════════════════ */
#input-wrap {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  padding: 11px 18px 17px;
  background: linear-gradient(to top, var(--bg) 65%, transparent);
}
#input-box {
  max-width: 860px;
  margin: 0 auto;
  background: var(--bg-c);
  border: 1px solid var(--b2);
  border-radius: 18px;
  padding: 8px 9px 8px 15px;
  transition: .2s;
}
#input-box:focus-within {
  border-color: rgba(124,58,237,.48);
  box-shadow: 0 0 0 3px rgba(124,58,237,.12), 0 12px 40px rgba(0,0,0,.3);
}

/* Attached file preview */
#attach-preview { display: none; margin-bottom: 8px; padding: 0 2px; }
.attach-chip {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 5px 11px;
  background: var(--bg-i);
  border: 1px solid var(--b2);
  border-radius: var(--rs);
  font: .77rem var(--fui);
  color: var(--t1);
  max-width: 100%;
}
.attach-chip img {
  width: 26px; height: 26px;
  object-fit: cover;
  border-radius: 4px;
}
.attach-info { flex: 1; min-width: 0; }
.attach-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: .78rem; }
.attach-meta { font: .67rem var(--fco); color: var(--t3); margin-top: 1px; }
.attach-large { color: var(--wrn); font-size: .67rem; }
.attach-rm {
  background: none; border: none;
  color: var(--t3); cursor: pointer;
  font-size: 11px; transition: .13s;
  flex-shrink: 0; padding: 2px 4px;
  border-radius: 4px;
}
.attach-rm:hover { color: var(--err); background: rgba(220,38,38,.1); }

.input-row { display: flex; align-items: flex-end; gap: 7px; }
#user-input {
  flex: 1;
  background: none;
  border: none;
  outline: none;
  color: var(--t1);
  font: .87rem/1.6 var(--fui);
  resize: none;
  max-height: 180px;
  min-height: 22px;
  overflow-y: auto;
  padding: 2px 0;
}
#user-input::placeholder { color: var(--t3); }

.input-actions { display: flex; align-items: center; gap: 4px; flex-shrink: 0; }
.i-btn {
  width: 31px; height: 31px;
  display: flex; align-items: center; justify-content: center;
  background: none;
  border: 1px solid var(--b1);
  border-radius: var(--rs);
  color: var(--t3);
  cursor: pointer;
  font-size: 13px;
  transition: .13s;
}
.i-btn:hover { background: var(--bg-i); color: var(--t1); border-color: var(--b2); }
#file-input { display: none; }

#send-btn {
  width: 36px; height: 36px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, var(--ac1), var(--ac2));
  border: none;
  border-radius: 10px;
  color: #fff;
  cursor: pointer;
  font-size: 16px;
  transition: .2s;
  box-shadow: 0 4px 16px rgba(124,58,237,.45);
  flex-shrink: 0;
}
#send-btn:hover { transform: scale(1.08); box-shadow: 0 6px 24px rgba(124,58,237,.6); }
#send-btn:active { transform: scale(.95); }
#send-btn:disabled { opacity: .4; cursor: not-allowed; transform: none; box-shadow: none; }

.input-footer {
  max-width: 860px;
  margin: 5px auto 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 3px;
}
.input-hint { font: .64rem var(--fui); color: var(--t3); }
#token-count { font: .64rem var(--fco); color: var(--t3); }

/* ═══ DRAG OVERLAY ══════════════════════════════════════════ */
#drag-overlay {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 500;
  background: rgba(3,3,9,.85);
  backdrop-filter: blur(12px);
  align-items: center;
  justify-content: center;
  flex-direction: column;
}
#drag-overlay.visible { display: flex; }
.drag-box {
  border: 2px dashed rgba(124,58,237,.55);
  border-radius: 28px;
  padding: 48px 68px;
  text-align: center;
  animation: dBorder 2.2s ease-in-out infinite;
}
@keyframes dBorder {
  0%, 100% { border-color: rgba(124,58,237,.55); }
  50%       { border-color: rgba(8,145,178,.75); }
}
.drag-icon { font-size: 52px; margin-bottom: 14px; }
.drag-title {
  font: 700 1.3rem var(--fui);
  background: linear-gradient(110deg, var(--ac1), var(--ac2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.drag-sub { font: .82rem var(--fui); color: var(--t2); margin-top: 6px; }

/* ═══ SETTINGS MODAL ════════════════════════════════════════ */
#settings-modal {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 400;
  align-items: center;
  justify-content: center;
  padding: 20px;
}
#settings-modal.open { display: flex; }
.modal-backdrop {
  position: absolute; inset: 0;
  background: rgba(0,0,0,.78);
  backdrop-filter: blur(12px);
}
.modal-card {
  position: relative;
  width: 100%;
  max-width: 545px;
  background: var(--bg-s);
  border: 1px solid var(--b3);
  border-radius: 22px;
  padding: 28px 28px 24px;
  box-shadow: 0 32px 90px rgba(0,0,0,.75), 0 0 60px rgba(124,58,237,.14);
  animation: modalIn .22s ease;
  max-height: 90vh;
  overflow-y: auto;
}
@keyframes modalIn {
  from { opacity: 0; transform: scale(.93) translateY(14px); }
  to   { opacity: 1; transform: none; }
}
.modal-title {
  font: 700 1.08rem var(--fui);
  margin-bottom: 22px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.modal-close {
  position: absolute; top: 16px; right: 16px;
  width: 28px; height: 28px;
  display: flex; align-items: center; justify-content: center;
  background: var(--bg-c);
  border: 1px solid var(--b1);
  border-radius: var(--rs);
  color: var(--t2);
  cursor: pointer;
  font-size: 11px;
  transition: .13s;
}
.modal-close:hover { background: var(--bg-i); color: var(--t1); }

.form-group { margin-bottom: 16px; }
.form-label {
  display: block;
  font: 500 .77rem var(--fui);
  color: var(--t2);
  margin-bottom: 5px;
}
.form-input, .form-textarea, .form-select {
  width: 100%;
  background: var(--bg-c);
  border: 1px solid var(--b1);
  border-radius: var(--rs);
  padding: 9px 13px;
  color: var(--t1);
  font: .84rem var(--fui);
  outline: none;
  transition: .2s;
}
.form-input:focus, .form-textarea:focus, .form-select:focus {
  border-color: rgba(124,58,237,.5);
  box-shadow: 0 0 0 3px rgba(124,58,237,.1);
}
.form-textarea { resize: vertical; min-height: 88px; line-height: 1.5; }
.form-select option { background: var(--bg-c); }
.key-row { display: flex; gap: 7px; }
.key-row .form-input { font-family: var(--fco); font-size: .77rem; }
.form-hint { font: .69rem var(--fui); color: var(--t3); margin-top: 4px; line-height: 1.4; }
.form-hint a { color: var(--ac2); text-decoration: none; }
.form-hint a:hover { text-decoration: underline; }
.key-status {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font: .71rem var(--fui);
}
.status-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  display: inline-block;
}
.status-dot.ok { background: var(--ok); box-shadow: 0 0 5px var(--ok); }
.status-dot.no { background: var(--err); box-shadow: 0 0 5px var(--err); }

.save-btn {
  width: 100%;
  padding: 11px;
  background: linear-gradient(135deg, var(--ac1), var(--ac2));
  border: none;
  border-radius: var(--rs);
  color: #fff;
  font: 600 .88rem var(--fui);
  cursor: pointer;
  transition: .2s;
  box-shadow: 0 4px 18px rgba(124,58,237,.38);
  margin-top: 6px;
}
.save-btn:hover { transform: translateY(-1px); box-shadow: 0 8px 28px rgba(124,58,237,.55); }

/* Key info section */
.key-section {
  margin-bottom: 20px;
  padding: 14px;
  background: var(--bg-c);
  border: 1px solid var(--b1);
  border-radius: var(--r);
}
.key-section-title {
  font: 600 .82rem var(--fui);
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
}

/* ═══ SHORTCUTS MODAL ═══════════════════════════════════════ */
#shortcuts-modal {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 400;
  align-items: center;
  justify-content: center;
  padding: 20px;
}
#shortcuts-modal.open { display: flex; }

.shortcuts-card {
  position: relative;
  width: 100%;
  max-width: 460px;
  background: var(--bg-s);
  border: 1px solid var(--b3);
  border-radius: 20px;
  padding: 24px;
  box-shadow: 0 24px 70px rgba(0,0,0,.7);
  animation: modalIn .22s ease;
  max-height: 80vh;
  overflow-y: auto;
}
.sc-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid var(--b1);
}
.sc-row:last-child { border-bottom: none; }
.sc-label { font: .82rem var(--fui); color: var(--t2); }
.sc-key {
  display: flex;
  gap: 4px;
}
.kbd {
  padding: 2px 7px;
  background: var(--bg-i);
  border: 1px solid var(--b2);
  border-radius: 4px;
  font: .7rem var(--fco);
  color: var(--t1);
}

/* ═══ TOAST ══════════════════════════════════════════════════ */
#toast {
  position: fixed;
  bottom: 88px;
  left: 50%;
  transform: translateX(-50%) translateY(12px);
  background: var(--bg-c);
  border: 1px solid var(--b3);
  border-radius: 11px;
  padding: 9px 18px;
  font: .79rem var(--fui);
  color: var(--t1);
  box-shadow: 0 12px 40px rgba(0,0,0,.5);
  opacity: 0;
  transition: .24s;
  z-index: 700;
  pointer-events: none;
  white-space: nowrap;
  max-width: 90vw;
}
#toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }

/* ═══ RESPONSIVE ════════════════════════════════════════════ */
@media (max-width: 640px) {
  #sidebar { position: absolute; height: 100%; }
  .bubble-user { max-width: 90%; }
  .w-title { font-size: 2rem; }
  #input-wrap { padding: 10px 12px 13px; }
  .model-dd { right: -10px; width: 270px; }
}

/* ═══ EXTRA THEMES ══════════════════════════════════════════ */
[data-theme="midnight"] {
  --bg:#05071a;--bg-s:#090c26;--bg-c:#0d1032;--bg-i:#11153e;--bg-h:#161b4a;--bg-a:#1c2258;
  --g1:rgba(255,255,255,.03);--g2:rgba(255,255,255,.06);--g3:rgba(255,255,255,.09);
  --b1:rgba(100,120,255,.1);--b2:rgba(100,120,255,.16);--b3:rgba(100,120,255,.24);
  --t1:#dde4ff;--t2:#7080c0;--t3:#363880;
  --ac1:#6366f1;--ac2:#a78bfa;--ac3:#38bdf8;
}
[data-theme="ocean"] {
  --bg:#010f12;--bg-s:#031418;--bg-c:#041c22;--bg-i:#06242c;--bg-h:#082e38;--bg-a:#0a3842;
  --g1:rgba(0,210,190,.03);--g2:rgba(0,210,190,.06);--g3:rgba(0,210,190,.09);
  --b1:rgba(0,210,190,.1);--b2:rgba(0,210,190,.16);--b3:rgba(0,210,190,.24);
  --t1:#ccf5f0;--t2:#4a9090;--t3:#1e4848;
  --ac1:#0d9488;--ac2:#06b6d4;--ac3:#22d3ee;
}
[data-theme="rose"] {
  --bg:#0f0508;--bg-s:#180810;--bg-c:#200c16;--bg-i:#2a1020;--bg-h:#34142a;--bg-a:#3e1934;
  --g1:rgba(255,80,120,.03);--g2:rgba(255,80,120,.06);--g3:rgba(255,80,120,.09);
  --b1:rgba(255,80,120,.1);--b2:rgba(255,80,120,.16);--b3:rgba(255,80,120,.24);
  --t1:#ffe4ec;--t2:#c06080;--t3:#602040;
  --ac1:#e11d48;--ac2:#f43f5e;--ac3:#fb7185;
}
[data-theme="forest"] {
  --bg:#020d04;--bg-s:#051208;--bg-c:#08190c;--bg-i:#0b2010;--bg-h:#0e2814;--bg-a:#123018;
  --g1:rgba(0,200,80,.03);--g2:rgba(0,200,80,.06);--g3:rgba(0,200,80,.09);
  --b1:rgba(0,200,80,.1);--b2:rgba(0,200,80,.16);--b3:rgba(0,200,80,.24);
  --t1:#d4f5dc;--t2:#4a9060;--t3:#1e4830;
  --ac1:#16a34a;--ac2:#22c55e;--ac3:#4ade80;
}

/* ═══ VOICE BUTTON ══════════════════════════════════════════ */
.voice-btn {
  width:31px;height:31px;
  display:flex;align-items:center;justify-content:center;
  background:none;border:1px solid var(--b1);border-radius:8px;
  color:var(--t3);cursor:pointer;font-size:14px;transition:.15s;
}
.voice-btn:hover{background:var(--bg-i);color:var(--t1);border-color:var(--b2);}
.voice-btn.listening{
  background:rgba(220,38,38,.15);border-color:rgba(220,38,38,.4);
  color:#f87171;
  animation:pulse-mic .8s ease-in-out infinite;
}
@keyframes pulse-mic{0%,100%{box-shadow:0 0 0 0 rgba(220,38,38,.4);}50%{box-shadow:0 0 0 6px rgba(220,38,38,.0);}}

/* ═══ SEARCH OVERLAY ════════════════════════════════════════ */
#search-bar {
  display:none;
  position:absolute;
  top:0;left:0;right:0;
  z-index:20;
  padding:10px 18px;
  background:rgba(3,3,9,.92);
  backdrop-filter:blur(16px);
  border-bottom:1px solid var(--b2);
  align-items:center;
  gap:10px;
  animation:slideDown .2s ease;
}
#search-bar.visible{display:flex;}
@keyframes slideDown{from{transform:translateY(-100%);opacity:0;}to{transform:none;opacity:1;}}
.sb-input{
  flex:1;background:var(--bg-c);border:1px solid var(--b2);border-radius:var(--rs);
  padding:8px 13px;color:var(--t1);font:.87rem var(--fui);outline:none;transition:.2s;
}
.sb-input:focus{border-color:rgba(124,58,237,.5);}
.sb-nav{display:flex;gap:5px;align-items:center;}
.sb-count{font:.72rem var(--fco);color:var(--t3);white-space:nowrap;}
mark.highlight-match{
  background:rgba(250,204,21,.32);color:inherit;
  border-radius:3px;padding:1px 2px;
  text-decoration:none;
}
mark.highlight-match.current{
  background:rgba(250,204,21,.72);
  outline:2px solid rgba(250,204,21,.8);
}

/* ═══ PROMPT LIBRARY MODAL ═══════════════════════════════════ */
#prompt-modal{
  display:none;position:fixed;inset:0;z-index:400;
  align-items:center;justify-content:center;padding:20px;
}
#prompt-modal.open{display:flex;}
.pm-card{
  position:relative;width:100%;max-width:660px;
  background:var(--bg-s);border:1px solid var(--b3);border-radius:22px;
  padding:24px;
  box-shadow:0 32px 90px rgba(0,0,0,.75);
  animation:modalIn .22s ease;
  max-height:88vh;overflow:hidden;display:flex;flex-direction:column;
}
.pm-search{
  display:flex;align-items:center;gap:8px;padding:8px 12px;
  background:var(--bg-c);border:1px solid var(--b1);border-radius:var(--rs);
  margin-bottom:14px;flex-shrink:0;
}
.pm-search input{
  flex:1;background:none;border:none;outline:none;
  color:var(--t1);font:.85rem var(--fui);
}
.pm-search input::placeholder{color:var(--t3);}
.pm-cats{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;flex-shrink:0;}
.pm-cat{
  padding:4px 12px;background:var(--g2);border:1px solid var(--b1);
  border-radius:99px;font:.74rem var(--fui);color:var(--t2);cursor:pointer;transition:.14s;
}
.pm-cat:hover,.pm-cat.active{background:var(--bg-h);border-color:var(--b3);color:var(--t1);}
.pm-list{flex:1;overflow-y:auto;display:grid;grid-template-columns:1fr 1fr;gap:8px;}
@media(max-width:520px){.pm-list{grid-template-columns:1fr;}}
.pm-item{
  padding:12px 13px;background:var(--bg-c);border:1px solid var(--b1);
  border-radius:var(--r);cursor:pointer;transition:.14s;position:relative;
}
.pm-item:hover{background:var(--bg-i);border-color:var(--b2);transform:translateY(-1px);}
.pm-item-top{display:flex;align-items:center;gap:7px;margin-bottom:4px;}
.pm-item-ico{font-size:16px;}
.pm-item-name{font:.82rem var(--fui);font-weight:600;color:var(--t1);}
.pm-item-text{font:.72rem var(--fui);color:var(--t3);line-height:1.4;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
.pm-item-del{
  position:absolute;top:6px;right:6px;
  background:none;border:none;color:var(--t3);cursor:pointer;
  font-size:10px;opacity:0;transition:.13s;padding:3px 5px;border-radius:4px;
}
.pm-item:hover .pm-item-del{opacity:1;}
.pm-item-del:hover{color:var(--err);background:rgba(220,38,38,.1);}
.pm-add-row{
  display:flex;gap:8px;margin-top:12px;flex-shrink:0;
  padding-top:12px;border-top:1px solid var(--b1);
}
.pm-add-row input{
  flex:1;background:var(--bg-c);border:1px solid var(--b1);border-radius:var(--rs);
  padding:8px 12px;color:var(--t1);font:.82rem var(--fui);outline:none;transition:.2s;
}
.pm-add-row input:focus{border-color:rgba(124,58,237,.4);}

/* ═══ STATS MODAL ════════════════════════════════════════════ */
#stats-modal{
  display:none;position:fixed;inset:0;z-index:400;
  align-items:center;justify-content:center;padding:20px;
}
#stats-modal.open{display:flex;}
.stats-card{
  position:relative;width:100%;max-width:420px;
  background:var(--bg-s);border:1px solid var(--b3);border-radius:20px;
  padding:24px;
  box-shadow:0 24px 70px rgba(0,0,0,.7);
  animation:modalIn .22s ease;
}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:14px 0;}
.stat-box{
  padding:14px;background:var(--bg-c);border:1px solid var(--b1);
  border-radius:var(--r);text-align:center;
}
.stat-val{font:700 1.6rem var(--fco);background:linear-gradient(135deg,var(--ac1),var(--ac2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1;}
.stat-lbl{font:.7rem var(--fui);color:var(--t3);margin-top:4px;}
.stat-bar-wrap{margin:8px 0;}
.stat-bar-label{display:flex;justify-content:space-between;font:.72rem var(--fui);color:var(--t3);margin-bottom:4px;}
.stat-bar-bg{height:6px;background:var(--bg-i);border-radius:6px;overflow:hidden;}
.stat-bar-fill{height:100%;border-radius:6px;background:linear-gradient(90deg,var(--ac1),var(--ac2));}

/* ═══ THEME PICKER ═══════════════════════════════════════════ */
.theme-grid{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;}
.theme-swatch{
  width:36px;height:36px;border-radius:10px;cursor:pointer;
  border:2px solid transparent;transition:.15s;position:relative;
}
.theme-swatch.active,.theme-swatch:hover{border-color:white;transform:scale(1.1);}
.theme-swatch span{
  position:absolute;bottom:-18px;left:50%;transform:translateX(-50%);
  font:.6rem var(--fui);color:var(--t3);white-space:nowrap;
}

/* ═══ TEMPERATURE SLIDER ════════════════════════════════════ */
.temp-row{display:flex;align-items:center;gap:10px;margin-top:6px;}
.temp-slider{
  flex:1;-webkit-appearance:none;height:4px;
  border-radius:4px;outline:none;cursor:pointer;
  background:linear-gradient(to right,var(--ac1) 0%,var(--ac2) 100%);
}
.temp-slider::-webkit-slider-thumb{
  -webkit-appearance:none;width:16px;height:16px;
  border-radius:50%;background:#fff;cursor:pointer;
  box-shadow:0 0 6px rgba(124,58,237,.5);
}
.temp-val{font:.78rem var(--fco);color:var(--t1);min-width:28px;text-align:center;}
.temp-labels{display:flex;justify-content:space-between;font:.64rem var(--fui);color:var(--t3);margin-top:3px;}

/* ═══ FONT SIZE ══════════════════════════════════════════════ */
body.font-sm, body.font-sm * { font-size: 13px !important; }
body.font-md, body.font-md * { font-size: 15px !important; }
body.font-lg, body.font-lg * { font-size: 17px !important; }
body.font-sm .logo-name { font-size: 1rem !important; }
body.font-lg .logo-name { font-size: 1.2rem !important; }
body.font-sm .w-title { font-size: 2rem !important; }
body.font-lg .w-title { font-size: 2.8rem !important; }
body.font-sm pre code { font-size: 0.75rem !important; }
body.font-lg pre code { font-size: 0.88rem !important; }
.font-btns{display:flex;gap:6px;}
.font-btns { display:flex; gap:8px; }
.font-btn{
  padding:8px 18px;background:var(--g1);border:1px solid var(--b1);
  border-radius:var(--rs);color:var(--t2);font-family:var(--fui);
  cursor:pointer;transition:.14s;text-align:center;min-width:60px;line-height:1.3;
}
.font-btn:hover{ background:var(--bg-i); border-color:var(--b2); color:var(--t1); }
.font-btn.active{
  background:rgba(88,101,242,.2) !important;
  border-color:rgba(88,101,242,.55) !important;
  color:#fff !important;
  box-shadow: 0 0 10px rgba(88,101,242,.25);
}

/* ═══ RESPONSE TIMER ════════════════════════════════════════ */
.resp-timer{font:.62rem var(--fco);color:var(--t3);margin-left:6px;}

/* ═══ STARRED CHAT ═══════════════════════════════════════════ */
.ci-star{
  background:none;border:none;color:var(--t3);cursor:pointer;
  font-size:12px;transition:.14s;flex-shrink:0;padding:2px 3px;
  opacity:0;
}
.chat-item:hover .ci-star{opacity:1;}
.ci-star.starred{opacity:1;color:#f59e0b;}
.ci-star.starred:hover{color:#d97706;}

/* ═══ TOAST VARIANTS ════════════════════════════════════════ */
#toast.info   {border-color:rgba(8,145,178,.4);background:rgba(8,145,178,.12);}
#toast.success{border-color:rgba(22,163,74,.4);background:rgba(22,163,74,.12);}
#toast.error  {border-color:rgba(220,38,38,.4);background:rgba(220,38,38,.12);}

/* ═══ TTS SPEAKING ══════════════════════════════════════════ */
.tts-speaking{
  color:#a78bfa !important;
  animation:ttsBlink .7s ease-in-out infinite alternate;
}
@keyframes ttsBlink{from{opacity:1;}to{opacity:.5;}}


/* ═══════════════════════════════════════════════════════════
   DISCORD NITRO REDESIGN
═══════════════════════════════════════════════════════════ */

/* Better root colors — richer purple like Nitro */
:root {
  --bg:         #0b0b14;
  --bg-s:       #0e0e1c;
  --bg-c:       #131325;
  --bg-i:       #18183a;
  --bg-h:       #1e1e45;
  --bg-a:       #252558;
  --g1: rgba(255,255,255,.035);
  --g2: rgba(255,255,255,.065);
  --g3: rgba(255,255,255,.10);
  --b1: rgba(88,101,242,.18);
  --b2: rgba(88,101,242,.30);
  --b3: rgba(88,101,242,.45);
  --t1: #f2f3ff;
  --t2: #8891cc;
  --t3: #40426a;
  --ac1: #5865f2;
  --ac2: #7289da;
  --ac3: #57f287;
  --nitro1: #f47fff;
  --nitro2: #a37dff;
  --nitro3: #5865f2;
  --r:  14px;
  --rs: 8px;
  --sw: 272px;
  --fui: "Outfit", sans-serif;
  --fco: "Fira Code", monospace;
}

/* Ambient orbs — richer */
body::before {
  background: radial-gradient(circle, #5865f2, transparent 70%);
  opacity: .16;
}
body::after {
  background: radial-gradient(circle, #f47fff, transparent 70%);
  opacity: .10;
}

/* Logo gem — Nitro gradient */
.logo-gem {
  background: linear-gradient(135deg, #5865f2, #a37dff, #f47fff) !important;
  box-shadow: 0 0 22px rgba(88,101,242,.55), 0 0 44px rgba(244,127,255,.2) !important;
  animation: logoGlow 4s ease-in-out infinite;
  overflow: hidden;
  padding: 2px;
}

/* New button — Nitro gradient border */
.new-btn {
  background: var(--g1) !important;
  border-color: rgba(88,101,242,.35) !important;
  position: relative;
}
.new-btn:hover {
  background: rgba(88,101,242,.12) !important;
  border-color: rgba(88,101,242,.6) !important;
  box-shadow: 0 4px 20px rgba(88,101,242,.25) !important;
}

/* Sidebar — subtle glass */
#sidebar {
  background: rgba(14,14,28,.95) !important;
  backdrop-filter: blur(20px);
  border-right-color: rgba(88,101,242,.12) !important;
}

/* Topbar — nicer */
#topbar {
  background: rgba(11,11,20,.88) !important;
  backdrop-filter: blur(24px);
  border-bottom-color: rgba(88,101,242,.1) !important;
}

/* Model pill */
.model-pill {
  background: rgba(88,101,242,.1) !important;
  border-color: rgba(88,101,242,.25) !important;
  border-radius: 99px !important;
}
.model-pill:hover {
  background: rgba(88,101,242,.2) !important;
  border-color: rgba(88,101,242,.5) !important;
}
.model-pill.open {
  border-color: rgba(88,101,242,.7) !important;
  box-shadow: 0 0 18px rgba(88,101,242,.25) !important;
}

/* Model dropdown */
.model-dd {
  background: rgba(14,14,28,.98) !important;
  border-color: rgba(88,101,242,.3) !important;
  backdrop-filter: blur(20px);
  box-shadow: 0 24px 70px rgba(0,0,0,.8), 0 0 40px rgba(88,101,242,.15) !important;
}
.model-opt:hover { background: rgba(88,101,242,.15) !important; }
.model-opt.selected { background: rgba(88,101,242,.22) !important; }

/* Send button — Nitro */
#send-btn {
  background: linear-gradient(135deg, #5865f2, #7289da) !important;
  box-shadow: 0 4px 18px rgba(88,101,242,.5) !important;
  border-radius: 12px !important;
}
#send-btn:hover {
  transform: scale(1.09) !important;
  box-shadow: 0 6px 26px rgba(88,101,242,.7) !important;
}

/* Input box */
#input-box {
  background: rgba(19,19,37,.95) !important;
  border-color: rgba(88,101,242,.25) !important;
  border-radius: 16px !important;
  backdrop-filter: blur(10px);
}
#input-box:focus-within {
  border-color: rgba(88,101,242,.6) !important;
  box-shadow: 0 0 0 3px rgba(88,101,242,.14), 0 12px 40px rgba(0,0,0,.3) !important;
}

/* User bubble — Nitro gradient */
.bubble-user {
  background: linear-gradient(135deg, rgba(88,101,242,.28), rgba(163,125,255,.18)) !important;
  border-color: rgba(88,101,242,.38) !important;
  border-radius: 18px 18px 4px 18px !important;
}

/* AI response header */
.ai-avatar {
  background: linear-gradient(135deg, #5865f2, #a37dff) !important;
}

/* Welcome orb */
.w-orb {
  background: linear-gradient(135deg, #5865f2, #a37dff, #f47fff) !important;
  box-shadow: 0 0 55px rgba(88,101,242,.55), 0 0 100px rgba(244,127,255,.22) !important;
  overflow: hidden;
}

/* Welcome title */
.w-title {
  background: linear-gradient(110deg, #fff 10%, #a37dff 50%, #f47fff 100%) !important;
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
}

/* Chips */
.chip {
  border-color: rgba(88,101,242,.2) !important;
}
.chip:hover {
  background: rgba(88,101,242,.18) !important;
  border-color: rgba(88,101,242,.5) !important;
  box-shadow: 0 5px 20px rgba(88,101,242,.22) !important;
  color: #fff !important;
}

/* Active chat item */
.chat-item.active { background: rgba(88,101,242,.18) !important; }
.chat-item.active::before {
  background: linear-gradient(to bottom, #5865f2, #a37dff) !important;
}
.chat-item:hover { background: rgba(88,101,242,.08) !important; }

/* Sidebar footer buttons */
.sf-btn:hover {
  background: rgba(88,101,242,.15) !important;
  border-color: rgba(88,101,242,.3) !important;
  color: #fff !important;
}

/* Modal card */
.modal-card, .pm-card, .stats-card, .shortcuts-card {
  background: rgba(14,14,28,.98) !important;
  border-color: rgba(88,101,242,.3) !important;
  backdrop-filter: blur(20px) !important;
  box-shadow: 0 32px 90px rgba(0,0,0,.82), 0 0 60px rgba(88,101,242,.18) !important;
}

/* Save button */
.save-btn {
  background: linear-gradient(135deg, #5865f2, #7289da) !important;
  box-shadow: 0 4px 18px rgba(88,101,242,.4) !important;
}
.save-btn:hover {
  box-shadow: 0 8px 28px rgba(88,101,242,.6) !important;
}

/* Stat values */
.stat-val {
  background: linear-gradient(135deg, #a37dff, #f47fff) !important;
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
}
.stat-bar-fill {
  background: linear-gradient(90deg, #5865f2, #a37dff) !important;
}
.stat-box {
  border-color: rgba(88,101,242,.18) !important;
}

/* Thinking dots */
.thinking-dot { background: #5865f2 !important; }
.thinking-dot:nth-child(2) { background: #a37dff !important; }
.thinking-dot:nth-child(3) { background: #f47fff !important; }

/* Code blocks */
.msg-content pre { border-color: rgba(88,101,242,.18) !important; }
.code-header { border-bottom-color: rgba(88,101,242,.12) !important; }

/* Voice btn listening */
.voice-btn.listening {
  background: rgba(88,101,242,.2) !important;
  border-color: rgba(88,101,242,.6) !important;
  color: #7289da !important;
}

/* Progress bar */
.progress-bar-fill {
  background: linear-gradient(90deg, #5865f2, #a37dff, #f47fff) !important;
}

/* Form inputs in settings */
.form-input:focus, .form-textarea:focus, .form-select:focus {
  border-color: rgba(88,101,242,.6) !important;
  box-shadow: 0 0 0 3px rgba(88,101,242,.14) !important;
}

/* Theme swatches active */
.theme-swatch.active {
  border-color: #f47fff !important;
  box-shadow: 0 0 8px rgba(244,127,255,.5);
}

/* Prompt library */
.pm-cat.active {
  background: rgba(88,101,242,.22) !important;
  border-color: rgba(88,101,242,.5) !important;
  color: #fff !important;
}
.pm-item:hover {
  border-color: rgba(88,101,242,.35) !important;
  background: rgba(88,101,242,.12) !important;
}
.pm-add-row input:focus { border-color: rgba(88,101,242,.5) !important; }

/* Kbd shortcuts */
.kbd { background: rgba(88,101,242,.15) !important; border-color: rgba(88,101,242,.3) !important; }

/* Toast */
#toast { border-color: rgba(88,101,242,.35) !important; }
#toast.success { border-color: rgba(87,242,135,.4) !important; background: rgba(87,242,135,.1) !important; }
#toast.error   { border-color: rgba(237,66,69,.4) !important;  background: rgba(237,66,69,.1)  !important; }
#toast.info    { border-color: rgba(88,101,242,.4) !important; background: rgba(88,101,242,.1) !important; }

/* Font size reset fix */
body.font-sm .msg-content { font-size: .82rem !important; }
body.font-lg .msg-content { font-size: .98rem !important; }

/* Scrollbar — Nitro purple */
::-webkit-scrollbar-thumb { background: rgba(88,101,242,.3) !important; }
::-webkit-scrollbar-thumb:hover { background: rgba(88,101,242,.5) !important; }


/* ═══ BACKGROUND IMAGE SYSTEM ══════════════════════════════ */

/* The bg layer sits behind everything */
#bg-layer {
  position: fixed;
  inset: 0;
  z-index: -1;
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  opacity: 0;
  transition: opacity .5s ease;
  pointer-events: none;
}
#bg-layer.active { opacity: 1; }

/* Dark overlay on top of bg image so text stays readable */
#bg-overlay {
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
  transition: background .4s;
}

/* When bg is active, make sidebar/topbar/input slightly glassy */
body.has-bg #sidebar {
  background: rgba(11,11,20,.82) !important;
  backdrop-filter: blur(18px) saturate(160%);
}
body.has-bg #topbar {
  background: rgba(11,11,20,.75) !important;
  backdrop-filter: blur(22px) saturate(160%);
}
body.has-bg #input-box {
  background: rgba(19,19,37,.82) !important;
  backdrop-filter: blur(14px) saturate(140%);
}
body.has-bg .modal-card,
body.has-bg .pm-card,
body.has-bg .stats-card,
body.has-bg .shortcuts-card {
  background: rgba(11,11,20,.92) !important;
  backdrop-filter: blur(24px) saturate(160%);
}
body.has-bg #chat-area {
  background: transparent;
}
body.has-bg .bubble-user {
  background: linear-gradient(135deg,rgba(88,101,242,.32),rgba(163,125,255,.22)) !important;
  backdrop-filter: blur(8px);
}
body.has-bg .msg-group:not(.user) .bubble-ai {
  /* subtle bg on AI messages */
}

/* ── BG Settings UI ── */
.bg-section {
  margin-bottom: 18px;
  padding: 16px;
  background: var(--bg-c);
  border: 1px solid var(--b1);
  border-radius: var(--r);
}
.bg-section-title {
  font: 600 .82rem var(--fui);
  color: var(--t1);
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 7px;
}

/* Preview box */
.bg-preview-wrap {
  position: relative;
  width: 100%;
  height: 110px;
  border-radius: var(--rs);
  overflow: hidden;
  background: var(--bg-i);
  border: 1px solid var(--b1);
  margin-bottom: 12px;
  cursor: pointer;
  transition: .2s;
}
.bg-preview-wrap:hover { border-color: rgba(88,101,242,.5); }
.bg-preview-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: none;
}
.bg-preview-img.visible { display: block; }
.bg-preview-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  color: var(--t3);
  font: .78rem var(--fui);
}
.bg-preview-placeholder svg { opacity: .4; }
.bg-preview-label {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  padding: 6px 10px;
  background: rgba(0,0,0,.55);
  font: .7rem var(--fui);
  color: rgba(255,255,255,.7);
  display: none;
}
.bg-preview-img.visible ~ .bg-preview-label { display: block; }

/* Slider rows */
.bg-slider-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.bg-slider-label {
  font: .75rem var(--fui);
  color: var(--t2);
  min-width: 58px;
}
.bg-slider {
  flex: 1;
  -webkit-appearance: none;
  height: 4px;
  border-radius: 4px;
  outline: none;
  cursor: pointer;
  background: var(--bg-h);
}
.bg-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 16px; height: 16px;
  border-radius: 50%;
  background: #fff;
  cursor: pointer;
  box-shadow: 0 0 6px rgba(88,101,242,.5);
  transition: transform .1s;
}
.bg-slider::-webkit-slider-thumb:hover { transform: scale(1.2); }
.bg-slider-val {
  font: .72rem var(--fco);
  color: var(--t1);
  min-width: 34px;
  text-align: right;
}

/* Preset gradients / solid colors */
.bg-presets {
  display: flex;
  gap: 7px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
.bg-preset {
  width: 38px; height: 38px;
  border-radius: 9px;
  cursor: pointer;
  border: 2px solid transparent;
  transition: .15s;
  flex-shrink: 0;
  position: relative;
}
.bg-preset:hover { transform: scale(1.1); }
.bg-preset.active { border-color: #fff; box-shadow: 0 0 8px rgba(255,255,255,.4); }
.bg-preset-none {
  background: var(--bg-i);
  display: flex; align-items: center; justify-content: center;
  color: var(--t3); font-size: 16px;
}

/* Button row */
.bg-btn-row { display: flex; gap: 8px; }
.bg-upload-btn {
  flex: 1;
  padding: 9px;
  background: linear-gradient(135deg,rgba(88,101,242,.2),rgba(163,125,255,.15));
  border: 1px solid rgba(88,101,242,.4);
  border-radius: var(--rs);
  color: var(--t1);
  font: 500 .8rem var(--fui);
  cursor: pointer;
  transition: .2s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}
.bg-upload-btn:hover {
  background: linear-gradient(135deg,rgba(88,101,242,.35),rgba(163,125,255,.25));
  border-color: rgba(88,101,242,.7);
  box-shadow: 0 4px 14px rgba(88,101,242,.25);
}
.bg-remove-btn {
  padding: 9px 14px;
  background: rgba(220,38,38,.1);
  border: 1px solid rgba(220,38,38,.3);
  border-radius: var(--rs);
  color: #fca5a5;
  font: .8rem var(--fui);
  cursor: pointer;
  transition: .2s;
}
.bg-remove-btn:hover {
  background: rgba(220,38,38,.2);
  border-color: rgba(220,38,38,.6);
}
#bg-file-input { display: none; }


/* ═══ MATH / KATEX ══════════════════════════════════════════ */
.katex { font-size: 1.05em; color: var(--t1); }
.katex-display {
  margin: .9em 0;
  padding: 14px 18px;
  background: var(--bg-c);
  border: 1px solid var(--b1);
  border-left: 3px solid var(--ac1);
  border-radius: 0 var(--rs) var(--rs) 0;
  overflow-x: auto;
}
.katex-display > .katex { font-size: 1.18em; }

/* ═══ CODE BLOCK — GEMINI STYLE ════════════════════════════ */
.msg-content pre { position: relative; }

/* Language label pill */
.code-lang {
  font: 600 .65rem var(--fco);
  color: var(--t2);
  text-transform: uppercase;
  letter-spacing: .07em;
  display: flex;
  align-items: center;
  gap: 6px;
}
.code-lang::before {
  content: '';
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--ac1);
  display: inline-block;
  flex-shrink: 0;
}

/* Copy button — Gemini style */
.code-copy {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.1);
  border-radius: 6px;
  color: var(--t2);
  font: 500 .72rem var(--fui);
  cursor: pointer;
  transition: all .15s;
  white-space: nowrap;
}
.code-copy:hover {
  background: rgba(88,101,242,.18);
  border-color: rgba(88,101,242,.4);
  color: #fff;
}
.code-copy.copied {
  background: rgba(87,242,135,.12);
  border-color: rgba(87,242,135,.35);
  color: #57f287;
}
.code-copy svg { flex-shrink: 0; }

/* Code header */
.code-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 14px;
  background: rgba(255,255,255,.04);
  border-bottom: 1px solid rgba(255,255,255,.06);
  gap: 10px;
}

/* Line numbers hint */
.code-lines {
  font: .62rem var(--fco);
  color: var(--t3);
  margin-left: auto;
  margin-right: 8px;
}


[data-theme="sand"] {
  --bg:   #faf8f3;
  --bg-s: #f3f1eb;
  --bg-c: #eceae4;
  --bg-i: #e4e2db;
  --bg-h: #dbd9d2;
  --bg-a: #d0cec7;
  --g1: rgba(0,0,0,.025);
  --g2: rgba(0,0,0,.05);
  --g3: rgba(0,0,0,.08);
  --b1: rgba(120,100,60,.12);
  --b2: rgba(120,100,60,.18);
  --b3: rgba(120,100,60,.26);
  --t1: #1a1810;
  --t2: #5c5848;
  --t3: #9c9880;
  --ac1: #b45309;
  --ac2: #0369a1;
  --ac3: #15803d;
}
[data-theme="sand"] #sidebar  { background: #f0ede5 !important; }
[data-theme="sand"] #topbar   { background: rgba(250,248,243,.92) !important; backdrop-filter:blur(20px); }
[data-theme="sand"] #input-box{ background: #fff !important; border-color: rgba(120,100,60,.18) !important; }
[data-theme="sand"] #send-btn { background: linear-gradient(135deg,#b45309,#0369a1) !important; }
[data-theme="sand"] .save-btn { background: linear-gradient(135deg,#b45309,#0369a1) !important; }
[data-theme="sand"] .bubble-user { background:linear-gradient(135deg,rgba(180,83,9,.14),rgba(3,105,161,.10)) !important; border-color:rgba(180,83,9,.22) !important; }
[data-theme="sand"] .chip { background:rgba(255,255,255,.7) !important; border-color:rgba(120,100,60,.15) !important; color:#3a3620 !important; }
[data-theme="sand"] .chip:hover { background:rgba(180,83,9,.1) !important; border-color:rgba(180,83,9,.35) !important; }
[data-theme="sand"] .w-title { background:linear-gradient(110deg,#1a1810 10%,#b45309 55%,#0369a1 100%) !important; -webkit-background-clip:text !important; -webkit-text-fill-color:transparent !important; }
[data-theme="sand"] .msg-content pre { background:#1e1e2e !important; }
[data-theme="sand"] body::before { background:radial-gradient(circle,rgba(180,83,9,.2),transparent 70%) !important; opacity:.15 !important; }
[data-theme="sand"] body::after  { background:radial-gradient(circle,rgba(3,105,161,.15),transparent 70%) !important; opacity:.12 !important; }


/* ═══ BUBBLE FRAME / CARD STYLES ══════════════════════════ */

/* Kiểu mặc định — không khung */
.msg-group .bubble-ai { }

/* Kiểu card — khung mờ nổi bật */
body.bubble-card .msg-group:not(.user) {
  padding: 14px 18px;
  background: rgba(10,10,25,.65);
  backdrop-filter: blur(14px) saturate(180%);
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 16px;
  margin: 2px 20px;
}
body.bubble-card .msg-group.user {
  margin: 2px 20px;
}

/* Kiểu frost — trắng mờ */
body.bubble-frost .msg-group:not(.user) {
  padding: 14px 18px;
  background: rgba(255,255,255,.08);
  backdrop-filter: blur(20px) saturate(200%);
  border: 1px solid rgba(255,255,255,.14);
  border-radius: 16px;
  margin: 2px 20px;
}

/* Kiểu line — chỉ viền trái */
body.bubble-line .msg-group:not(.user) {
  padding: 10px 18px 10px 20px;
  border-left: 3px solid var(--ac1);
  margin: 2px 20px;
  background: rgba(88,101,242,.06);
  border-radius: 0 12px 12px 0;
}

/* Light mode adjustments */
[data-theme="light"] body.bubble-card .msg-group:not(.user),
[data-theme="sand"]  body.bubble-card .msg-group:not(.user) {
  background: rgba(255,255,255,.72) !important;
  border-color: rgba(0,0,0,.1) !important;
}
[data-theme="light"] body.bubble-frost .msg-group:not(.user),
[data-theme="sand"]  body.bubble-frost .msg-group:not(.user) {
  background: rgba(255,255,255,.55) !important;
  border-color: rgba(0,0,0,.12) !important;
}

/* User bubble also gets card treatment */
body.bubble-card .bubble-user,
body.bubble-frost .bubble-user {
  backdrop-filter: blur(12px);
}

/* ── Bubble opacity slider ── */
.bubble-preview {
  height: 52px;
  border-radius: var(--rs);
  overflow: hidden;
  margin-bottom: 10px;
  position: relative;
  background: linear-gradient(135deg, var(--bg-c), var(--bg-i));
  border: 1px solid var(--b1);
  display: flex;
  align-items: center;
  padding: 0 14px;
  gap: 10px;
}
.bubble-preview-dot {
  width: 26px; height: 26px;
  border-radius: 8px;
  background: linear-gradient(135deg, var(--ac1), var(--ac2));
  flex-shrink: 0;
}
.bubble-preview-text {
  font: .78rem var(--fui);
  color: var(--t1);
}

/* Bubble style buttons */
.bubble-style-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.bs-btn {
  padding: 10px 12px;
  background: var(--bg-c);
  border: 1px solid var(--b1);
  border-radius: var(--rs);
  cursor: pointer;
  transition: .15s;
  text-align: left;
}
.bs-btn:hover { background: var(--bg-i); border-color: var(--b2); }
.bs-btn.active {
  background: rgba(88,101,242,.14) !important;
  border-color: rgba(88,101,242,.5) !important;
}
.bs-name { font: 600 .79rem var(--fui); color: var(--t1); }
.bs-desc { font: .67rem var(--fui); color: var(--t3); margin-top: 2px; }


/* Error toast — lớn hơn, dễ thấy */
#toast.error {
  border-color: rgba(220,38,38,.5) !important;
  background: rgba(30,0,0,.92) !important;
  color: #fca5a5 !important;
  font-size: .85rem !important;
  padding: 12px 22px !important;
  max-width: 480px !important;
  white-space: normal !important;
  text-align: center !important;
  line-height: 1.5 !important;
  box-shadow: 0 8px 32px rgba(220,38,38,.35) !important;
}

</style>
</head>
<body>

<!-- Background layer -->
<div id="bg-layer" style="display:none;position:fixed;inset:0;z-index:0;pointer-events:none;
  background-size:cover;background-position:center;background-repeat:no-repeat;
  transition:opacity .4s"></div>

<!-- Background image layer -->
<div id="bg-layer"></div>
<div id="bg-overlay"></div>

<!-- ═══ DRAG OVERLAY ══════════════════════════════════════════ -->
<div id="drag-overlay">
  <div class="drag-box">
    <div class="drag-icon">📂</div>
    <div class="drag-title">Thả file vào đây</div>
    <div class="drag-sub">Code (.py .js .ts…) · Text · Ảnh — tối đa 20MB</div>
  </div>
</div>

<!-- ═══ SIDEBAR ═══════════════════════════════════════════════ -->
<div id="sidebar">
  <div class="s-top">
    <div class="logo">
      <div class="logo-gem"><img src="/icon.png" alt="H" style="width:24px;height:24px;border-radius:6px;object-fit:cover"></div>
      <div class="logo-text">
        <div class="logo-name">Hnhat AI</div>
        <span class="logo-ver">v3.0</span>
      </div>
    </div>
    <button class="new-btn" onclick="newChat()">
      <span class="new-ic" style="font-size:15px;background:none;font-style:normal;display:flex;align-items:center;justify-content:center"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg></span>
      Cuộc trò chuyện mới
    </button>
  </div>

  <div class="s-search">
    <div class="search-box">
      <span>🔍</span>
      <input class="search-input" id="search-input" placeholder="Tìm kiếm chat…"
             oninput="filterChats(this.value)">
    </div>
  </div>

  <div class="cl-label">Lịch sử trò chuyện</div>
  <div id="chat-list"></div>

  <div class="s-foot">
    <button class="sf-btn" onclick="openSettings()"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.1 4.9C19.9 5.7 20 7 19.4 8l-1 1.7c.2.8.3 1.5.3 2.3s-.1 1.5-.3 2.3l1 1.7c.6 1 .5 2.3-.3 3.1l-.3.3c-.8.8-2.1.9-3.1.3l-1.7-1c-.8.2-1.5.3-2.3.3s-1.5-.1-2.3-.3l-1.7 1c-1 .6-2.3.5-3.1-.3l-.3-.3c-.8-.8-.9-2.1-.3-3.1l1-1.7C4.1 13.5 4 12.8 4 12s.1-1.5.3-2.3l-1-1.7c-.6-1-.5-2.3.3-3.1l.3-.3c.8-.8 2.1-.9 3.1-.3l1.7 1c.8-.2 1.5-.3 2.3-.3s1.5.1 2.3.3l1.7-1c1-.6 2.3-.5 3.1.3z"/></svg> Cài đặt</button>
    <button class="sf-btn" onclick="toggleTheme()" id="theme-btn"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg></button>
    <button class="sf-btn" onclick="openShortcuts()"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M8 14h8"/></svg></button>
    <button class="sf-btn" onclick="showExportMenu()">📥</button>
  </div>
</div>

<!-- ═══ MAIN ══════════════════════════════════════════════════ -->
<div id="main">

  <!-- Topbar -->
  <div id="topbar">
    <button class="tb-btn" onclick="toggleSidebar()" title="Toggle sidebar"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg></button>
    <div id="chat-title" title="Double-click để đổi tên">Hnhat AI</div>

    <div class="tb-actions">
      <button class="tb-btn" onclick="openSearchBar()" title="Tìm trong chat (Ctrl+F)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></button>
      <button class="tb-btn" onclick="confirmClearChat()" title="Xóa messages"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="3,6 5,6 21,6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg></button>

      <!-- Model pill + dropdown -->
      <div class="model-pill" id="model-pill" onclick="toggleModelDD()">
        <div class="pill-dot" id="pill-dot"></div>
        <span class="pill-name" id="pill-name">Hnhat Pro</span>
        <span class="pill-arrow">▼</span>

        <div class="model-dd" id="model-dd">
          <!-- Groq section -->
          

          <div class="model-opt" data-model="Hnhat Fast" onclick="selectModel('Hnhat Fast', event)">
            <div class="mopt-icon">⚡</div>
            <div class="mopt-body">
              <div class="mopt-name">Hnhat Fast</div>
              <div class="mopt-desc">Siêu nhanh · Tiết kiệm</div>
              <div class="mopt-meta">
                <span class="mopt-badge" style="background:rgba(245,158,11,.15);color:#f59e0b;border:1px solid rgba(245,158,11,.3)">FAST</span>
                <span class="mopt-ctx">128K</span>
              </div>
            </div>
          </div>

          <div class="model-opt selected" data-model="Hnhat Pro" onclick="selectModel('Hnhat Pro', event)">
            <div class="mopt-icon">🔥</div>
            <div class="mopt-body">
              <div class="mopt-name">Hnhat Pro</div>
              <div class="mopt-desc">Cân bằng · Thông minh nhất</div>
              <div class="mopt-meta">
                <span class="mopt-badge" style="background:rgba(139,92,246,.15);color:#8b5cf6;border:1px solid rgba(139,92,246,.3)">PRO</span>
                <span class="mopt-ctx">128K</span>
              </div>
            </div>
          </div>

          <div class="model-opt" data-model="Hnhat Master" onclick="selectModel('Hnhat Master', event)">
            <div class="mopt-icon">👑</div>
            <div class="mopt-body">
              <div class="mopt-name">Hnhat Master</div>
              <div class="mopt-desc">Suy luận sâu · Phân tích phức tạp</div>
              <div class="mopt-meta">
                <span class="mopt-badge" style="background:rgba(6,182,212,.15);color:#06b6d4;border:1px solid rgba(6,182,212,.3)">MASTER</span>
                <span class="mopt-ctx">128K</span>
              </div>
            </div>
          </div>

          <div class="model-opt" data-model="Hnhat Code" onclick="selectModel('Hnhat Code', event)">
            <div class="mopt-icon">💻</div>
            <div class="mopt-body">
              <div class="mopt-name">Hnhat Code</div>
              <div class="mopt-desc">Code chuyên sâu · Upload 20MB</div>
              <div class="mopt-meta">
                <span class="mopt-badge" style="background:rgba(16,185,129,.15);color:#10b981;border:1px solid rgba(16,185,129,.3)">CODE</span>
                <span class="mopt-ctx">128K</span>
              </div>
            </div>
          </div>

          <div class="model-opt" data-model="Hnhat Reason" onclick="selectModel('Hnhat Reason', event)">
            <div class="mopt-icon">🧠</div>
            <div class="mopt-body">
              <div class="mopt-name">Hnhat Reason</div>
              <div class="mopt-desc">Chain of Thought · Tư duy chuỗi</div>
              <div class="mopt-meta">
                <span class="mopt-badge" style="background:rgba(249,115,22,.15);color:#f97316;border:1px solid rgba(249,115,22,.3)">REASON</span>
                <span class="mopt-ctx">128K</span>
              </div>
            </div>
          </div>

          <div class="model-opt" data-model="Mixtral" onclick="selectModel('Mixtral', event)">
            <div class="mopt-icon">🌀</div>
            <div class="mopt-body">
              <div class="mopt-name">Mixtral</div>
              <div class="mopt-desc">Mixture of Experts · Đa năng</div>
              <div class="mopt-meta">
                <span class="mopt-badge" style="background:rgba(14,165,233,.15);color:#0ea5e9;border:1px solid rgba(14,165,233,.3)">MIXTRAL</span>
                <span class="mopt-ctx">32K</span>
              </div>
            </div>
          </div>

          <div class="model-opt" data-model="Gemma 2" onclick="selectModel('Gemma 2', event)">
            <div class="mopt-icon">💎</div>
            <div class="mopt-body">
              <div class="mopt-name">Gemma 2</div>
              <div class="mopt-desc">Gemma 2 · Nhỏ gọn & nhanh</div>
              <div class="mopt-meta">
                <span class="mopt-badge" style="background:rgba(20,184,166,.15);color:#14b8a6;border:1px solid rgba(20,184,166,.3)">GEMMA</span>
                <span class="mopt-ctx">8K</span>
              </div>
            </div>
          </div>

          <div class="model-opt" data-model="Compound" onclick="selectModel('Compound', event)">
            <div class="mopt-icon">⚗️</div>
            <div class="mopt-body">
              <div class="mopt-name">Compound</div>
              <div class="mopt-desc">Compound AI · Đa năng</div>
              <div class="mopt-meta">
                <span class="mopt-badge" style="background:rgba(168,85,247,.15);color:#a855f7;border:1px solid rgba(168,85,247,.3)">COMP</span>
                <span class="mopt-ctx">128K</span>
              </div>
            </div>
          </div>

          <div class="dd-divider"></div>
          <!-- Gemini section -->
          

          <div class="model-opt" data-model="Hnhat Vision" onclick="selectModel('Hnhat Vision', event)">
            <div class="mopt-icon">👁</div>
            <div class="mopt-body">
              <div class="mopt-name">Hnhat Vision</div>
              <div class="mopt-desc">Phân tích hình ảnh · Nhanh</div>
              <div class="mopt-meta">
                <span class="mopt-badge" style="background:rgba(66,133,244,.15);color:#4285f4;border:1px solid rgba(66,133,244,.3)">VISION</span>
                <span class="mopt-ctx">1M</span>
              </div>
            </div>
          </div>

          <div class="model-opt" data-model="Hnhat Vision+" onclick="selectModel('Hnhat Vision+', event)">
            <div class="mopt-icon">🔭</div>
            <div class="mopt-body">
              <div class="mopt-name">Hnhat Vision+</div>
              <div class="mopt-desc">Phân tích hình ảnh · Chất lượng cao</div>
              <div class="mopt-meta">
                <span class="mopt-badge" style="background:rgba(26,115,232,.15);color:#1a73e8;border:1px solid rgba(26,115,232,.3)">VIS+</span>
                <span class="mopt-ctx">1M</span>
              </div>
            </div>
          </div>

        </div><!-- end model-dd -->
      </div><!-- end model-pill -->
    </div><!-- end tb-actions -->
  </div><!-- end topbar -->

  <!-- Search bar -->
  <div id="search-bar">
    <span style="font-size:14px;color:var(--t3)">🔍</span>
    <input class="sb-input" id="sb-input" placeholder="Tìm kiếm trong chat…"
           oninput="doSearch(this.value)" onkeydown="searchKey(event)">
    <div class="sb-nav">
      <span class="sb-count" id="sb-count"></span>
      <button class="tb-btn" onclick="searchNav(-1)" title="Kết quả trước">↑</button>
      <button class="tb-btn" onclick="searchNav(1)"  title="Kết quả sau">↓</button>
    </div>
    <button class="tb-btn" onclick="closeSearchBar()">✕</button>
  </div>

  <!-- Chat area -->
  <div id="chat-area">

    <!-- Welcome -->
    <div id="welcome">
      <div class="w-orb"><img src="/icon.png" alt="Hnhat" style="width:50px;height:50px;border-radius:14px;object-fit:cover"></div>
      <div class="w-title">Xin chào, tôi là Hnhat AI</div>
      <div class="w-sub">
        Model AI Made By Hnhat<br>
        Hỏi bất cứ điều gì hoặc kéo thả vào đây!
      </div>
      <div class="w-chips">
        <div class="chip" onclick="qi('Viết REST API Python Flask + JWT + SQLite, có swagger docs')">🌐 Flask REST API</div>
        <div class="chip" onclick="qi('Hướng dẫn build .exe từ Python dùng PyInstaller, single file')">📦 Build .exe</div>
        <div class="chip" onclick="qi('Tạo web app Next.js 14 với TypeScript, Tailwind, tối ưu SEO')">⚛️ Next.js App</div>
        <div class="chip" onclick="qi('Viết Dockerfile + docker-compose cho Flask + PostgreSQL + Redis')">🐳 Docker Stack</div>
        <div class="chip" onclick="qi('Tối ưu PostgreSQL query cho bảng 50 triệu dòng, index strategy')">🗄️ SQL Optimize</div>
        <div class="chip" onclick="qi('Hướng dẫn deploy app lên VPS Ubuntu: Nginx + Certbot SSL + systemd')">🚀 Deploy VPS</div>
        <div class="chip" onclick="qi('Giải thích async/await và event loop Python với asyncio chi tiết')">⚡ Async Python</div>
        <div class="chip" onclick="qi('Viết CI/CD pipeline GitHub Actions: test + build + deploy tự động')">🔄 CI/CD Pipeline</div>
        <div class="chip" onclick="switchModel('Hnhat Reason');qi('Giải bài toán thuật toán này từng bước: tìm đường đi ngắn nhất trong đồ thị có trọng số')">🧠 Suy luận thuật toán</div>
        <div class="chip" onclick="switchModel('Hnhat Code');qi('Review code này, tìm bug, đề xuất cải tiến theo best practices')">💻 Code Review</div>
        <div class="chip" onclick="switchModel('Hnhat Vision');qi('Phân tích chi tiết hình ảnh này')">👁 Phân tích ảnh</div>
        <div class="chip" onclick="qi('Tạo script Python automation: đọc Excel, xử lý data, xuất report PDF')">📊 Data Automation</div>
      </div>
    </div>

    <!-- Messages -->
    <div id="messages" style="display:none"></div>

  </div><!-- end chat-area -->

  <!-- Input area -->
  <div id="input-wrap">
    <div id="input-box"
         ondragover="onDragOver(event)"
         ondragleave="onDragLeave()"
         ondrop="onDrop(event)">

      <!-- Attached file preview -->
      <div id="attach-preview"></div>

      <div class="input-row">
        <textarea id="user-input" rows="1"
                  placeholder="Nhắn tin với Hnhat AI…"
                  onkeydown="onInputKey(event)"
                  oninput="onInputChange(this)"></textarea>
        <div class="input-actions">
          <button class="i-btn" onclick="document.getElementById('file-input').click()"
                  title="Đính kèm file / ảnh (tối đa 20MB)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg></button>
          <button class="voice-btn" id="voice-btn" onclick="toggleVoiceInput()"
                  title="Nhập bằng giọng nói (Chrome)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg></button>
          <input type="file" id="file-input" accept="*/*" onchange="onFileSelect(this)">
          <button id="send-btn" onclick="sendMessage()" title="Gửi (Enter)"><svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="m22 2-11 20-4-9-9-4 24-7z"/></svg></button>
        </div>
      </div>
    </div>
    <div class="input-footer">
      <span class="input-hint">Enter gửi · Shift+Enter ↵ · Ctrl+K chat mới · Kéo thả file</span>
      <span id="token-count"></span>
    </div>
  </div>

</div><!-- end main -->

<!-- ═══ SETTINGS MODAL ════════════════════════════════════════ -->
<div id="settings-modal">
  <div class="modal-backdrop" onclick="closeSettings()"></div>
  <div class="modal-card">
    <button class="modal-close" onclick="closeSettings()">✕</button>
    <div class="modal-title"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.1 4.9C19.9 5.7 20 7 19.4 8l-1 1.7c.2.8.3 1.5.3 2.3s-.1 1.5-.3 2.3l1 1.7c.6 1 .5 2.3-.3 3.1l-.3.3c-.8.8-2.1.9-3.1.3l-1.7-1c-.8.2-1.5.3-2.3.3s-1.5-.1-2.3-.3l-1.7 1c-1 .6-2.3.5-3.1-.3l-.3-.3c-.8-.8-.9-2.1-.3-3.1l1-1.7C4.1 13.5 4 12.8 4 12s.1-1.5.3-2.3l-1-1.7c-.6-1-.5-2.3.3-3.1l.3-.3c.8-.8 2.1-.9 3.1-.3l1.7 1c.8-.2 1.5-.3 2.3-.3s1.5.1 2.3.3l1.7-1c1-.6 2.3-.5 3.1.3z"/></svg> Cài đặt Hnhat AI</div>

    <!-- Groq Key -->
    <div class="key-section">
      <div class="key-section-title">
        🔑 API Key chính
        &nbsp;<span class="key-status" id="groq-key-status"></span>
      </div>
      <div class="form-group" style="margin-bottom:8px">
        <div class="key-row">
          <input type="password" class="form-input" id="groq-key-input"
                 placeholder="gsk_xxxxxxxxxxxxxxxxxxxx">
          <button class="i-btn" style="width:38px;height:38px;flex-shrink:0"
                  onclick="toggleKeyVis('groq-key-input')" title="Hiện/ẩn">👁</button>
        </div>
        <div class="form-hint">
          Lấy miễn phí tại <a href="https://console.groq.com/keys" target="_blank">console.groq.com/keys</a>
          · Dùng cho 8 text models
        </div>
      </div>
    </div>

    <!-- Gemini Key -->
    <div class="key-section">
      <div class="key-section-title">
        🖼️ API Key phụ (Vision)
        &nbsp;<span class="key-status" id="gemini-key-status"></span>
      </div>
      <div class="form-group" style="margin-bottom:8px">
        <div class="key-row">
          <input type="password" class="form-input" id="gemini-key-input"
                 placeholder="AIzaSyxxxxxxxxxxxxxxxxxx">
          <button class="i-btn" style="width:38px;height:38px;flex-shrink:0"
                  onclick="toggleKeyVis('gemini-key-input')" title="Hiện/ẩn">👁</button>
        </div>
        <div class="form-hint">
          Lấy miễn phí tại <a href="https://aistudio.google.com/app/apikey" target="_blank">aistudio.google.com</a>
          · Dùng cho Hnhat Vision / Vision+
        </div>
      </div>
    </div>

    <div class="form-group">
      <label class="form-label">🎭 Model mặc định</label>
      <select class="form-select" id="default-model-input">
        <option value="Hnhat Fast">⚡ Hnhat Fast — Siêu nhanh</option>
        <option value="Hnhat Pro" selected>🔥 Hnhat Pro — Tốt nhất</option>
        <option value="Hnhat Master">👑 Hnhat Master — DeepSeek R1</option>
        <option value="Hnhat Code">💻 Hnhat Code — Code 20MB</option>
        <option value="Hnhat Reason">🧠 Hnhat Reason — QwQ</option>
        <option value="Hnhat Vision">👁 Hnhat Vision — Gemini Flash</option>
        <option value="Hnhat Vision+">🔭 Hnhat Vision+ — Gemini Pro</option>
        <option value="Kimi K2">🌙 Kimi K2 — Đa năng</option>
        <option value="Gemma 2">💎 Gemma 2</option>
        <option value="Compound">⚗️ Compound</option>
      </select>
    </div>

    <div class="form-group">
      <label class="form-label">🧠 System Prompt tùy chỉnh</label>
      <textarea class="form-textarea" id="system-prompt-input" rows="3"
                placeholder="Để trống = dùng system prompt mặc định của từng model…"></textarea>
    </div>

    <div class="form-group">
      <label class="form-label">🎨 Giao diện (Theme)</label>
      <div class="theme-grid" id="theme-grid">
        <div class="theme-swatch" data-t="dark"
             style="background:linear-gradient(135deg,#030309,#7c3aed)"
             onclick="applyTheme('dark')" title="Dark"><span>Dark</span></div>
        <div class="theme-swatch" data-t="midnight"
             style="background:linear-gradient(135deg,#05071a,#6366f1)"
             onclick="applyTheme('midnight')" title="Midnight"><span>Midnight</span></div>
        <div class="theme-swatch" data-t="ocean"
             style="background:linear-gradient(135deg,#010f12,#0d9488)"
             onclick="applyTheme('ocean')" title="Ocean"><span>Ocean</span></div>
        <div class="theme-swatch" data-t="forest"
             style="background:linear-gradient(135deg,#020d04,#16a34a)"
             onclick="applyTheme('forest')" title="Forest"><span>Forest</span></div>
        <div class="theme-swatch" data-t="rose"
             style="background:linear-gradient(135deg,#0f0508,#e11d48)"
             onclick="applyTheme('rose')" title="Rose"><span>Rose</span></div>
        <div class="theme-swatch" data-t="light"
             style="background:linear-gradient(135deg,#f5f4f0,#4f46e5)"
             onclick="applyTheme('light')" title="Light (Soft)"><span>Light</span></div>
        <div class="theme-swatch" data-t="sand"
             style="background:linear-gradient(135deg,#faf8f3,#b45309)"
             onclick="applyTheme('sand')" title="Sand (Warm)"><span>Sand</span></div>
      </div>
    </div>

    <div class="form-group">
      <label class="form-label">🌡️ Độ sáng tạo (Temperature)</label>
      <div class="temp-row">
        <input type="range" class="temp-slider" id="temp-slider"
               min="0" max="2" step="0.1" value="0.7"
               oninput="document.getElementById('temp-val').textContent=parseFloat(this.value).toFixed(1)">
        <span class="temp-val" id="temp-val">0.7</span>
      </div>
      <div class="temp-labels"><span>Chính xác</span><span>Cân bằng</span><span>Sáng tạo</span></div>
    </div>

    <div class="form-group">
      <label class="form-label">🔤 Cỡ chữ chat</label>
      <div class="font-btns">
        <button class="font-btn" id="fs-sm" onclick="setFontSize('sm')">A<span style="font-size:.62rem;display:block;opacity:.6">Nhỏ</span></button>
        <button class="font-btn" id="fs-md" onclick="setFontSize('md')">A<span style="font-size:.62rem;display:block;opacity:.6">Vừa</span></button>
        <button class="font-btn" id="fs-lg" onclick="setFontSize('lg')" style="font-size:1.05rem">A<span style="font-size:.62rem;display:block;opacity:.6">Lớn</span></button>
      </div>
    </div>

    <div class="form-group">
      <label class="form-label">🖼️ Hình nền Background</label>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <button class="font-btn" id="bg-pick-btn" onclick="pickBackground()"
          style="padding:8px 14px;display:flex;align-items:center;gap:6px">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21,15 16,10 5,21"/></svg>
          Chọn ảnh từ máy
        </button>
        <button class="font-btn" id="bg-remove-btn" onclick="removeBackground()" style="padding:8px 14px">✕ Xóa nền</button>
        <span id="bg-file-name" style="font:.72rem var(--fco);color:var(--t3)">Chưa chọn</span>
      </div>
      <input type="file" id="bg-file-input" accept="image/*" style="display:none" onchange="onBgFileSelect(this)">
      <div id="bg-preview-wrap" style="display:none;margin-top:10px;border-radius:var(--rs);overflow:hidden;position:relative;max-height:120px">
        <img id="bg-preview-img" style="width:100%;object-fit:cover;border-radius:var(--rs);border:1px solid var(--b1)">
        <div style="position:absolute;inset:0;background:rgba(0,0,0,.4);display:flex;align-items:center;justify-content:center">
          <span style="color:#fff;font:.78rem var(--fui)">Preview</span>
        </div>
      </div>
      <div style="display:flex;gap:10px;margin-top:10px;align-items:center">
        <label class="form-label" style="margin:0;white-space:nowrap">Độ mờ:</label>
        <input type="range" class="temp-slider" id="bg-blur-slider"
               min="0" max="20" step="1" value="0"
               oninput="previewBg()">
        <span id="bg-blur-val" style="font:.72rem var(--fco);color:var(--t1);min-width:22px">0px</span>
        <label class="form-label" style="margin:0;white-space:nowrap">Tối:</label>
        <input type="range" class="temp-slider" id="bg-dim-slider"
               min="0" max="0.9" step="0.05" value="0.5"
               oninput="previewBg()">
        <span id="bg-dim-val" style="font:.72rem var(--fco);color:var(--t1);min-width:28px">50%</span>
      </div>
    </div>

    <div class="form-group">
      <label class="form-label">💬 Kiểu bong bóng chat</label>
      <select class="form-select" id="bubble-style-input">
        <option value="default">Mặc định (bo tròn gradient)</option>
        <option value="flat">Flat (phẳng nhẹ)</option>
        <option value="minimal">Minimal (chỉ border)</option>
        <option value="glass">Glass morphism</option>
      </select>
    </div>

    <div class="form-group">
      <label class="form-label" style="display:flex;align-items:center;justify-content:space-between">
        ⌨️ Gửi bằng Enter
        <label style="display:flex;align-items:center;gap:6px;margin:0;cursor:pointer">
          <input type="checkbox" id="send-enter-toggle" checked
                 style="width:16px;height:16px;accent-color:var(--ac1)">
          <span style="font:.78rem var(--fui);color:var(--t1)">Bật</span>
        </label>
      </label>
      <div style="font:.7rem var(--fui);color:var(--t3)">Tắt = Chỉ gửi bằng nút ➤, Shift+Enter = xuống dòng</div>
    </div>


    <!-- ── Background Image ── -->
    <div class="bg-section">
      <div class="bg-section-title">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21,15 16,10 5,21"/></svg>
        Hình nền Background
      </div>

      <!-- Preview -->
      <div class="bg-preview-wrap" onclick="document.getElementById('bg-file-input').click()" title="Click để chọn ảnh">
        <img id="bg-preview-img" class="bg-preview-img" alt="preview">
        <div class="bg-preview-placeholder" id="bg-placeholder">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21,15 16,10 5,21"/></svg>
          Click để chọn ảnh từ máy tính
        </div>
        <div class="bg-preview-label" id="bg-preview-label">📷 Click để thay đổi</div>
      </div>
      <input type="file" id="bg-file-input" accept="image/*" onchange="onBgFileSelect(this)">

      <!-- Preset gradients -->
      <div style="font:.72rem var(--fui);color:var(--t3);margin-bottom:7px">Hoặc chọn gradient có sẵn:</div>
      <div class="bg-presets" id="bg-presets">
        <div class="bg-preset bg-preset-none active" data-preset="none" onclick="setBgPreset('none',this)" title="Không có">✕</div>
        <div class="bg-preset" data-preset="galaxy"
             style="background:linear-gradient(135deg,#0f0c29,#302b63,#24243e)"
             onclick="setBgPreset('galaxy',this)" title="Galaxy"></div>
        <div class="bg-preset" data-preset="aurora"
             style="background:linear-gradient(135deg,#000428,#004e92,#00b4db)"
             onclick="setBgPreset('aurora',this)" title="Aurora"></div>
        <div class="bg-preset" data-preset="sunset"
             style="background:linear-gradient(135deg,#f953c6,#b91d73,#ee0979)"
             onclick="setBgPreset('sunset',this)" title="Sunset"></div>
        <div class="bg-preset" data-preset="forest"
             style="background:linear-gradient(135deg,#0a3d0a,#1a6b1a,#2ecc71)"
             onclick="setBgPreset('forest',this)" title="Forest"></div>
        <div class="bg-preset" data-preset="ocean"
             style="background:linear-gradient(135deg,#001a2c,#005f73,#0096c7)"
             onclick="setBgPreset('ocean',this)" title="Ocean"></div>
        <div class="bg-preset" data-preset="volcano"
             style="background:linear-gradient(135deg,#1a0000,#7f1d1d,#dc2626)"
             onclick="setBgPreset('volcano',this)" title="Volcano"></div>
        <div class="bg-preset" data-preset="neon"
             style="background:linear-gradient(135deg,#0d0221,#26005f,#7b00d4)"
             onclick="setBgPreset('neon',this)" title="Neon"></div>
        <div class="bg-preset" data-preset="midnight"
             style="background:linear-gradient(135deg,#0a0a0a,#1a1a2e,#16213e)"
             onclick="setBgPreset('midnight',this)" title="Midnight"></div>
      </div>

      <!-- Opacity slider -->
      <div class="bg-slider-row">
        <span class="bg-slider-label">Độ sáng</span>
        <input type="range" class="bg-slider" id="bg-opacity"
               min="10" max="100" step="5" value="100"
               oninput="updateBgOpacity(this.value)">
        <span class="bg-slider-val" id="bg-opacity-val">100%</span>
      </div>

      <!-- Blur slider -->
      <div class="bg-slider-row">
        <span class="bg-slider-label">Làm mờ</span>
        <input type="range" class="bg-slider" id="bg-blur"
               min="0" max="20" step="1" value="0"
               oninput="updateBgBlur(this.value)">
        <span class="bg-slider-val" id="bg-blur-val">0px</span>
      </div>

      <!-- Overlay darkness -->
      <div class="bg-slider-row">
        <span class="bg-slider-label">Tối nền</span>
        <input type="range" class="bg-slider" id="bg-dark"
               min="0" max="90" step="5" value="50"
               oninput="updateBgDark(this.value)">
        <span class="bg-slider-val" id="bg-dark-val">50%</span>
      </div>

      <!-- Buttons -->
      <div class="bg-btn-row">
        <button class="bg-upload-btn" onclick="document.getElementById('bg-file-input').click()">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17,8 12,3 7,8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          Chọn ảnh từ máy
        </button>
        <button class="bg-remove-btn" onclick="removeBg()">✕ Xóa</button>
      </div>
    </div>


    <!-- ── Khung tin nhắn ── -->
    <div class="bg-section">
      <div class="bg-section-title">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        Khung tin nhắn AI
      </div>

      <!-- Preview -->
      <div class="bubble-preview" id="bubble-preview">
        <div class="bubble-preview-dot"></div>
        <div class="bubble-preview-text">Tin nhắn Hnhat AI sẽ hiển thị như thế này…</div>
      </div>

      <!-- Style grid -->
      <div class="bubble-style-grid">
        <div class="bs-btn active" data-bs="default" onclick="setBubbleStyle('default',this)">
          <div class="bs-name">⬜ Không khung</div>
          <div class="bs-desc">Mặc định, thoáng</div>
        </div>
        <div class="bs-btn" data-bs="card" onclick="setBubbleStyle('card',this)">
          <div class="bs-name">🌫️ Card tối</div>
          <div class="bs-desc">Khung mờ đen, dễ đọc</div>
        </div>
        <div class="bs-btn" data-bs="frost" onclick="setBubbleStyle('frost',this)">
          <div class="bs-name">❄️ Frost trắng</div>
          <div class="bs-desc">Khung trắng mờ</div>
        </div>
        <div class="bs-btn" data-bs="line" onclick="setBubbleStyle('line',this)">
          <div class="bs-name">📌 Viền trái</div>
          <div class="bs-desc">Gọn, accent tím</div>
        </div>
      </div>

      <!-- Bubble opacity -->
      <div class="bg-slider-row" style="margin-top:12px">
        <span class="bg-slider-label">Độ mờ</span>
        <input type="range" class="bg-slider" id="bubble-opacity-slider"
               min="10" max="100" step="5" value="65"
               oninput="updateBubbleOpacity(this.value)">
        <span class="bg-slider-val" id="bubble-opacity-val">65%</span>
      </div>
    </div>

    <button class="save-btn" onclick="saveSettings()">💾 Lưu cài đặt</button>
  </div>
</div>

<!-- ═══ SHORTCUTS MODAL ═══════════════════════════════════════ -->
<div id="shortcuts-modal">
  <div class="modal-backdrop" onclick="closeShortcuts()"></div>
  <div class="shortcuts-card">
    <button class="modal-close" onclick="closeShortcuts()">✕</button>
    <div class="modal-title" style="margin-bottom:18px">⌨️ Phím tắt</div>
    <div class="sc-row"><span class="sc-label">Gửi tin nhắn</span><div class="sc-key"><span class="kbd">Enter</span></div></div>
    <div class="sc-row"><span class="sc-label">Xuống dòng</span><div class="sc-key"><span class="kbd">Shift</span><span class="kbd">Enter</span></div></div>
    <div class="sc-row"><span class="sc-label">Chat mới</span><div class="sc-key"><span class="kbd">Ctrl</span><span class="kbd">K</span></div></div>
    <div class="sc-row"><span class="sc-label">Export chat</span><div class="sc-key"><span class="kbd">Ctrl</span><span class="kbd">E</span></div></div>
    <div class="sc-row"><span class="sc-label">Toggle sidebar</span><div class="sc-key"><span class="kbd">Ctrl</span><span class="kbd">\</span></div></div>
    <div class="sc-row"><span class="sc-label">Mở Settings</span><div class="sc-key"><span class="kbd">Ctrl</span><span class="kbd">,</span></div></div>
    <div class="sc-row"><span class="sc-label">Đóng modal</span><div class="sc-key"><span class="kbd">Esc</span></div></div>
    <div class="sc-row"><span class="sc-label">Đính kèm file</span><div class="sc-key"><span class="kbd">Ctrl</span><span class="kbd">U</span></div></div>
    <div class="sc-row"><span class="sc-label">Tìm kiếm</span><div class="sc-key"><span class="kbd">Ctrl</span><span class="kbd">F</span></div></div>
    <div class="sc-row"><span class="sc-label">Prompt Library</span><div class="sc-key"><span class="kbd">Ctrl</span><span class="kbd">P</span></div></div>
    <div class="sc-row"><span class="sc-label">Voice input</span><div class="sc-key"><span class="kbd">Ctrl</span><span class="kbd">M</span></div></div>
  </div>
</div>


<!-- ═══ PROMPT LIBRARY MODAL ════════════════════════════════ -->
<div id="prompt-modal">
  <div class="modal-backdrop" onclick="closePromptLibrary()"></div>
  <div class="pm-card">
    <button class="modal-close" onclick="closePromptLibrary()">✕</button>
    <div class="modal-title">📚 Thư viện Prompt</div>
    <div class="pm-search">
      <span>🔍</span>
      <input id="pm-search-input" placeholder="Tìm prompt…" oninput="filterPrompts(this.value)">
    </div>
    <div class="pm-cats" id="pm-cats">
      <div class="pm-cat active" data-cat="all" onclick="setCat('all',this)">Tất cả</div>
      <div class="pm-cat" data-cat="code"    onclick="setCat('code',this)">💻 Code</div>
      <div class="pm-cat" data-cat="web"     onclick="setCat('web',this)">🌐 Web</div>
      <div class="pm-cat" data-cat="data"    onclick="setCat('data',this)">📊 Data</div>
      <div class="pm-cat" data-cat="devops"  onclick="setCat('devops',this)">🐳 DevOps</div>
      <div class="pm-cat" data-cat="write"   onclick="setCat('write',this)">✍️ Viết</div>
      <div class="pm-cat" data-cat="custom"  onclick="setCat('custom',this)">⭐ Của tôi</div>
    </div>
    <div class="pm-list" id="pm-list"></div>
    <div class="pm-add-row">
      <input id="pm-new-name" placeholder="Tên prompt…" style="max-width:140px">
      <input id="pm-new-text" placeholder="Nội dung prompt…">
      <button class="save-btn" style="width:auto;padding:8px 14px;margin:0"
              onclick="saveCustomPrompt()">+ Lưu</button>
    </div>
  </div>
</div>

<!-- ═══ STATS MODAL ══════════════════════════════════════════ -->
<div id="stats-modal">
  <div class="modal-backdrop" onclick="closeStats()"></div>
  <div class="stats-card">
    <button class="modal-close" onclick="closeStats()">✕</button>
    <div class="modal-title">📊 Thống kê</div>
    <div class="stat-grid" id="stat-grid">
      <div class="stat-box"><div class="stat-val" id="stat-total-chats">0</div><div class="stat-lbl">Tổng chat</div></div>
      <div class="stat-box"><div class="stat-val" id="stat-total-msgs">0</div><div class="stat-lbl">Tổng tin nhắn</div></div>
      <div class="stat-box"><div class="stat-val" id="stat-fav">0</div><div class="stat-lbl">Yêu thích</div></div>
      <div class="stat-box"><div class="stat-val" id="stat-tokens">0</div><div class="stat-lbl">Est. Tokens</div></div>
    </div>
    <div id="stat-model-bars"></div>
  </div>
</div>

<!-- ═══ RETRY MODEL PICKER ═══════════════════════════════════ -->
<div id="retry-picker" style="display:none;position:fixed;z-index:500;
     background:var(--bg-c);border:1px solid var(--b3);border-radius:var(--r);
     padding:8px;box-shadow:0 16px 50px rgba(0,0,0,.6);min-width:200px;">
  <div style="font:.72rem var(--fui);color:var(--t3);padding:4px 8px 8px">Thử lại với model:</div>
  <div id="retry-picker-list"></div>
</div>

<!-- Toast -->
<div id="toast"></div>

<!-- ═══ JAVASCRIPT ════════════════════════════════════════════ -->
<script>
"use strict";

/* ──────────────────────────────────────────────────────────
   STATE
────────────────────────────────────────────────────────── */
let CURRENT_CHAT_ID  = null;
let CURRENT_MODEL    = "Hnhat Pro";
let IS_BUSY          = false;
let ATTACHED_FILE    = null;   // {type, name, file_id?, b64?, mime?, size, lines?, tokens?, large?}
let ALL_CHATS        = [];
let TOAST_TIMER      = null;

const MODEL_COLORS = {
  "Hnhat Fast":    "#f59e0b",
  "Hnhat Pro":     "#8b5cf6",
  "Hnhat Master":  "#06b6d4",
  "Hnhat Code":    "#10b981",
  "Hnhat Reason":  "#f97316",
  "Hnhat Vision":  "#4285f4",
  "Hnhat Vision+": "#1a73e8",
  "Kimi K2":       "#0ea5e9",
  "Gemma 2":       "#14b8a6",
  "Compound":      "#a855f7",
};

const VISION_MODELS = new Set(["Hnhat Vision", "Hnhat Vision+"]);

/* ──────────────────────────────────────────────────────────
   INIT
────────────────────────────────────────────────────────── */

/* ══════════════════════════════════════════════════════════
   🖼️ BACKGROUND SYSTEM — gradients + custom image
══════════════════════════════════════════════════════════ */

// State object
let _bgState = {
  type:    "none",   // "none" | "image" | "preset"
  data:    null,     // base64 dataURL for images
  preset:  null,     // gradient CSS string for presets
  opacity: 100,      // 10-100
  blur:    0,        // 0-20px
  dark:    50,       // 0-90 overlay darkness %
};

const BG_PRESETS = {
  galaxy:   "linear-gradient(135deg,#0f0c29,#302b63,#24243e)",
  aurora:   "linear-gradient(135deg,#000428,#004e92,#00b4db)",
  sunset:   "linear-gradient(135deg,#f953c6,#b91d73,#ee0979)",
  forest:   "linear-gradient(135deg,#0a3d0a,#1a6b1a,#2ecc71)",
  ocean:    "linear-gradient(135deg,#001a2c,#005f73,#0096c7)",
  volcano:  "linear-gradient(135deg,#1a0000,#7f1d1d,#dc2626)",
  neon:     "linear-gradient(135deg,#0d0221,#26005f,#7b00d4)",
  midnight: "linear-gradient(135deg,#0a0a0a,#1a1a2e,#16213e)",
};

/* Apply current _bgState to the DOM */
function applyBgState() {
  const layer   = document.getElementById("bg-layer");
  const overlay = document.getElementById("bg-overlay");
  if (!layer || !overlay) return;

  if (_bgState.type === "none") {
    layer.classList.remove("active");
    layer.style.backgroundImage = "";
    layer.style.filter = "";
    layer.style.transform = "";
    overlay.style.background = "";
    document.body.classList.remove("has-bg");
    return;
  }

  let bgValue = "";
  if (_bgState.type === "image" && _bgState.data) {
    bgValue = `url("${_bgState.data}")`;
    layer.style.backgroundSize     = "cover";
    layer.style.backgroundPosition = "center";
    layer.style.backgroundRepeat   = "no-repeat";
  } else if (_bgState.type === "preset" && _bgState.preset) {
    bgValue = _bgState.preset;
    layer.style.backgroundSize = "100% 100%";
  }

  layer.style.backgroundImage = bgValue;
  layer.style.opacity   = _bgState.opacity / 100;
  layer.style.filter    = _bgState.blur > 0 ? `blur(${_bgState.blur}px)` : "";
  layer.style.transform = _bgState.blur > 0 ? `scale(1.04)` : "";

  const d = _bgState.dark / 100;
  overlay.style.background = `rgba(0,0,0,${d.toFixed(2)})`;

  layer.classList.add("active");
  document.body.classList.add("has-bg");
}

/* File select from disk */
function onBgFileSelect(input) {
  const file = input.files[0];
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) {
    showToast("❌ Ảnh quá lớn! Tối đa 8MB", "error");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    _bgState.type   = "image";
    _bgState.data   = e.target.result;
    _bgState.preset = null;
    _showPreview(_bgState.data);
    _clearPresetActive();
    saveBgState();
    applyBgState();
    showToast("🖼️ Đã đặt hình nền!", "success");
  };
  reader.readAsDataURL(file);
  input.value = "";
}

/* Select gradient preset */
function setBgPreset(name, el) {
  // Mark active swatch
  document.querySelectorAll(".bg-preset").forEach(e => e.classList.remove("active"));
  if (el) el.classList.add("active");

  if (name === "none") {
    removeBg();
    return;
  }

  const grad = BG_PRESETS[name];
  if (!grad) { showToast("❌ Preset không hợp lệ", "error"); return; }

  _bgState.type   = "preset";
  _bgState.preset = grad;
  _bgState.data   = null;

  // Clear image preview
  _clearPreview();
  saveBgState();
  applyBgState();
  showToast("🎨 " + name.charAt(0).toUpperCase() + name.slice(1), "success");
}

/* Slider handlers */
function updateBgOpacity(val) {
  _bgState.opacity = parseInt(val);
  const el = document.getElementById("bg-opacity-val");
  if (el) el.textContent = val + "%";
  applyBgState();
  saveBgState();
}
function updateBgBlur(val) {
  _bgState.blur = parseInt(val);
  const el = document.getElementById("bg-blur-val");
  if (el) el.textContent = val + "px";
  applyBgState();
  saveBgState();
}
function updateBgDark(val) {
  _bgState.dark = parseInt(val);
  const el = document.getElementById("bg-dark-val");
  if (el) el.textContent = val + "%";
  applyBgState();
  saveBgState();
}

/* Remove background */
function removeBg() {
  _bgState = { type:"none", data:null, preset:null, opacity:100, blur:0, dark:50 };
  _clearPreview();
  _clearPresetActive();
  const none = document.querySelector('.bg-preset[data-preset="none"]');
  if (none) none.classList.add("active");
  applyBgState();
  saveBgState();
  showToast("🗑️ Đã xóa hình nền");
}

/* Save to localStorage */
function saveBgState() {
  try {
    const meta = {
      type:    _bgState.type,
      preset:  _bgState.preset,
      opacity: _bgState.opacity,
      blur:    _bgState.blur,
      dark:    _bgState.dark,
    };
    localStorage.setItem("hnhat-bg-meta", JSON.stringify(meta));
    if (_bgState.type === "image" && _bgState.data) {
      localStorage.setItem("hnhat-bg-img", _bgState.data);
    } else {
      localStorage.removeItem("hnhat-bg-img");
    }
  } catch(e) {
    if (e.name === "QuotaExceededError") {
      showToast("⚠️ Ảnh quá lớn cho bộ nhớ. Hãy chọn ảnh nhỏ hơn.", "error");
    }
  }
}

/* Load from localStorage */
function loadBgState() {
  try {
    const raw = localStorage.getItem("hnhat-bg-meta");
    if (!raw) return;
    const meta = JSON.parse(raw);
    if (!meta || !meta.type || meta.type === "none") return;

    _bgState.type    = meta.type    || "none";
    _bgState.preset  = meta.preset  || null;
    _bgState.opacity = meta.opacity ?? 100;
    _bgState.blur    = meta.blur    ?? 0;
    _bgState.dark    = meta.dark    ?? 50;

    if (meta.type === "image") {
      _bgState.data = localStorage.getItem("hnhat-bg-img") || null;
    }
    applyBgState();
  } catch(e) {
    console.warn("BG load error:", e);
  }
}

/* Sync UI sliders + preview when settings opens */
function syncBgSliders() {
  const op  = document.getElementById("bg-opacity");
  const bl  = document.getElementById("bg-blur");
  const dk  = document.getElementById("bg-dark");
  const opv = document.getElementById("bg-opacity-val");
  const blv = document.getElementById("bg-blur-val");
  const dkv = document.getElementById("bg-dark-val");

  if (op)  { op.value  = _bgState.opacity; }
  if (opv) { opv.textContent = _bgState.opacity + "%"; }
  if (bl)  { bl.value  = _bgState.blur; }
  if (blv) { blv.textContent = _bgState.blur + "px"; }
  if (dk)  { dk.value  = _bgState.dark; }
  if (dkv) { dkv.textContent = _bgState.dark + "%"; }

  // Sync preview
  if (_bgState.type === "image" && _bgState.data) {
    _showPreview(_bgState.data);
  } else {
    _clearPreview();
  }

  // Sync preset active state
  _clearPresetActive();
  if (_bgState.type === "preset" && _bgState.preset) {
    const key = Object.keys(BG_PRESETS).find(k => BG_PRESETS[k] === _bgState.preset);
    if (key) {
      const el = document.querySelector(`.bg-preset[data-preset="${key}"]`);
      if (el) el.classList.add("active");
    }
  } else if (_bgState.type === "none") {
    const el = document.querySelector('.bg-preset[data-preset="none"]');
    if (el) el.classList.add("active");
  }
}

/* Internal helpers */
function _showPreview(dataUrl) {
  const img = document.getElementById("bg-preview-img");
  const ph  = document.getElementById("bg-placeholder");
  if (img) { img.src = dataUrl; img.classList.add("visible"); }
  if (ph)  { ph.style.display = "none"; }
}
function _clearPreview() {
  const img = document.getElementById("bg-preview-img");
  const ph  = document.getElementById("bg-placeholder");
  if (img) { img.src = ""; img.classList.remove("visible"); }
  if (ph)  { ph.style.display = ""; }
}
function _clearPresetActive() {
  document.querySelectorAll(".bg-preset").forEach(e => e.classList.remove("active"));
}


/* ══════════════════════════════════════════════════════════
   💬 BUBBLE FRAME SYSTEM
══════════════════════════════════════════════════════════ */
let _bubbleStyle   = "default";
let _bubbleOpacity = 65;

function setBubbleStyle(style, el) {
  _bubbleStyle = style;
  document.body.classList.remove("bubble-card","bubble-frost","bubble-line");
  if (style !== "default") document.body.classList.add("bubble-" + style);

  // Update buttons
  document.querySelectorAll(".bs-btn").forEach(e => e.classList.remove("active"));
  if (el) el.classList.add("active");

  // Update preview
  updateBubblePreview();
  localStorage.setItem("hnhat-bubble", style);
  showToast("🖼️ Kiểu khung: " + style);
}

function updateBubbleOpacity(val) {
  _bubbleOpacity = parseInt(val);
  const vEl = document.getElementById("bubble-opacity-val");
  if (vEl) vEl.textContent = val + "%";
  _applyBubbleOpacity();
  localStorage.setItem("hnhat-bubble-opacity", val);
}

function _applyBubbleOpacity() {
  const pct = _bubbleOpacity / 100;
  // Dynamic CSS variable for bubble bg
  document.documentElement.style.setProperty(
    "--bubble-bg-opacity", pct.toFixed(2)
  );
  // Update existing card/frost backgrounds
  const rule = _bubbleStyle === "frost"
    ? `rgba(255,255,255,${(pct * 0.12).toFixed(2)})`
    : `rgba(10,10,25,${(pct * 0.75).toFixed(2)})`;
  document.querySelectorAll("body.bubble-card .msg-group:not(.user), body.bubble-frost .msg-group:not(.user)")
    .forEach(el => { el.style.background = rule; });
}

function updateBubblePreview() {
  const p = document.getElementById("bubble-preview");
  if (!p) return;
  if (_bubbleStyle === "card") {
    p.style.background = `rgba(10,10,25,${(_bubbleOpacity/100*0.65).toFixed(2)})`;
    p.style.backdropFilter = "blur(14px)";
    p.style.border = "1px solid rgba(255,255,255,.1)";
  } else if (_bubbleStyle === "frost") {
    p.style.background = `rgba(255,255,255,${(_bubbleOpacity/100*0.1).toFixed(2)})`;
    p.style.backdropFilter = "blur(20px)";
    p.style.border = "1px solid rgba(255,255,255,.15)";
  } else if (_bubbleStyle === "line") {
    p.style.background = "rgba(88,101,242,.06)";
    p.style.backdropFilter = "";
    p.style.border = "none";
    p.style.borderLeft = "3px solid #5865f2";
  } else {
    p.style.background = "";
    p.style.backdropFilter = "";
    p.style.border = "1px solid var(--b1)";
  }
}

function restoreBubblePrefs() {
  const style   = localStorage.getItem("hnhat-bubble") || "default";
  const opacity = parseInt(localStorage.getItem("hnhat-bubble-opacity") || "65");
  _bubbleStyle   = style;
  _bubbleOpacity = opacity;

  document.body.classList.remove("bubble-card","bubble-frost","bubble-line");
  if (style !== "default") document.body.classList.add("bubble-" + style);

  // Sync UI
  const opEl = document.getElementById("bubble-opacity-slider");
  const opVl = document.getElementById("bubble-opacity-val");
  if (opEl) opEl.value = opacity;
  if (opVl) opVl.textContent = opacity + "%";

  document.querySelectorAll(".bs-btn").forEach(e => {
    e.classList.toggle("active", e.dataset.bs === style);
  });
  updateBubblePreview();
}

/* ══════════════════════════════════════════════════════════
   🔧 FIX: Background sliders — direct DOM manipulation
   (belt-and-suspenders: update both state AND element)
══════════════════════════════════════════════════════════ */
// Override updateBgOpacity/Blur/Dark with robust versions
function updateBgOpacity(val) {
  _bgState.opacity = parseInt(val);
  const vEl = document.getElementById("bg-opacity-val");
  if (vEl) vEl.textContent = val + "%";
  const layer = document.getElementById("bg-layer");
  if (layer && _bgState.type !== "none") layer.style.opacity = _bgState.opacity / 100;
  saveBgState();
}

function updateBgBlur(val) {
  _bgState.blur = parseInt(val);
  const vEl = document.getElementById("bg-blur-val");
  if (vEl) vEl.textContent = val + "px";
  const layer = document.getElementById("bg-layer");
  if (layer && _bgState.type !== "none") {
    layer.style.filter    = val > 0 ? `blur(${val}px)` : "";
    layer.style.transform = val > 0 ? "scale(1.04)" : "";
  }
  saveBgState();
}

function updateBgDark(val) {
  _bgState.dark = parseInt(val);
  const vEl = document.getElementById("bg-dark-val");
  if (vEl) vEl.textContent = val + "%";
  const overlay = document.getElementById("bg-overlay");
  if (overlay) overlay.style.background = `rgba(0,0,0,${(parseInt(val)/100).toFixed(2)})`;
  saveBgState();
}


/* ══════════════════════════════════════════════════════════
   📐 KATEX — render math trong tin nhắn
══════════════════════════════════════════════════════════ */
function renderMathInDocument() {
  if (typeof renderMathInElement !== "function") return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true  },
      { left: "$",  right: "$",  display: false },
      { left: "\\[", right: "\\]", display: true  },
      { left: "\\(", right: "\\)", display: false },
    ],
    throwOnError: false,
    errorColor: "#ff6b6b",
  });
}

function renderMathIn(el) {
  if (!el) return;
  if (typeof renderMathInElement !== "function") return;
  try {
    renderMathInElement(el, {
      delimiters: [
        { left: "$$", right: "$$", display: true  },
        { left: "$",  right: "$",  display: false },
        { left: "\\[", right: "\\]", display: true  },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  } catch(e) {
    console.warn("KaTeX render error:", e);
  }
}


/* ══════════════════════════════════════════════════════════
   ❌ API ERROR — hiện toast đẹp, không thêm vào chat
══════════════════════════════════════════════════════════ */
function showApiError(msg, model) {
  // Extract readable message from Groq error
  let clean = msg;

  // Parse JSON error message if needed
  if (typeof msg === "string") {
    const m = msg.match(/'message':\s*'([^']+)'/);
    if (m) clean = m[1];
    // Decommissioned model
    if (msg.includes("decommissioned")) {
      clean = `Model ${model} không còn hoạt động. Hãy chọn model khác trong menu.`;
    }
    // Auth error
    if (msg.includes("401") || msg.includes("api_key")) {
      clean = "API Key không hợp lệ. Vào ⚙️ Cài đặt để kiểm tra lại.";
    }
    // Rate limit
    if (msg.includes("429") || msg.includes("rate_limit")) {
      clean = "Quá giới hạn request. Hãy đợi vài giây rồi thử lại.";
    }
    // Context too long
    if (msg.includes("context") && msg.includes("long")) {
      clean = "Tin nhắn quá dài. Hãy tạo chat mới (Ctrl+K).";
    }
  }

  // Show big error toast
  const el = document.getElementById("toast");
  el.innerHTML = `<span style="font-size:1rem">⚠️</span> ${clean}`;
  el.className = "show error";
  clearTimeout(TOAST_TIMER);
  TOAST_TIMER = setTimeout(() => el.classList.remove("show"), 6000);

  // Also log to console for debugging
  console.error("[Hnhat AI Error]", msg);
}

async function boot() {
  try {
    const cfg = await apiGet("/api/config");
    CURRENT_MODEL = cfg.default_model || "Hnhat Pro";
    setModelUI(CURRENT_MODEL);
    if (cfg.theme === "light") document.documentElement.setAttribute("data-theme", "light");
    if (!cfg.has_groq) setTimeout(openSettings, 800);
  } catch(e) {
    console.warn("Config load failed:", e);
  }

  await loadChatList();
  setupGlobalDrag();
  setupChatTitleRename();
  restoreLocalPrefs();
  loadBgState();
  restoreBubblePrefs();
}

/* ──────────────────────────────────────────────────────────
   CHAT LIST
────────────────────────────────────────────────────────── */
async function loadChatList() {
  try {
    ALL_CHATS = await apiGet("/api/chats");
    renderChatList(ALL_CHATS);
  } catch(e) {
    console.warn("Chat list load failed:", e);
  }
}

function renderChatList(chats) {
  const el = document.getElementById("chat-list");
  if (!chats.length) {
    el.innerHTML = '<div style="padding:14px;font-size:.74rem;color:var(--t3);text-align:center">Chưa có cuộc trò chuyện nào</div>';
    return;
  }
  el.innerHTML = chats.map(c => `
    <div class="chat-item ${c.id === CURRENT_CHAT_ID ? "active" : ""}"
         id="ci-${escAttr(c.id)}" onclick="loadChat('${escAttr(c.id)}')">
      <span class="ci-icon">💬</span>
      <div class="ci-body">
        <div class="ci-title">${escHtml(c.title)}</div>
        <div class="ci-meta">${escHtml(c.model || "Pro")} · ${relTime(c.updated)} · ${c.count} tin</div>
      </div>
      <button class="ci-star ${c.favorite ? 'starred' : ''}" onclick="toggleFavorite('${escAttr(c.id)}', event)" title="Yêu thích">⭐</button>
      <button class="ci-del" onclick="deleteChat('${escAttr(c.id)}', event)" title="Xóa">✕</button>
    </div>
  `).join("");
}

function filterChats(query) {
  const q = query.toLowerCase().trim();
  renderChatList(q ? ALL_CHATS.filter(c => c.title.toLowerCase().includes(q)) : ALL_CHATS);
}

async function loadChat(id) {
  try {
    const data = await apiGet(`/api/chats/${id}`);
    if (!data || data.error) return;

    CURRENT_CHAT_ID = id;
    CURRENT_MODEL   = data.model || "Hnhat Pro";
    setModelUI(CURRENT_MODEL);

    document.getElementById("chat-title").textContent = data.title || "Chat";

    const msgEl  = document.getElementById("messages");
    const welEl  = document.getElementById("welcome");
    msgEl.innerHTML = "";

    if (data.messages && data.messages.length) {
      welEl.style.display  = "none";
      msgEl.style.display  = "block";
      data.messages.forEach(m => addMessageToDOM(m.role, m.content, false, m.model || CURRENT_MODEL));
    } else {
      welEl.style.display  = "flex";
      msgEl.style.display  = "none";
    }

    document.querySelectorAll(".chat-item").forEach(el => el.classList.remove("active"));
    document.getElementById("ci-" + id)?.classList.add("active");
    scrollToBottom();
  } catch(e) {
    showToast("❌ Lỗi tải chat: " + e.message);
  }
}

function newChat() {
  CURRENT_CHAT_ID = null;
  ATTACHED_FILE   = null;
  clearAttachPreview();

  document.getElementById("welcome").style.display    = "flex";
  const msgEl = document.getElementById("messages");
  msgEl.style.display  = "none";
  msgEl.innerHTML      = "";
  document.getElementById("chat-title").textContent   = "Hnhat AI";
  document.getElementById("token-count").textContent  = "";
  document.querySelectorAll(".chat-item").forEach(el => el.classList.remove("active"));
}

async function deleteChat(id, event) {
  event.stopPropagation();
  if (!confirm("Xóa cuộc trò chuyện này?")) return;
  await apiCall(`/api/chats/${id}`, "DELETE");
  if (CURRENT_CHAT_ID === id) newChat();
  await loadChatList();
  showToast("🗑️ Đã xóa cuộc trò chuyện");
}

function confirmClearChat() {
  if (!CURRENT_CHAT_ID) return;
  if (!confirm("Xóa toàn bộ tin nhắn trong chat này?")) return;
  clearCurrentChat();
}

async function clearCurrentChat() {
  if (!CURRENT_CHAT_ID) return;
  await apiPost(`/api/chats/${CURRENT_CHAT_ID}/clear`, {});
  const msgEl = document.getElementById("messages");
  msgEl.innerHTML = "";
  msgEl.style.display = "none";
  document.getElementById("welcome").style.display = "flex";
  document.getElementById("chat-title").textContent = "Hnhat AI";
  await loadChatList();
  showToast("🧹 Đã xóa messages");
}

/* ──────────────────────────────────────────────────────────
   CHAT TITLE RENAME
────────────────────────────────────────────────────────── */
function setupChatTitleRename() {
  const el = document.getElementById("chat-title");
  el.addEventListener("dblclick", () => {
    el.contentEditable = "true";
    el.focus();
    const range = document.createRange();
    range.selectNodeContents(el);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
  });
  el.addEventListener("blur", async () => {
    el.contentEditable = "false";
    if (!CURRENT_CHAT_ID) return;
    const title = el.textContent.trim() || "Chat";
    el.textContent = title;
    await apiPost(`/api/chats/${CURRENT_CHAT_ID}/rename`, { title });
    await loadChatList();
  });
  el.addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); el.blur(); }
  });
}

/* ──────────────────────────────────────────────────────────
   SEND MESSAGE
────────────────────────────────────────────────────────── */
async function sendMessage() {
  if (IS_BUSY) return;

  const inputEl  = document.getElementById("user-input");
  const userText = inputEl.value.trim();
  if (!userText && !ATTACHED_FILE) return;

  inputEl.value = "";
  onInputChange(inputEl);

  // Show chat
  document.getElementById("welcome").style.display = "none";
  const msgEl = document.getElementById("messages");
  msgEl.style.display = "block";

  // Create new chat if needed
  if (!CURRENT_CHAT_ID) {
    try {
      const c = await apiPost("/api/chats", { model: CURRENT_MODEL });
      CURRENT_CHAT_ID = c.id;
      await loadChatList();
    } catch(e) {
      showToast("❌ Không thể tạo chat: " + e.message);
      return;
    }
  }

  // Display user message
  const displayText = userText || (ATTACHED_FILE ? `[📎 ${ATTACHED_FILE.name}]` : "");
  addMessageToDOM("user", displayText, false, CURRENT_MODEL);

  // Snapshot and clear attachment
  const fileMeta    = ATTACHED_FILE;
  ATTACHED_FILE     = null;
  clearAttachPreview();
  document.getElementById("token-count").textContent = "";

  IS_BUSY = true;
  document.getElementById("send-btn").disabled = true;

  // Route based on model and file
  const isVision    = VISION_MODELS.has(CURRENT_MODEL);
  const isCodeModel = CURRENT_MODEL === "Hnhat Code";
  const isLargeFile = fileMeta && fileMeta.type === "text" && fileMeta.large;

  if (isVision && fileMeta && fileMeta.type === "image") {
    await streamVision(userText, fileMeta);
  } else if (isCodeModel && isLargeFile) {
    await streamCodeAnalyze(userText, fileMeta);
  } else {
    await streamGroqChat(userText, fileMeta);
  }

  IS_BUSY = false;
  document.getElementById("send-btn").disabled = false;
}

/* ── Groq streaming chat ── */
async function streamGroqChat(userText, fileMeta) {
  const thinkId = insertThinkingIndicator();
  let aiGroup   = null;
  let contentEl = null;
  let fullText  = "";

  try {
    const body = {
      chat_id:  CURRENT_CHAT_ID,
      message:  userText,
      model:    CURRENT_MODEL,
      file:     fileMeta || null,
    };

    const reader = await startSSE("/api/chat/stream", body);
    const dec    = new TextDecoder();
    let   buf    = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (raw === "[DONE]") continue;
        let parsed;
        try { parsed = JSON.parse(raw); } catch { continue; }

        if (parsed.error) {
          removeElement(thinkId);
          showApiError(parsed.error, CURRENT_MODEL);
          return;
        }
        if (parsed.content) {
          if (!aiGroup) {
            removeElement(thinkId);
            const result = createStreamingAIGroup(CURRENT_MODEL);
            aiGroup   = result.group;
            contentEl = result.contentEl;
            document.getElementById("messages").appendChild(aiGroup);
          }
          fullText += parsed.content;
          contentEl.innerHTML = renderMarkdown(fullText);
          highlightCodeBlocks(contentEl);
          if (typeof renderMathIn === "function") renderMathIn(contentEl);
          scrollToBottom();
        }
        if (parsed.done) {
          if (parsed.chat_id) CURRENT_CHAT_ID = parsed.chat_id;
          if (aiGroup) {
            const actEl = aiGroup.querySelector(".msg-actions");
            if (actEl) actEl.innerHTML = buildActionButtons(fullText);
          }
          await loadChatList();
          updateChatTitleFromList();
        }
      }
    }
  } catch(e) {
    removeElement(thinkId);
    showApiError("Lỗi kết nối: " + e.message, CURRENT_MODEL);
  }
}

/* ── Gemini Vision streaming ── */
async function streamVision(userText, fileMeta) {
  const thinkId = insertThinkingIndicator("Đang phân tích ảnh với Hnhat");
  let aiGroup   = null;
  let contentEl = null;
  let fullText  = "";

  try {
    const body = {
      chat_id:  CURRENT_CHAT_ID,
      b64:      fileMeta.b64,
      mime:     fileMeta.mime,
      prompt:   userText || "Phân tích chi tiết hình ảnh này",
      user_msg: userText || "Phân tích ảnh",
      model:    CURRENT_MODEL,
    };

    const reader = await startSSE("/api/vision", body);
    const dec    = new TextDecoder();
    let   buf    = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (raw === "[DONE]") continue;
        let parsed;
        try { parsed = JSON.parse(raw); } catch { continue; }

        if (parsed.error) {
          removeElement(thinkId);
          showApiError(parsed.error, CURRENT_MODEL);
          return;
        }
        if (parsed.content) {
          if (!aiGroup) {
            removeElement(thinkId);
            const result = createStreamingAIGroup(CURRENT_MODEL);
            aiGroup   = result.group;
            contentEl = result.contentEl;
            document.getElementById("messages").appendChild(aiGroup);
          }
          fullText += parsed.content;
          contentEl.innerHTML = renderMarkdown(fullText);
          highlightCodeBlocks(contentEl);
          if (typeof renderMathIn === "function") renderMathIn(contentEl);
          scrollToBottom();
        }
        if (parsed.done) {
          if (aiGroup) {
            const actEl = aiGroup.querySelector(".msg-actions");
            if (actEl) actEl.innerHTML = buildActionButtons(fullText);
          }
          await loadChatList();
        }
      }
    }
  } catch(e) {
    removeElement(thinkId);
    showApiError("Gemini lỗi: " + e.message, CURRENT_MODEL);
  }
}

/* ── Code chunked analysis ── */
async function streamCodeAnalyze(userText, fileMeta) {
  const msgEl   = document.getElementById("messages");
  const thinkId = insertThinkingIndicator("Đang chuẩn bị phân tích file lớn…");

  // Progress block
  const progressId = "prog-" + Date.now();
  const progEl = document.createElement("div");
  progEl.className = "msg-group";
  progEl.id = progressId;
  progEl.innerHTML = `
    <div class="bubble-ai">
      <div class="ai-header">
        <div class="ai-avatar">💻</div>
        <span class="ai-who">Hnhat Code</span>
        <span class="ai-model-tag">Chunked Analysis</span>
      </div>
      <div class="progress-wrap" id="${progressId}-wrap">
        <div class="progress-label" id="${progressId}-label">Khởi động…</div>
        <div class="progress-bar-bg"><div class="progress-bar-fill" id="${progressId}-fill" style="width:0%"></div></div>
        <div class="progress-pct" id="${progressId}-pct">0%</div>
      </div>
    </div>`;
  msgEl.appendChild(progEl);
  removeElement(thinkId);
  scrollToBottom();

  let aiGroup   = null;
  let contentEl = null;
  let fullText  = "";

  try {
    const body = {
      file_id:  fileMeta.file_id,
      filename: fileMeta.name,
      question: userText || "Phân tích toàn bộ code này, tóm tắt cấu trúc và chức năng chính",
      chat_id:  CURRENT_CHAT_ID,
      model:    CURRENT_MODEL,
    };

    const reader = await startSSE("/api/code/analyze", body);
    const dec    = new TextDecoder();
    let   buf    = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (raw === "[DONE]") continue;
        let parsed;
        try { parsed = JSON.parse(raw); } catch { continue; }

        if (parsed.error) {
          removeElement(progressId);
          addMessageToDOM("assistant", "❌ " + parsed.error, true, CURRENT_MODEL);
          return;
        }

        // Progress update
        if (parsed.progress) {
          const p = parsed.progress;
          const pLabel = document.getElementById(progressId + "-label");
          const pFill  = document.getElementById(progressId + "-fill");
          const pPct   = document.getElementById(progressId + "-pct");
          if (pLabel) pLabel.textContent = p.label || "";
          if (pFill)  pFill.style.width  = p.pct + "%";
          if (pPct)   pPct.textContent   = p.pct + "%";
          scrollToBottom();
          continue;
        }

        // Info/status line
        if (parsed.info) {
          const pLabel = document.getElementById(progressId + "-label");
          if (pLabel) pLabel.textContent = parsed.info;
          scrollToBottom();
          continue;
        }

        // Stream content
        if (parsed.content) {
          if (!aiGroup) {
            removeElement(progressId);
            const result = createStreamingAIGroup(CURRENT_MODEL);
            aiGroup   = result.group;
            contentEl = result.contentEl;
            msgEl.appendChild(aiGroup);
          }
          fullText += parsed.content;
          contentEl.innerHTML = renderMarkdown(fullText);
          highlightCodeBlocks(contentEl);
          if (typeof renderMathIn === "function") renderMathIn(contentEl);
          scrollToBottom();
        }

        if (parsed.done) {
          removeElement(progressId);
          if (aiGroup) {
            const actEl = aiGroup.querySelector(".msg-actions");
            if (actEl) actEl.innerHTML = buildActionButtons(fullText);
          }
          await loadChatList();
          updateChatTitleFromList();
        }
      }
    }
  } catch(e) {
    removeElement(progressId);
    showApiError("Code analysis lỗi: " + e.message, CURRENT_MODEL);
  }
}

/* ──────────────────────────────────────────────────────────
   DOM HELPERS
────────────────────────────────────────────────────────── */
function addMessageToDOM(role, content, withActions, model) {
  const msgEl = document.getElementById("messages");
  const div   = document.createElement("div");
  div.className = "msg-group" + (role === "user" ? " user" : "");

  if (role === "user") {
    div.innerHTML = `
      <div class="bubble-user">
        <div class="msg-content">${escHtml(content)}</div>
        <div class="bubble-ts">${nowTime()}</div>
      </div>`;
  } else {
    const actions = withActions ? buildActionButtons(content) : "";
    div.innerHTML = `
      <div class="bubble-ai">
        <div class="ai-header">
          <div class="ai-avatar"><img src="/icon.png" style="width:100%;height:100%;border-radius:6px;object-fit:cover"></div>
          <span class="ai-who">Hnhat AI</span>
          ${model ? `<span class="ai-model-tag">${escHtml(model)}</span>` : ""}
          <span class="ai-timestamp">${nowTime()}</span>
        </div>
        <div class="msg-content">${renderMarkdown(content)}</div>
      </div>
      <div class="msg-actions">${actions}</div>`;
    setTimeout(() => { highlightCodeBlocks(div); if (typeof renderMathIn === "function") renderMathIn(div); }, 0);
  }

  msgEl.appendChild(div);
  scrollToBottom();
  return div;
}

function createStreamingAIGroup(model) {
  const group = document.createElement("div");
  group.className = "msg-group";
  group.innerHTML = `
    <div class="bubble-ai">
      <div class="ai-header">
        <div class="ai-avatar"><img src="/icon.png" style="width:100%;height:100%;border-radius:6px;object-fit:cover"></div>
        <span class="ai-who">Hnhat AI</span>
        <span class="ai-model-tag">${escHtml(model)}</span>
        <span class="ai-timestamp">${nowTime()}</span>
      </div>
      <div class="msg-content" id="streaming-mc"></div>
    </div>
    <div class="msg-actions"></div>`;
  const contentEl = group.querySelector("#streaming-mc");
  contentEl.removeAttribute("id");
  return { group, contentEl };
}

function insertThinkingIndicator(text) {
  const id  = "think-" + Date.now();
  const div = document.createElement("div");
  div.className = "msg-group";
  div.id = id;
  div.innerHTML = `
    <div class="bubble-ai">
      <div class="thinking-row">
        <div class="ai-avatar"><img src="/icon.png" style="width:100%;height:100%;border-radius:6px;object-fit:cover"></div>
        <div class="thinking-dots">
          <div class="thinking-dot"></div>
          <div class="thinking-dot"></div>
          <div class="thinking-dot"></div>
        </div>
        <span class="thinking-text">${escHtml(text || "Đang suy nghĩ…")}</span>
      </div>
    </div>`;
  document.getElementById("messages").appendChild(div);
  scrollToBottom();
  return id;
}

function buildActionButtons(text) {
  const safe = encodeB64(text);
  return `
    <button class="act-btn" onclick="copyText(this, '${safe}')">📋 Sao chép</button>
    <button class="act-btn" onclick="regenerateLastMessage()">🔄 Làm lại</button>
    <button class="act-btn" id="tts-btn-${Date.now()}" onclick="toggleTTS(this, '${safe}')">🔊 Đọc</button>
    <button class="act-btn" onclick="showRetryPicker(this, '${safe}')">🔀 Thử model</button>
    <button class="act-btn" onclick="copyAsMarkdown(this, '${safe}')">📄 MD</button>
  `;
}

function removeElement(id) {
  document.getElementById(id)?.remove();
}

function updateChatTitleFromList() {
  const el = document.querySelector(`#ci-${CURRENT_CHAT_ID} .ci-title`);
  if (el) document.getElementById("chat-title").textContent = el.textContent;
}

/* ──────────────────────────────────────────────────────────
   REGENERATE
────────────────────────────────────────────────────────── */
async function regenerateLastMessage() {
  if (!CURRENT_CHAT_ID || IS_BUSY) return;
  try {
    const data   = await apiGet(`/api/chats/${CURRENT_CHAT_ID}`);
    const msgs   = data.messages || [];
    const lastU  = [...msgs].reverse().find(m => m.role === "user");
    if (!lastU) return;

    const msgEl  = document.getElementById("messages");
    const groups = msgEl.querySelectorAll(".msg-group");
    if (groups.length) groups[groups.length - 1].remove();

    IS_BUSY = true;
    document.getElementById("send-btn").disabled = true;
    await streamGroqChat(lastU.content, null);
  } catch(e) {
    showToast("❌ Lỗi: " + e.message);
  } finally {
    IS_BUSY = false;
    document.getElementById("send-btn").disabled = false;
  }
}

/* ──────────────────────────────────────────────────────────
   FILE UPLOAD
────────────────────────────────────────────────────────── */
function onFileSelect(input) {
  const f = input.files[0];
  if (f) uploadFile(f);
  input.value = "";
}

async function uploadFile(file) {
  showToast("⏳ Đang tải file…");
  const fd = new FormData();
  fd.append("file", file);

  try {
    const res  = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();

    if (data.error) {
      showToast("❌ " + data.error);
      return;
    }

    ATTACHED_FILE = data;
    showAttachPreview(data);

    const sizeKB = Math.round((data.size || 0) / 1024);
    showToast(`✅ ${data.name} (${sizeKB}KB)`);

    // Auto-switch model
    if (data.type === "image" && !VISION_MODELS.has(CURRENT_MODEL)) {
      setModelUI("Hnhat Vision");
      showToast("👁 Đã chuyển sang Hnhat Vision (Gemini)");
    } else if (data.type === "text" && CURRENT_MODEL !== "Hnhat Code") {
      setModelUI("Hnhat Code");
      showToast("💻 Đã chuyển sang Hnhat Code");
    }
  } catch(e) {
    showToast("❌ Upload lỗi: " + e.message);
  }
}

function showAttachPreview(data) {
  const apEl = document.getElementById("attach-preview");
  apEl.style.display = "block";

  const sizeKB = Math.round((data.size || 0) / 1024);
  const largeHint = data.large
    ? `<span class="attach-large">⚠️ File lớn — sẽ dùng phân tích theo chunks</span>`
    : "";

  if (data.type === "image") {
    apEl.innerHTML = `
      <div class="attach-chip">
        <img src="data:${data.mime};base64,${data.b64}" alt="preview">
        <div class="attach-info">
          <div class="attach-name">${escHtml(data.name)}</div>
          <div class="attach-meta">${sizeKB}KB · Ảnh</div>
        </div>
        <button class="attach-rm" onclick="clearAttachPreview()">✕</button>
      </div>`;
  } else {
    const tok = data.tokens ? `~${data.tokens.toLocaleString()} tokens` : "";
    apEl.innerHTML = `
      <div class="attach-chip">
        <div class="attach-info">
          <div class="attach-name">📄 ${escHtml(data.name)}</div>
          <div class="attach-meta">${sizeKB}KB · ${(data.lines || 0).toLocaleString()} dòng · ${tok}</div>
          ${largeHint}
        </div>
        <button class="attach-rm" onclick="clearAttachPreview()">✕</button>
      </div>`;
  }

  if (data.tokens) {
    document.getElementById("token-count").textContent = `~${data.tokens.toLocaleString()} tok`;
  }
}

function clearAttachPreview() {
  ATTACHED_FILE = null;
  const apEl = document.getElementById("attach-preview");
  apEl.style.display = "none";
  apEl.innerHTML = "";
  document.getElementById("token-count").textContent = "";
}

/* Drag & drop */
function setupGlobalDrag() {
  const ov = document.getElementById("drag-overlay");
  let counter = 0;
  document.addEventListener("dragenter", e => { e.preventDefault(); counter++; if (counter === 1) ov.classList.add("visible"); });
  document.addEventListener("dragleave", () => { counter--; if (counter <= 0) { counter = 0; ov.classList.remove("visible"); } });
  document.addEventListener("dragover",  e => e.preventDefault());
  document.addEventListener("drop", e => {
    e.preventDefault(); counter = 0; ov.classList.remove("visible");
    const f = e.dataTransfer?.files?.[0];
    if (f) uploadFile(f);
  });
}

function onDragOver(e) { e.preventDefault(); document.getElementById("input-box").style.borderColor = "rgba(124,58,237,.55)"; }
function onDragLeave()  { document.getElementById("input-box").style.borderColor = ""; }
function onDrop(e) { e.preventDefault(); onDragLeave(); const f = e.dataTransfer?.files?.[0]; if (f) uploadFile(f); }

/* ──────────────────────────────────────────────────────────
   MODEL SELECTION
────────────────────────────────────────────────────────── */
function toggleModelDD() {
  document.getElementById("model-dd").classList.toggle("visible");
  document.getElementById("model-pill").classList.toggle("open");
}

function selectModel(name, event) {
  if (event) event.stopPropagation();
  setModelUI(name);
  document.getElementById("model-dd").classList.remove("visible");
  document.getElementById("model-pill").classList.remove("open");
}

function switchModel(name) {
  setModelUI(name);
}

function setModelUI(name) {
  CURRENT_MODEL = name;
  const color = MODEL_COLORS[name] || "#8b5cf6";

  document.getElementById("pill-name").textContent = name;
  const dot = document.getElementById("pill-dot");
  dot.style.background  = color;
  dot.style.boxShadow   = `0 0 7px ${color}`;

  document.querySelectorAll(".model-opt").forEach(el => {
    el.classList.toggle("selected", el.dataset.model === name);
  });
}

document.addEventListener("click", e => {
  if (!e.target.closest("#model-pill")) {
    document.getElementById("model-dd").classList.remove("visible");
    document.getElementById("model-pill").classList.remove("open");
  }
});

/* ──────────────────────────────────────────────────────────
   SETTINGS
────────────────────────────────────────────────────────── */
/* openSettings moved above */

function closeSettings() { document.getElementById("settings-modal").classList.remove("open"); }

/* toggleKeyVis moved above */

/* (duplicate removed) */

/* ──────────────────────────────────────────────────────────
   SHORTCUTS MODAL
────────────────────────────────────────────────────────── */
function openShortcuts()  { document.getElementById("shortcuts-modal").classList.add("open"); }
function closeShortcuts() { document.getElementById("shortcuts-modal").classList.remove("open"); }

/* ──────────────────────────────────────────────────────────
   EXPORT
────────────────────────────────────────────────────────── */
/* showExportMenu moved above */

/* (duplicate removed) */

/* ──────────────────────────────────────────────────────────
   THEME
────────────────────────────────────────────────────────── */
/* toggleTheme moved above */

/* ──────────────────────────────────────────────────────────
   SIDEBAR
────────────────────────────────────────────────────────── */
function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("collapsed");
}

/* ──────────────────────────────────────────────────────────
   MARKDOWN + CODE HIGHLIGHT
────────────────────────────────────────────────────────── */
function renderMarkdown(text) {
  marked.setOptions({ breaks: true, gfm: true });
  let html = marked.parse(text);

  // Gemini-style code blocks with language dot, line count, copy button
  html = html.replace(
    /<pre><code class="language-(\w+)">([\s\S]*?)<\/code><\/pre>/g,
    (_, lang, code) => {
      const lines = code.split('\n').length;
      const lineHint = lines > 3 ? `<span class="code-lines">${lines} dòng</span>` : '';
      const copyIcon = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
      return (
        `<pre>` +
        `<div class="code-header">` +
          `<span class="code-lang">${lang}</span>` +
          `${lineHint}` +
          `<button class="code-copy" onclick="copyCode(this)">${copyIcon} Sao chép</button>` +
        `</div>` +
        `<code class="language-${lang}">${code}</code>` +
        `</pre>`
      );
    }
  );

  // Generic code blocks (no language specified)
  html = html.replace(
    /<pre><code(?! class)([\s\S]*?)>([\s\S]*?)<\/code><\/pre>/g,
    (_, attrs, code) => {
      const copyIcon = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
      return (
        `<pre>` +
        `<div class="code-header">` +
          `<span class="code-lang">code</span>` +
          `<button class="code-copy" onclick="copyCode(this)">${copyIcon} Sao chép</button>` +
        `</div>` +
        `<code${attrs}>${code}</code>` +
        `</pre>`
      );
    }
  );

  return html;
}

function highlightCodeBlocks(container) {
  container.querySelectorAll("pre code").forEach(block => {
    try { hljs.highlightElement(block); } catch(e) { /* ignore */ }
  });
}

function copyCode(btn) {
  const code = btn.closest("pre")?.querySelector("code")?.innerText || "";
  navigator.clipboard.writeText(code).then(() => {
    const origHTML = btn.innerHTML;
    const checkIcon = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="20,6 9,17 4,12"/></svg>`;
    btn.innerHTML = checkIcon + " Đã sao chép";
    btn.classList.add("copied");
    setTimeout(() => {
      btn.innerHTML = origHTML;
      btn.classList.remove("copied");
    }, 2200);
  }).catch(() => {
    // Fallback for older browsers
    const ta = document.createElement("textarea");
    ta.value = code; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
    btn.innerHTML = "✓ Đã sao chép"; btn.classList.add("copied");
    setTimeout(() => { btn.classList.remove("copied"); }, 2000);
  });
}

function copyText(btn, b64) {
  try {
    const text = decodeB64(b64);
    navigator.clipboard.writeText(text).then(() => {
      const orig = btn.textContent;
      btn.textContent = "✅ Đã sao chép";
      setTimeout(() => btn.textContent = orig, 2000);
    });
  } catch(e) { showToast("❌ Không thể sao chép"); }
}

/* ──────────────────────────────────────────────────────────
   INPUT HELPERS
────────────────────────────────────────────────────────── */
/* onInputKey moved above */

function onInputChange(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 180) + "px";
  const chars = el.value.length;
  const tc = document.getElementById("token-count");
  if (chars > 0 && !ATTACHED_FILE) {
    tc.textContent = `~${Math.round(chars / 3.8)} tok`;
  } else if (!ATTACHED_FILE) {
    tc.textContent = "";
  }
}

function qi(text) {
  document.getElementById("user-input").value = text;
  document.getElementById("user-input").focus();
}

function switchAndAsk(model, text) {
  switchModel(model);
  qi(text);
}

function scrollToBottom() {
  const ca = document.getElementById("chat-area");
  ca.scrollTop = ca.scrollHeight;
}

/* ══════════════════════════════════════════════════════════
   🎨 THEME SYSTEM — FULL FIX
══════════════════════════════════════════════════════════ */
const THEMES = {
  dark:     { "--bg":"#0b0b14","--bg-s":"#0e0e1c","--bg-c":"#131325","--bg-i":"#18183a","--bg-h":"#1e1e45","--bg-a":"#252558","--b1":"rgba(88,101,242,.18)","--b2":"rgba(88,101,242,.30)","--b3":"rgba(88,101,242,.45)","--t1":"#f2f3ff","--t2":"#8891cc","--t3":"#40426a","--ac1":"#5865f2","--ac2":"#7289da" },
  midnight: { "--bg":"#05071a","--bg-s":"#090c26","--bg-c":"#0d1032","--bg-i":"#11153e","--bg-h":"#161b4a","--bg-a":"#1c2258","--b1":"rgba(100,120,255,.15)","--b2":"rgba(100,120,255,.28)","--b3":"rgba(100,120,255,.42)","--t1":"#dde4ff","--t2":"#7080c0","--t3":"#363880","--ac1":"#6366f1","--ac2":"#a78bfa" },
  ocean:    { "--bg":"#011012","--bg-s":"#031418","--bg-c":"#041c22","--bg-i":"#06242c","--bg-h":"#082e38","--bg-a":"#0a3842","--b1":"rgba(0,210,190,.12)","--b2":"rgba(0,210,190,.22)","--b3":"rgba(0,210,190,.35)","--t1":"#ccf5f0","--t2":"#4a9090","--t3":"#1e4848","--ac1":"#0d9488","--ac2":"#06b6d4" },
  forest:   { "--bg":"#020d04","--bg-s":"#051208","--bg-c":"#08190c","--bg-i":"#0b2010","--bg-h":"#0e2814","--bg-a":"#123018","--b1":"rgba(0,200,80,.10)","--b2":"rgba(0,200,80,.20)","--b3":"rgba(0,200,80,.32)","--t1":"#d4f5dc","--t2":"#4a9060","--t3":"#1e4830","--ac1":"#16a34a","--ac2":"#22c55e" },
  rose:     { "--bg":"#0f0508","--bg-s":"#180810","--bg-c":"#200c16","--bg-i":"#2a1020","--bg-h":"#34142a","--bg-a":"#3e1934","--b1":"rgba(255,80,120,.12)","--b2":"rgba(255,80,120,.22)","--b3":"rgba(255,80,120,.35)","--t1":"#ffe4ec","--t2":"#c06080","--t3":"#602040","--ac1":"#e11d48","--ac2":"#f43f5e" },
  light:    { "--bg":"#f0f0fa","--bg-s":"#e8e8f5","--bg-c":"#e0e0ef","--bg-i":"#d8d8ea","--bg-h":"#d0d0e4","--bg-a":"#c4c4dc","--b1":"rgba(0,0,0,.08)","--b2":"rgba(0,0,0,.14)","--b3":"rgba(0,0,0,.22)","--t1":"#0a0a1e","--t2":"#4a4a72","--t3":"#9898c0","--ac1":"#5865f2","--ac2":"#7289da" },
};

function applyTheme(name) {
  const vars = THEMES[name];
  if (!vars) return;
  const root = document.documentElement;
  root.setAttribute("data-theme", name);
  for (const [k, v] of Object.entries(vars)) {
    root.style.setProperty(k, v);
  }
  // Update theme-grid active state
  document.querySelectorAll(".theme-swatch").forEach(el => {
    el.classList.toggle("active", el.dataset.t === name);
  });
  const themeBtn = document.getElementById("theme-btn");
  if (themeBtn) themeBtn.innerHTML = name === "light"
    ? '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>'
    : '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
  localStorage.setItem("hnhat-theme", name);
  apiPost("/api/config", { theme: name }).catch(() => {});
}

function toggleTheme() {
  const cur = document.documentElement.getAttribute("data-theme") || "dark";
  applyTheme(cur === "dark" ? "light" : "dark");
}

/* ══════════════════════════════════════════════════════════
   🔤 FONT SIZE — FULL FIX
══════════════════════════════════════════════════════════ */
function setFontSize(sz) {
  const map = { sm: "13px", md: "15px", lg: "17px" };
  const px = map[sz] || "15px";
  // Apply to root so all rem/em scale
  document.documentElement.style.fontSize = px;
  // Mark active button
  document.querySelectorAll(".font-btn[id^='fs-']").forEach(el => el.classList.remove("active"));
  const btn = document.getElementById("fs-" + sz);
  if (btn) btn.classList.add("active");
  localStorage.setItem("hnhat-fontsize", sz);
  apiPost("/api/config", { font_size: sz }).catch(() => {});
}

/* ══════════════════════════════════════════════════════════
   🖼️ BACKGROUND IMAGE
══════════════════════════════════════════════════════════ */
let _bgDataURL = "";

function pickBackground() {
  document.getElementById("bg-file-input").click();
}

function onBgFileSelect(input) {
  const file = input.files[0];
  if (!file) return;
  if (file.size > 15 * 1024 * 1024) {
    showToast("❌ Ảnh quá lớn (tối đa 15MB)", "error"); return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    _bgDataURL = e.target.result;
    document.getElementById("bg-file-name").textContent = file.name;
    const prev = document.getElementById("bg-preview-img");
    prev.src = _bgDataURL;
    document.getElementById("bg-preview-wrap").style.display = "block";
    previewBg();
  };
  reader.readAsDataURL(file);
}

function previewBg() {
  const blur   = document.getElementById("bg-blur-slider")?.value || 0;
  const dim    = document.getElementById("bg-dim-slider")?.value || 0.5;
  document.getElementById("bg-blur-val").textContent = blur + "px";
  document.getElementById("bg-dim-val").textContent  = Math.round(dim * 100) + "%";
  if (_bgDataURL) applyBackground(_bgDataURL, parseFloat(blur), parseFloat(dim));
}

function applyBackground(dataURL, blur, dim) {
  const layer = document.getElementById("bg-layer");
  if (!dataURL) { layer.style.display = "none"; return; }
  layer.style.display = "block";
  layer.style.backgroundImage = "url(" + dataURL + ")";
  layer.style.filter = blur > 0 ? "blur(" + blur + "px)" : "none";
  layer.style.opacity = "1";
  // Dim overlay via pseudo — use inline child div
  let overlay = document.getElementById("bg-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "bg-overlay";
    overlay.style.cssText = "position:fixed;inset:0;z-index:0;pointer-events:none;transition:.3s";
    document.body.insertBefore(overlay, document.body.firstChild);
  }
  overlay.style.background = "rgba(0,0,0," + dim + ")";
  overlay.style.display = "block";
}

function removeBackground() {
  _bgDataURL = "";
  document.getElementById("bg-layer").style.display = "none";
  const ov = document.getElementById("bg-overlay");
  if (ov) ov.style.display = "none";
  document.getElementById("bg-file-name").textContent = "Chưa chọn";
  document.getElementById("bg-preview-wrap").style.display = "none";
  document.getElementById("bg-file-input").value = "";
  localStorage.removeItem("hnhat-bg");
  apiPost("/api/config", { bg_image: "" }).catch(() => {});
  showToast("🗑️ Đã xóa hình nền", "info");
}

function saveBgToStorage(dataURL, blur, dim) {
  try {
    localStorage.setItem("hnhat-bg", JSON.stringify({ dataURL, blur, dim }));
    apiPost("/api/config", { bg_blur: blur, bg_opacity: dim }).catch(() => {});
  } catch(e) { /* storage full */ }
}

/* ══════════════════════════════════════════════════════════
   💬 BUBBLE STYLE
══════════════════════════════════════════════════════════ */
function applyBubbleStyle(style) {
  const root = document.documentElement;
  const styles = {
    default: "linear-gradient(135deg, rgba(88,101,242,.28), rgba(163,125,255,.18))",
    flat:    "rgba(88,101,242,.16)",
    minimal: "transparent",
    glass:   "rgba(255,255,255,.06)",
  };
  const borders = {
    default: "rgba(88,101,242,.38)",
    flat:    "rgba(88,101,242,.25)",
    minimal: "rgba(88,101,242,.35)",
    glass:   "rgba(255,255,255,.14)",
  };
  document.querySelectorAll(".bubble-user").forEach(el => {
    el.style.background   = styles[style] || styles.default;
    el.style.borderColor  = borders[style] || borders.default;
    if (style === "glass") el.style.backdropFilter = "blur(10px)";
    if (style === "minimal") el.style.padding = "10px 14px";
  });
  localStorage.setItem("hnhat-bubble", style);
}

/* ══════════════════════════════════════════════════════════
   ⭐ FAVORITE CHATS
══════════════════════════════════════════════════════════ */
async function toggleFavorite(id, event) {
  event.stopPropagation();
  try {
    const res = await apiPost(`/api/chats/${id}/favorite`, {});
    await loadChatList();
    showToast(res.favorite ? "⭐ Đã thêm yêu thích" : "✅ Đã bỏ yêu thích");
  } catch(e) { showToast("❌ " + e.message, "error"); }
}

/* ══════════════════════════════════════════════════════════
   📊 STATISTICS
══════════════════════════════════════════════════════════ */
async function openStats() {
  try {
    const chats = await apiGet("/api/chats");
    const totalChats = chats.length;
    const totalMsgs  = chats.reduce((a, c) => a + (c.count || 0), 0);
    const favCount   = chats.filter(c => c.favorite).length;
    const estTok     = Math.round(totalMsgs * 180);

    document.getElementById("stat-total-chats").textContent = totalChats;
    document.getElementById("stat-total-msgs").textContent  = totalMsgs;
    document.getElementById("stat-fav").textContent         = favCount;
    document.getElementById("stat-tokens").textContent      = estTok > 1000 ? (estTok/1000).toFixed(1)+"K" : estTok;

    const modelCount = {};
    chats.forEach(c => { if (c.model) modelCount[c.model] = (modelCount[c.model] || 0) + 1; });
    const sorted = Object.entries(modelCount).sort((a,b) => b[1]-a[1]).slice(0, 5);
    const maxVal = sorted[0]?.[1] || 1;

    document.getElementById("stat-model-bars").innerHTML = sorted.map(([m, cnt]) => `
      <div class="stat-bar-wrap">
        <div class="stat-bar-label"><span>${escHtml(m)}</span><span>${cnt} chat</span></div>
        <div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${Math.round(cnt/maxVal*100)}%"></div></div>
      </div>`).join("");

    document.getElementById("stats-modal").classList.add("open");
  } catch(e) { showToast("❌ " + e.message, "error"); }
}

function closeStats() {
  document.getElementById("stats-modal").classList.remove("open");
}

/* ══════════════════════════════════════════════════════════
   📚 PROMPT LIBRARY
══════════════════════════════════════════════════════════ */
const BUILT_IN_PROMPTS = [
  { id:"b1",  cat:"code",   icon:"🔍", name:"Review code",       text:"Review code này chi tiết: tìm bug, security issues, performance, đề xuất cải tiến theo best practices." },
  { id:"b2",  cat:"code",   icon:"📖", name:"Giải thích code",   text:"Giải thích từng dòng code này một cách rõ ràng, nêu logic và mục đích của từng phần." },
  { id:"b3",  cat:"code",   icon:"🐛", name:"Debug lỗi",         text:"Tôi gặp lỗi sau. Hãy phân tích nguyên nhân và đưa ra cách fix chi tiết:\n\n```\n[dán lỗi vào đây]\n```" },
  { id:"b4",  cat:"code",   icon:"⚡", name:"Tối ưu code",       text:"Tối ưu đoạn code này về hiệu suất và độ đọc được, giải thích từng thay đổi bạn thực hiện." },
  { id:"b5",  cat:"code",   icon:"🧪", name:"Viết unit test",    text:"Viết unit test đầy đủ cho code này dùng pytest, bao gồm edge cases và mock API khi cần." },
  { id:"b6",  cat:"code",   icon:"🔄", name:"Refactor SOLID",    text:"Refactor code này theo nguyên tắc SOLID và clean code. Giải thích từng thay đổi, giữ nguyên chức năng." },
  { id:"b7",  cat:"web",    icon:"🌐", name:"REST API Flask",    text:"Viết REST API Python Flask với CRUD endpoint đầy đủ, JWT authentication, validation, error handling, swagger docs. Chủ đề: " },
  { id:"b8",  cat:"web",    icon:"⚛️", name:"React Component",  text:"Tạo React component với TypeScript và Tailwind CSS. Mô tả component: " },
  { id:"b9",  cat:"web",    icon:"🎨", name:"Landing page",      text:"Tạo landing page HTML/CSS/JS glassmorphism, responsive, animation đẹp cho: " },
  { id:"b10", cat:"web",    icon:"📱", name:"Responsive CSS",    text:"Tạo CSS responsive đẹp cho layout sau, hỗ trợ mobile/tablet/desktop với breakpoints chuẩn: " },
  { id:"b11", cat:"data",   icon:"🗄️", name:"SQL Optimize",      text:"Viết SQL query tối ưu cho bài toán, thêm index phù hợp, giải thích execution plan: " },
  { id:"b12", cat:"data",   icon:"🐍", name:"Pandas xử lý",     text:"Viết code Python Pandas để xử lý DataFrame: đọc CSV, làm sạch data, thống kê, xuất kết quả." },
  { id:"b13", cat:"data",   icon:"📊", name:"Visualize data",   text:"Viết code Python matplotlib/seaborn để visualize dataset với biểu đồ phù hợp nhất: " },
  { id:"b14", cat:"devops", icon:"🐳", name:"Dockerfile",        text:"Viết Dockerfile + docker-compose.yml tối ưu, multi-stage build cho app: " },
  { id:"b15", cat:"devops", icon:"🔄", name:"CI/CD Pipeline",   text:"Viết GitHub Actions workflow hoàn chỉnh: test → build Docker → deploy lên VPS cho app Python/Node." },
  { id:"b16", cat:"devops", icon:"🚀", name:"Deploy VPS",        text:"Hướng dẫn deploy app lên Ubuntu VPS: Nginx reverse proxy + SSL Let\'s Encrypt + systemd service. App: " },
  { id:"b17", cat:"write",  icon:"✉️", name:"Email chuyên nghiệp", text:"Viết email chuyên nghiệp tiếng Việt, trang trọng và lịch sự về chủ đề: " },
  { id:"b18", cat:"write",  icon:"📋", name:"Tóm tắt",          text:"Tóm tắt nội dung sau thành 5 điểm chính quan trọng nhất, rõ ràng và súc tích:\n\n" },
  { id:"b19", cat:"write",  icon:"📝", name:"Báo cáo kỹ thuật", text:"Viết báo cáo kỹ thuật chuyên nghiệp, đầy đủ sections (tổng quan, phân tích, giải pháp, kết luận) cho: " },
  { id:"b20", cat:"write",  icon:"💡", name:"Brainstorm",        text:"Brainstorm 10 ý tưởng sáng tạo và khả thi nhất cho: " },
];

let currentCat = "all";
let customPrompts = [];

async function openPromptLibrary() {
  customPrompts = await apiGet("/api/prompts").catch(() => []);
  renderPrompts("all", "");
  currentCat = "all";
  document.querySelectorAll(".pm-cat").forEach(el => el.classList.toggle("active", el.dataset.cat === "all"));
  document.getElementById("prompt-modal").classList.add("open");
}

function closePromptLibrary() {
  document.getElementById("prompt-modal").classList.remove("open");
}

function setCat(cat, el) {
  currentCat = cat;
  document.querySelectorAll(".pm-cat").forEach(e => e.classList.remove("active"));
  el.classList.add("active");
  renderPrompts(cat, document.getElementById("pm-search-input")?.value || "");
}

function filterPrompts(q) { renderPrompts(currentCat, q); }

function renderPrompts(cat, q) {
  const all = [
    ...BUILT_IN_PROMPTS,
    ...customPrompts.map(p => ({ ...p, cat: "custom" }))
  ];
  let list = cat === "all" ? all : all.filter(p => p.cat === cat);
  if (q) list = list.filter(p =>
    p.name.toLowerCase().includes(q.toLowerCase()) ||
    p.text.toLowerCase().includes(q.toLowerCase())
  );
  const el = document.getElementById("pm-list");
  if (!list.length) {
    el.innerHTML = '<div style="color:var(--t3);font-size:.82rem;padding:12px;grid-column:1/-1;text-align:center">Không có prompt nào</div>';
    return;
  }
  el.innerHTML = list.map(p => `
    <div class="pm-item" onclick="usePrompt('${encodeB64(p.text)}')">
      <div class="pm-item-top">
        <span class="pm-item-ico">${p.icon || "📝"}</span>
        <span class="pm-item-name">${escHtml(p.name)}</span>
      </div>
      <div class="pm-item-text">${escHtml(p.text)}</div>
      ${p.cat === "custom" ? `<button class="pm-item-del" onclick="deletePrompt('${escAttr(p.id)}',event)">✕</button>` : ""}
    </div>`).join("");
}

function usePrompt(b64) {
  const text = decodeB64(b64);
  const inp = document.getElementById("user-input");
  inp.value = text;
  onInputChange(inp);
  closePromptLibrary();
  inp.focus();
}

async function saveCustomPrompt() {
  const name = document.getElementById("pm-new-name").value.trim();
  const text = document.getElementById("pm-new-text").value.trim();
  if (!name || !text) { showToast("Nhập tên và nội dung prompt", "error"); return; }
  await apiPost("/api/prompts", { name, text, icon: "📝" });
  document.getElementById("pm-new-name").value = "";
  document.getElementById("pm-new-text").value = "";
  customPrompts = await apiGet("/api/prompts").catch(() => []);
  renderPrompts(currentCat, "");
  showToast("✅ Đã lưu prompt!", "success");
}

async function deletePrompt(id, event) {
  event.stopPropagation();
  await apiCall(`/api/prompts/${id}`, "DELETE");
  customPrompts = await apiGet("/api/prompts").catch(() => []);
  renderPrompts(currentCat, "");
  showToast("🗑️ Đã xóa prompt");
}

/* ══════════════════════════════════════════════════════════
   🔍 IN-CHAT SEARCH
══════════════════════════════════════════════════════════ */
let searchMatches = [];
let searchIdx = 0;

function openSearchBar() {
  const bar = document.getElementById("search-bar");
  bar.classList.add("visible");
  const inp = document.getElementById("sb-input");
  inp.value = "";
  inp.focus();
  document.getElementById("sb-count").textContent = "";
}

function closeSearchBar() {
  document.getElementById("search-bar").classList.remove("visible");
  _clearSearchHL();
  document.getElementById("sb-count").textContent = "";
  document.getElementById("sb-input").value = "";
  _searchResults = []; _searchCur = -1;
}

function doSearch(query) {
  _clearSearchHL();
  _searchResults = []; _searchCur = -1;
  const cnt = document.getElementById("sb-count");
  const q = query.trim();
  if (!q) { cnt.textContent = ""; return; }

  const msgEl = document.getElementById("messages");
  if (!msgEl || msgEl.style.display === "none") {
    cnt.textContent = "Không có chat"; return;
  }

  // Find all .msg-content elements
  const contentEls = msgEl.querySelectorAll(".msg-content, .bubble-user .msg-content");
  const qLow = q.toLowerCase();

  contentEls.forEach(el => {
    const text = el.innerText || el.textContent;
    let idx = text.toLowerCase().indexOf(qLow);
    if (idx === -1) return;
    _searchResults.push(el);
  });

  if (!_searchResults.length) {
    cnt.textContent = "Không tìm thấy";
    return;
  }

  _searchCur = 0;
  _applySearchHL(q);
  _jumpTo(_searchCur);
  cnt.textContent = `1 / ${_searchResults.length}`;
}

function _applySearchHL(q) {
  const qLow = q.toLowerCase();
  _searchResults.forEach((el, i) => {
    const text = el.innerHTML;
    const re = new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
    el.innerHTML = el.innerHTML.replace(re, m =>
      `<mark class="highlight-match${i === _searchCur ? ' current' : ''}" data-si="${i}">${m}</mark>`
    );
  });
}

function _clearSearchHL() {
  document.querySelectorAll("mark.highlight-match").forEach(m => {
    const parent = m.parentNode;
    if (parent) {
      parent.replaceChild(document.createTextNode(m.textContent), m);
      parent.normalize();
    }
  });
}

function _jumpTo(i) {
  if (i < 0 || i >= _searchResults.length) return;
  // Update active highlight
  document.querySelectorAll("mark.highlight-match").forEach(m => m.classList.remove("current"));
  document.querySelectorAll(`mark.highlight-match[data-si="${i}"]`).forEach(m => m.classList.add("current"));
  _searchResults[i].scrollIntoView({ behavior: "smooth", block: "center" });
  document.getElementById("sb-count").textContent = `${i+1} / ${_searchResults.length}`;
}

function searchNav(dir) {
  if (!_searchResults.length) return;
  _searchCur = (_searchCur + dir + _searchResults.length) % _searchResults.length;
  _jumpTo(_searchCur);
}

function searchKey(e) {
  if (e.key === "Enter")  { e.preventDefault(); searchNav(e.shiftKey ? -1 : 1); }
  if (e.key === "Escape") closeSearchBar();
}

/* ══════════════════════════════════════════════════════════
   🎙️ VOICE INPUT
══════════════════════════════════════════════════════════ */
let voiceRecognition = null;
let isListening = false;

function toggleVoiceInput() {
  const btn = document.getElementById("voice-btn");
  const SR  = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { showToast("❌ Dùng Chrome để dùng Voice Input", "error"); return; }
  if (isListening) {
    voiceRecognition?.stop(); return;
  }
  voiceRecognition = new SR();
  voiceRecognition.lang = "vi-VN";
  voiceRecognition.continuous = false;
  voiceRecognition.interimResults = true;
  voiceRecognition.onstart = () => {
    isListening = true;
    if (btn) { btn.classList.add("listening"); }
    showToast("🎙️ Đang nghe…", "info");
  };
  voiceRecognition.onresult = (e) => {
    const inp = document.getElementById("user-input");
    if (inp) {
      inp.value = Array.from(e.results).map(r => r[0].transcript).join("");
      onInputChange(inp);
    }
  };
  voiceRecognition.onend = () => {
    isListening = false;
    if (btn) btn.classList.remove("listening");
  };
  voiceRecognition.onerror = (e) => {
    isListening = false;
    if (btn) btn.classList.remove("listening");
    showToast("❌ Mic lỗi: " + e.error, "error");
  };
  voiceRecognition.start();
}

/* ══════════════════════════════════════════════════════════
   🔊 TEXT-TO-SPEECH
══════════════════════════════════════════════════════════ */
let isSpeaking = false;

function toggleTTS(btn, b64) {
  if (isSpeaking) {
    speechSynthesis.cancel(); isSpeaking = false;
    if (btn) { btn.classList.remove("tts-speaking"); btn.textContent = "🔊 Đọc"; }
    return;
  }
  try {
    const text = decodeB64(b64)
      .replace(/```[\s\S]*?```/g, "(code block)")
      .replace(/[#*_`>]/g, "")
      .trim();
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "vi-VN"; utter.rate = 1.0;
    const voices = speechSynthesis.getVoices();
    const vi = voices.find(v => v.lang.startsWith("vi"));
    if (vi) utter.voice = vi;
    utter.onstart = () => {
      isSpeaking = true;
      if (btn) { btn.classList.add("tts-speaking"); btn.textContent = "⏹️ Dừng"; }
    };
    utter.onend = utter.onerror = () => {
      isSpeaking = false;
      if (btn) { btn.classList.remove("tts-speaking"); btn.textContent = "🔊 Đọc"; }
    };
    speechSynthesis.speak(utter);
  } catch(e) { showToast("❌ TTS lỗi: " + e.message, "error"); }
}

/* ══════════════════════════════════════════════════════════
   🔀 RETRY WITH DIFFERENT MODEL
══════════════════════════════════════════════════════════ */
const ALL_MODELS_LIST = ["Hnhat Fast","Hnhat Pro","Hnhat Master","Hnhat Code","Hnhat Reason","Kimi K2","Gemma 2","Compound","Hnhat Vision","Hnhat Vision+"];

function getModelIcon(name) {
  const icons = { "Hnhat Fast":"⚡","Hnhat Pro":"🔥","Hnhat Master":"👑","Hnhat Code":"💻",
    "Hnhat Reason":"🧠","Kimi K2":"🌀","Gemma 2":"💎","Compound":"⚗️","Hnhat Vision":"👁","Hnhat Vision+":"🔭" };
  return icons[name] || "🤖";
}

function showRetryPicker(btn, b64) {
  const picker = document.getElementById("retry-picker");
  if (!picker) return;
  const rect = btn.getBoundingClientRect();
  picker.style.top    = (rect.bottom + window.scrollY + 6) + "px";
  picker.style.left   = Math.min(rect.left, window.innerWidth - 220) + "px";
  picker.style.display = "block";
  const list = picker.querySelector("#retry-picker-list");
  if (list) list.innerHTML = ALL_MODELS_LIST
    .filter(m => m !== CURRENT_MODEL)
    .map(m => `<div class="model-opt" style="padding:7px 10px" onclick="retryWithModel('${escAttr(m)}',event)">
      <span style="font-size:14px;width:22px">${getModelIcon(m)}</span>
      <span style="font:.8rem var(--fui);color:var(--t1)">${escHtml(m)}</span>
    </div>`).join("");
  setTimeout(() => document.addEventListener("click", closePicker), 10);
}

function closePicker() {
  const p = document.getElementById("retry-picker");
  if (p) p.style.display = "none";
  document.removeEventListener("click", closePicker);
}

async function retryWithModel(modelName, event) {
  event.stopPropagation();
  closePicker();
  setModelUI(modelName);
  await regenerateLastMessage();
  showToast("🔀 Thử lại với " + modelName);
}

/* ══════════════════════════════════════════════════════════
   📋 PASTE IMAGE FROM CLIPBOARD
══════════════════════════════════════════════════════════ */
document.addEventListener("paste", async (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      e.preventDefault();
      const file = item.getAsFile();
      if (file) { await uploadFile(file); showToast("📋 Đã dán ảnh từ clipboard!", "success"); }
      break;
    }
  }
});

/* ══════════════════════════════════════════════════════════
   📄 COPY AS MARKDOWN
══════════════════════════════════════════════════════════ */
function copyAsMarkdown(btn, b64) {
  const text = decodeB64(b64);
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.textContent;
    btn.textContent = "✅ Copied!";
    setTimeout(() => btn.textContent = orig, 2000);
  });
}

/* ══════════════════════════════════════════════════════════
   🌡️ TEMPERATURE
══════════════════════════════════════════════════════════ */
function getTemperature() {
  const el = document.getElementById("temp-slider");
  return el ? parseFloat(el.value) : 0.7;
}

/* ══════════════════════════════════════════════════════════
   💾 RESTORE LOCAL PREFERENCES
══════════════════════════════════════════════════════════ */
function restoreLocalPrefs() {
  // Font size
  const fs = localStorage.getItem("hnhat-fontsize") || "md";
  setFontSize(fs);
  // Theme
  const th = localStorage.getItem("hnhat-theme") || "dark";
  applyTheme(th);
  // Background
  try {
    const bgStr = localStorage.getItem("hnhat-bg");
    if (bgStr) {
      const bg = JSON.parse(bgStr);
      _bgDataURL = bg.dataURL || "";
      if (_bgDataURL) {
        applyBackground(_bgDataURL, bg.blur || 0, bg.dim || 0.5);
        const prev = document.getElementById("bg-preview-img");
        if (prev) { prev.src = _bgDataURL; document.getElementById("bg-preview-wrap").style.display = "block"; }
      }
    }
  } catch(e) {}
  // Bubble style
  const bs = localStorage.getItem("hnhat-bubble");
  if (bs) setTimeout(() => applyBubbleStyle(bs), 300);
  // Send-on-enter pref
  const se = localStorage.getItem("hnhat-send-enter");
  if (se !== null) {
    const el = document.getElementById("send-enter-toggle");
    if (el) el.checked = se === "true";
  }
}

/* ══════════════════════════════════════════════════════════
   ⚙️ OPEN SETTINGS — RESTORE VALUES
══════════════════════════════════════════════════════════ */
async function openSettings() {
  try {
    const cfg = await apiGet("/api/config");
    document.getElementById("default-model-input").value    = cfg.default_model || "Hnhat Pro";
    const gks = document.getElementById("groq-key-status");
    const gems = document.getElementById("gemini-key-status");
    if (gks)  gks.innerHTML  = cfg.has_groq   ? '<span class="status-dot ok"></span>Đã cài'   : '<span class="status-dot no"></span>Chưa cài';
    if (gems) gems.innerHTML = cfg.has_gemini ? '<span class="status-dot ok"></span>Đã cài'   : '<span class="status-dot no"></span>Chưa cài';

    // Restore temp slider
    const ts = document.getElementById("temp-slider");
    if (ts) ts.value = 0.7;

    // Restore theme swatches
    const curTheme = localStorage.getItem("hnhat-theme") || "dark";
    document.querySelectorAll(".theme-swatch").forEach(el => {
      el.classList.toggle("active", el.dataset.t === curTheme);
    });

    // Restore font-size buttons
    const curFs = localStorage.getItem("hnhat-fontsize") || "md";
    document.querySelectorAll(".font-btn[id^='fs-']").forEach(el => el.classList.remove("active"));
    const fsBtn = document.getElementById("fs-" + curFs);
    if (fsBtn) fsBtn.classList.add("active");

    // Restore bubble style
    const bs = document.getElementById("bubble-style-input");
    if (bs) bs.value = localStorage.getItem("hnhat-bubble") || "default";

    // Restore send-on-enter
    const se = document.getElementById("send-enter-toggle");
    if (se) se.checked = localStorage.getItem("hnhat-send-enter") !== "false";

        // Sync bg sliders
    syncBgSliders();

  } catch(e) { console.warn("Settings load:", e); }
  document.getElementById("settings-modal").classList.add("open");
  syncBgSliders();
}

function closeSettings() {
  document.getElementById("settings-modal").classList.remove("open");
}

function toggleKeyVis(inputId) {
  const el = document.getElementById(inputId);
  if (el) el.type = el.type === "password" ? "text" : "password";
}

async function saveSettings() {
  const groqKey  = (document.getElementById("groq-key-input")?.value   || "").trim();
  const gemKey   = (document.getElementById("gemini-key-input")?.value  || "").trim();
  const defModel = document.getElementById("default-model-input")?.value || "Hnhat Pro";
  const sysPr    = (document.getElementById("system-prompt-input")?.value || "").trim();
  const temp     = getTemperature();
  const bubble   = document.getElementById("bubble-style-input")?.value || "default";
  const sendEnter= document.getElementById("send-enter-toggle")?.checked !== false;

  const body = { default_model: defModel, temperature: temp };
  if (groqKey) body.groq_key       = groqKey;
  if (gemKey)  body.gemini_key     = gemKey;
  if (sysPr)   body.system_prompt  = sysPr;

  // Apply bg
  if (_bgDataURL) {
    const blur = parseFloat(document.getElementById("bg-blur-slider")?.value || 0);
    const dim  = parseFloat(document.getElementById("bg-dim-slider")?.value  || 0.5);
    applyBackground(_bgDataURL, blur, dim);
    saveBgToStorage(_bgDataURL, blur, dim);
  }

  // Apply bubble
  applyBubbleStyle(bubble);

  // Save send-enter
  localStorage.setItem("hnhat-send-enter", sendEnter);

  try {
    await apiPost("/api/config", body);
    CURRENT_MODEL = defModel;
    setModelUI(defModel);
    closeSettings();
    showToast("✅ Đã lưu cài đặt!", "success");
  } catch(e) {
    showToast("❌ Lưu thất bại: " + e.message, "error");
  }
}

/* ══════════════════════════════════════════════════════════
   ⌨️ SEND-ON-ENTER HANDLING
══════════════════════════════════════════════════════════ */
function onInputKey(e) {
  const sendEnter = localStorage.getItem("hnhat-send-enter") !== "false";
  if (e.key === "Enter") {
    if (sendEnter && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    else if (!sendEnter && (e.ctrlKey || e.metaKey)) { e.preventDefault(); sendMessage(); }
  }
}

/* ═══════════════════════════════════════════════════════════
   📥 EXPORT
═══════════════════════════════════════════════════════════ */
async function showExportMenu() {
  if (!CURRENT_CHAT_ID) { showToast("Chưa có chat để export", "error"); return; }
  const format = prompt("Xuất định dạng:\n  md   = Markdown\n  txt  = Văn bản thuần\n  json = JSON\n\nNhập:", "md");
  if (!format) return;
  await exportChat(format.trim().toLowerCase());
}

async function exportChat(format) {
  try {
    const data = await apiGet(`/api/chats/${CURRENT_CHAT_ID}`);
    const msgs = data.messages || [];
    let content, ext, mime;
    if (format === "json") {
      content = JSON.stringify(data, null, 2); ext = "json"; mime = "application/json";
    } else if (format === "txt") {
      content = `${data.title}\nModel: ${data.model}\nNgày: ${new Date().toLocaleString("vi")}\n${"─".repeat(50)}\n\n`;
      msgs.forEach(m => { content += (m.role === "user" ? "Bạn: " : "Hnhat AI: ") + m.content + "\n\n"; });
      ext = "txt"; mime = "text/plain";
    } else {
      content = `# ${data.title}\n**Model:** ${data.model} | **Ngày:** ${new Date().toLocaleString("vi")}\n\n---\n\n`;
      msgs.forEach(m => { content += m.role === "user" ? `**🧑 Bạn:**\n${m.content}\n\n` : `**⚡ Hnhat AI:**\n${m.content}\n\n---\n\n`; });
      ext = "md"; mime = "text/markdown";
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([content], { type: mime }));
    a.download = `hnhat-${Date.now()}.${ext}`;
    a.click(); URL.revokeObjectURL(a.href);
    showToast(`📥 Đã export ${format.toUpperCase()}!`, "success");
  } catch(e) { showToast("❌ Export lỗi: " + e.message, "error"); }
}

/* ──────────────────────────────────────────────────────────
   KEYBOARD SHORTCUTS
────────────────────────────────────────────────────────── */
document.addEventListener("keydown", e => {
  const ctrl = e.ctrlKey || e.metaKey;
  if (ctrl && e.key === "k")  { e.preventDefault(); newChat(); }
  if (ctrl && e.key === "f")  { e.preventDefault(); openSearchBar(); }
  if (ctrl && e.key === "e")  { e.preventDefault(); showExportMenu(); }
  if (ctrl && e.key === "\\") { e.preventDefault(); toggleSidebar(); }
  if (ctrl && e.key === ",")  { e.preventDefault(); openSettings(); }
  if (ctrl && e.key === "u")  { e.preventDefault(); document.getElementById("file-input").click(); }
  if (ctrl && e.key === "p")  { e.preventDefault(); openPromptLibrary(); }
  if (ctrl && e.key === "m")  { e.preventDefault(); toggleVoiceInput(); }
  if (e.key === "Escape") {
    closeSettings();
    closeShortcuts();
    document.getElementById("model-dd").classList.remove("visible");
    document.getElementById("model-pill").classList.remove("open");
  }
});

/* ──────────────────────────────────────────────────────────
   UTILS
────────────────────────────────────────────────────────── */
async function apiGet(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

async function apiPost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

async function apiCall(url, method) {
  const r = await fetch(url, { method });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

async function startSSE(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.body.getReader();
}

function escHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>");
}

function escAttr(s) {
  return String(s || "").replace(/['"]/g, "");
}

function encodeB64(text) {
  try { return btoa(unescape(encodeURIComponent(text))); }
  catch(e) { return btoa(text.substring(0, 10000)); }
}

function decodeB64(b64) {
  return decodeURIComponent(escape(atob(b64)));
}

function nowTime() {
  return new Date().toLocaleTimeString("vi", { hour: "2-digit", minute: "2-digit" });
}

function relTime(iso) {
  if (!iso) return "";
  const diff = (Date.now() - new Date(iso)) / 1000;
  if (diff < 60)    return "Vừa xong";
  if (diff < 3600)  return Math.floor(diff / 60) + "p trước";
  if (diff < 86400) return Math.floor(diff / 3600) + "h trước";
  return new Date(iso).toLocaleDateString("vi");
}

function showToast(msg, type) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = "show" + (type ? " " + type : "");
  clearTimeout(TOAST_TIMER);
  TOAST_TIMER = setTimeout(() => el.classList.remove("show"), 3000);
}

/* ──────────────────────────────────────────────────────────
   BOOT
────────────────────────────────────────────────────────── */
boot();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def _open_app_window():
        import time
        import subprocess
        import webbrowser
        time.sleep(1.5) # Đợi server khởi động
        
        url = "http://127.0.0.1:5000"
        try:
            # Ưu tiên 1: Mở bằng Microsoft Edge (có sẵn trên mọi Windows) ở chế độ Ứng dụng độc lập
            subprocess.Popen(['msedge', f'--app={url}'])
        except Exception:
            try:
                # Ưu tiên 2: Thử mở bằng Google Chrome ở chế độ Ứng dụng
                subprocess.Popen(['chrome', f'--app={url}'])
            except Exception:
                # Dự phòng: Trở về mở bằng tab trình duyệt bình thường
                webbrowser.open(url)

    print("Đang khởi động Hnhat AI Desktop...")
    # Chạy lệnh mở cửa sổ ở một luồng phụ
    threading.Thread(target=_open_app_window, daemon=True).start()
    
    # Khởi động server
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)