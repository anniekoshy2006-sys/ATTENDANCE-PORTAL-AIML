"""
Face-Recognition Attendance Portal  -  Flask server.

    python app.py                 # HTTPS on port 5000, reachable from the whole Wi-Fi
    python app.py --http          # plain HTTP (use behind ngrok / cloudflared tunnels)
    python app.py --port 8443

Optional: set PORTAL_PIN=1234 to require a PIN before the portal can be used.
"""
import argparse
import csv
import io
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, make_response, request, send_from_directory

import certs
from face_engine import FaceEngine, HERE

DATA_DIR = os.path.join(HERE, "data")
DB_PATH = os.path.join(DATA_DIR, "attendance.db")
os.makedirs(DATA_DIR, exist_ok=True)

# ---- attendance rules ------------------------------------------------------
MARK_FRAMES = 3          # same student must be recognised in this many consecutive frames...
MARK_WINDOW = 2.5        # ...with no gap longer than this many seconds
MARK_MIN_CONF = 70.0     # ...and every one of those frames must reach this confidence (%)
PIN = os.environ.get("PORTAL_PIN", "").strip()

app = Flask(__name__, static_folder=None)
engine = FaceEngine()
_db_lock = threading.Lock()
_streaks = {}            # name -> {"n": consecutive hits, "t": last hit time}
_streak_lock = threading.Lock()


# ------------------------------------------------------------------ database
def db():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS attendance(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student TEXT NOT NULL,
            day TEXT NOT NULL,
            time TEXT NOT NULL,
            confidence REAL,
            UNIQUE(student, day))""")


def mark_present(name, conf):
    """Insert today's record once. Returns (is_new, time_string)."""
    now = datetime.now()
    day, tm = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    with _db_lock, db() as con:
        cur = con.execute("INSERT OR IGNORE INTO attendance(student, day, time, confidence) "
                          "VALUES (?,?,?,?)", (name, day, tm, round(conf, 1)))
        if cur.rowcount:
            return True, tm
        row = con.execute("SELECT time FROM attendance WHERE student=? AND day=?",
                          (name, day)).fetchone()
        return False, row["time"]


def register_hit(name, conf):
    """Debounce: only mark attendance after several consecutive confident frames."""
    now = time.time()
    with _streak_lock:
        s = _streaks.get(name)
        if s and now - s["t"] <= MARK_WINDOW and conf >= MARK_MIN_CONF:
            s["n"] += 1
            s["t"] = now
        else:
            _streaks[name] = {"n": 1 if conf >= MARK_MIN_CONF else 0, "t": now}
            s = _streaks[name]
        return s["n"] >= MARK_FRAMES


# ---------------------------------------------------------------------- auth
@app.before_request
def _guard():
    if not request.path.startswith("/api/") or request.path == "/api/login":
        return None
    if PIN and request.cookies.get("portal_pin") != PIN:
        return jsonify(error="pin_required"), 401
    return None


@app.after_request
def _nocache(resp):
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.post("/api/login")
def login():
    given = str((request.get_json(silent=True) or {}).get("pin", "")).strip()
    if PIN and given != PIN:
        return jsonify(error="wrong_pin"), 403
    resp = make_response(jsonify(ok=True))
    resp.set_cookie("portal_pin", PIN, max_age=60 * 60 * 24 * 30, samesite="Lax", httponly=True)
    return resp


# ----------------------------------------------------------------------- pages
@app.get("/")
def index():
    return send_from_directory(os.path.join(HERE, "templates"), "index.html")


# ------------------------------------------------------------------------- API
@app.get("/api/status")
def status():
    rep = {}
    try:
        import json
        from face_engine import REPORT_FILE
        with open(REPORT_FILE) as f:
            rep = json.load(f)
    except Exception:
        pass
    val = rep.get("validation") or {}
    return jsonify(backend=engine.backend, students=engine.students,
                   pin_required=bool(PIN), validation_accuracy=val.get("accuracy"),
                   server_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@app.post("/api/recognize")
def recognize():
    data = request.get_data(cache=False)
    if not data or len(data) > 6_000_000:
        return jsonify(error="bad image"), 400
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="could not decode image"), 400
    h, w = img.shape[:2]
    faces = engine.identify(img)
    seen = set()
    for f in faces:
        f["status"] = None
        if not f["known"]:
            continue
        name = f["name"]
        # if two boxes claim the same person in one frame, only the stronger one counts
        if name in seen:
            continue
        seen.add(name)
        if register_hit(name, f["confidence"]):
            is_new, tm = mark_present(name, f["confidence"])
            f["status"] = "marked" if is_new else "already"
            f["time"] = tm
    return jsonify(width=w, height=h, faces=faces)


def _roster_for(day):
    with db() as con:
        rows = con.execute("SELECT student, time, confidence FROM attendance WHERE day=? "
                           "ORDER BY time", (day,)).fetchall()
    present = {r["student"]: r for r in rows}
    roster = []
    for name in engine.students:
        r = present.get(name)
        roster.append({"name": name, "present": bool(r),
                       "time": r["time"] if r else None,
                       "confidence": r["confidence"] if r else None})
    # students who have a record but are no longer in the trained set
    for name, r in present.items():
        if name not in engine.students:
            roster.append({"name": name, "present": True, "time": r["time"],
                           "confidence": r["confidence"]})
    return roster


@app.get("/api/attendance")
def attendance():
    day = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    try:
        datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        return jsonify(error="bad date"), 400
    roster = _roster_for(day)
    with db() as con:
        days = [r["day"] for r in con.execute(
            "SELECT DISTINCT day FROM attendance ORDER BY day DESC LIMIT 60")]
    n_present = sum(1 for r in roster if r["present"])
    return jsonify(date=day, today=datetime.now().strftime("%Y-%m-%d"), roster=roster,
                   present=n_present, total=len(roster), absent=len(roster) - n_present,
                   days_with_records=days)


@app.get("/api/export.csv")
def export_csv():
    day = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["Date", "Student", "Status", "Time", "Confidence %"])
    if day == "all":
        with db() as con:
            rows = con.execute("SELECT day, student, time, confidence FROM attendance "
                               "ORDER BY day DESC, time").fetchall()
        for r in rows:
            wr.writerow([r["day"], r["student"], "Present", r["time"], r["confidence"]])
    else:
        for r in _roster_for(day):
            wr.writerow([day, r["name"], "Present" if r["present"] else "Absent",
                         r["time"] or "", r["confidence"] if r["confidence"] is not None else ""])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=attendance_{day}.csv"})


# --------------------------------------------------------------------- startup
def prepare_model():
    """Load the model; train it on first run (or if the saved model is missing)."""
    from face_engine import MODELS_DIR
    train_dir = os.path.join(HERE, "dataset", "train")
    val_dir = os.path.join(HERE, "dataset", "validation")
    if engine.load():
        return
    print("No trained model found - training now from ./dataset/train ...")
    try:
        engine.train(train_dir, val_dir)
    except Exception as e:
        # neural back-end failed at run time -> degrade to the offline back-end
        print(f"  neural back-end failed ({e}); retrying with offline LBPH")
        fallback = FaceEngine(prefer="lbph")
        fallback.train(train_dir, val_dir)
        engine.__dict__.update(fallback.__dict__)


def selftest():
    """Make sure the loaded back-end really runs, otherwise fall back to LBPH."""
    try:
        engine.identify(np.zeros((240, 320, 3), np.uint8))
    except Exception as e:
        print(f"  {engine.backend} back-end crashed on a test frame ({e}); using LBPH instead")
        fallback = FaceEngine(prefer="lbph")
        if not fallback.load():
            fallback.train(os.path.join(HERE, "dataset", "train"),
                           os.path.join(HERE, "dataset", "validation"))
        engine.__dict__.update(fallback.__dict__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--http", action="store_true", help="serve plain HTTP (for use behind a tunnel)")
    a = ap.parse_args()

    print("\n=== Face Attendance Portal ===")
    init_db()
    prepare_model()
    selftest()
    print(f"Recognition back-end : {engine.backend}")
    print(f"Students             : {', '.join(engine.students)}")
    if engine.backend == "lbph":
        print("\n  !! Running in BASIC OFFLINE mode - accuracy is limited.")
        print("  !! For good accuracy run:  python download_models.py   then   python train.py\n")

    ips = certs.lan_ips()
    ssl_ctx, scheme = None, "http"
    if not a.http:
        try:
            ssl_ctx, scheme = certs.ensure_cert(ips), "https"
        except Exception as e:
            print(f"  could not create HTTPS certificate ({e}); install 'cryptography' or use --http")
            scheme = "http"
    print("\nOpen the portal:")
    print(f"  On this computer          : {scheme}://localhost:{a.port}")
    for ip in ips:
        print(f"  Other devices (same Wi-Fi): {scheme}://{ip}:{a.port}")
    if scheme == "https":
        print("\n  Browsers will warn about the self-signed certificate the first time:")
        print("  tap 'Advanced' -> 'Proceed / Accept the risk'. That is expected and safe on your own network.")
    if PIN:
        print("\n  PIN protection is ON.")
    print()
    app.run(host=a.host, port=a.port, ssl_context=ssl_ctx, threaded=True, debug=False)


if __name__ == "__main__":
    main()
