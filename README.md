# Face-Recognition Attendance Portal

A web portal that marks attendance automatically from a webcam. Open it on a laptop, phone or tablet, point the
camera at a student, and the page draws a **box around each face with the name and confidence score**. Once the
face is recognised, attendance is marked and shown live and in the records view.

```
dataset/train/<Student>/*.jpg   ->  train.py  ->  models/   ->  app.py (Flask)  ->  browser (webcam + UI)
```

## 1. Quick start

Requires Python 3.9+ and an internet connection for the first run.

```bash
cd attendance_portal
pip install -r requirements.txt
python download_models.py      # one-time, ~37 MB: the accurate neural-network models
python app.py                  # trains on first run, then starts the portal
```

(`start.sh` on Linux/macOS or `start.bat` on Windows does all of the above for you.)

The server prints the addresses to open, for example:

```
On this computer          : https://localhost:5000
Other devices (same Wi-Fi): https://192.168.1.23:5000
```

> **Important - accuracy:** the *accurate* recognizer needs the files from `download_models.py`.
> Without them the app still runs, but falls back to a basic offline recognizer (shown with an orange banner)
> that is deliberately strict and will show most faces as "Unknown". See section 4.

## 2. Opening the portal from other devices

**Same Wi-Fi (easiest).** Phones/laptops connected to the same Wi-Fi as the server open the
`https://<server-ip>:5000` link printed at start-up.

* The first visit shows a certificate warning (the portal makes its own certificate). Tap
  **Advanced -> Proceed**. This is needed because **browsers only allow the camera on HTTPS pages**.
* If it does not load, allow Python / port 5000 through the firewall on the server computer (Windows shows a pop-up the first time - click *Allow*).
  Some public/campus Wi-Fi networks block devices from talking to each other ("client isolation") - use your phone's hotspot or the tunnel below instead.

**Different Wi-Fi / mobile data / anywhere on the internet.** Your computer's local IP is only reachable on its own network, so
use a tunnel that gives you a public HTTPS link (and a normal, warning-free certificate):

```bash
python app.py --http                 # terminal 1  (the tunnel provides HTTPS)
ngrok http 5000                      # terminal 2  -> prints https://xxxx.ngrok-free.app
#   or:  cloudflared tunnel --url http://localhost:5000
```

Share the printed `https://...` link. The server computer must stay on while people use it.

**Protect it with a PIN** whenever the link is shared publicly:

```bash
PORTAL_PIN=2468 python app.py        # Windows (cmd):  set PORTAL_PIN=2468 && python app.py
```

## 3. Using the portal

* **Live** tab - press *Start camera*. Recognised faces get a green box, `Name  93%` and a confidence bar; unknown faces get an orange box.
  A toast announces `✓ Name marked present`. The right-hand panel shows who is present / not yet seen. *Switch camera* flips front/back camera on phones.
* **Records** tab - pick any date, see Present/Absent, time and match confidence, and export to CSV (one day or all days).
* Data is stored in `data/attendance.db` (SQLite). Each student is recorded **once per day** (first recognition time).

**When is attendance marked?** A student must be recognised in **3 consecutive frames (~1 s)** with confidence >= 70 %.
This avoids marking someone because of a single lucky frame. Change `MARK_FRAMES` / `MARK_MIN_CONF` at the top of `app.py`.

## 4. Accuracy - what to know

* Detection uses OpenCV **YuNet** and recognition uses **SFace** (deep face embeddings). `train.py` computes an embedding for each
  training photo (plus its mirror image) and prints how well students are separated, plus accuracy on `dataset/validation`.
* The **confidence %** is a rescaled similarity score: a match must reach 70 % to be accepted, and ~100 % is a near-perfect match.
  The default accept threshold is cosine similarity 0.38. If real-world use shows missed students, lower it
  (`ATTENDANCE_ACCEPT=0.34`); if the wrong person is ever accepted, raise it (`0.45`).
* To improve results: 8-15 photos per student, different days, lighting, angles and expressions; face clearly visible; avoid heavy filters.
* This is a **basic** system: it has no liveness check, so a printed photo or a phone screen held to the camera can be accepted.
  For real exams/payroll, supervise the camera.
* Offline fallback: if the models cannot be downloaded (e.g. GitHub blocked), download the two `.onnx` files manually into `models/`
  (links inside `download_models.py`), then run `python train.py`.

## 5. Adding / changing students

1. Put photos in `dataset/train/<Student Name>/` (one folder per student).
2. Delete the `models/` files and run `python train.py` (or just restart `app.py` after deleting `models/*`).
3. The roster on the portal updates automatically.

`dataset/validation/` holds one test photo per student named `NAME_1.jpg`; `train.py` reports accuracy on them.

## 6. Files

| File | Purpose |
|---|---|
| `app.py` | Flask server, API, SQLite attendance, HTTPS start-up, optional PIN |
| `face_engine.py` | Detection + recognition (SFace main back-end, LBPH offline fallback) and training |
| `train.py` | Builds the model from `dataset/` and prints a validation report |
| `download_models.py` | Downloads the two ONNX model files |
| `certs.py` | Generates the self-signed HTTPS certificate for LAN use |
| `templates/index.html` | The whole web UI (live camera, overlay boxes, records) |
