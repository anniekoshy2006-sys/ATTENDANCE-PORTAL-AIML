"""
face_engine.py  -  face detection + recognition for the Attendance Portal.

Two interchangeable back-ends, picked automatically:

  1. "sface"  (best accuracy)  OpenCV YuNet detector + SFace deep-embedding model.
     Needs two small ONNX files (downloaded once into ./models on first run).
     "Training" = computing a 128-d embedding for every photo of every student.
     Recognition = cosine similarity between a live face and the stored embeddings.

  2. "lbph"   (fully offline fallback)  Haar-cascade detector + LBPH recognizer
     trained directly on the dataset photos. No downloads needed.

Everything else in the app talks only to FaceEngine, so the back-end is invisible.
"""
import glob
import json
import os
import pickle
import threading
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

SFACE_MODEL_FILE = os.path.join(MODELS_DIR, "face_model_sface.pkl")
LBPH_MODEL_FILE = os.path.join(MODELS_DIR, "face_model_lbph.yml")
LBPH_META_FILE = os.path.join(MODELS_DIR, "face_model_lbph.json")
REPORT_FILE = os.path.join(MODELS_DIR, "training_report.json")

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ---- tunables ------------------------------------------------------------
SFACE_ACCEPT = float(os.environ.get("ATTENDANCE_ACCEPT", 0.38))  # cosine sim to accept (OpenCV: 0.363)
SFACE_MARGIN = 0.05        # best person must beat the runner-up by this much
YUNET_SCORE = 0.80         # detector confidence needed to count as a face
LBPH_FACE = 120            # LBPH face crop size (pixels)
LBPH_ACCEPT = 70.0         # LBPH distance (lower = better); strict on purpose - LBPH is only a fallback


def _list_images(folder):
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        if p.lower().endswith(IMG_EXT):
            out.append(p)
    return out


def _load_resized(path, max_side=1000):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _download(url, dest, timeout=25):
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        f.write(r.read())
    if os.path.getsize(tmp) < 10_000:
        os.remove(tmp)
        raise IOError("downloaded file too small")
    os.replace(tmp, dest)


_DOWNLOAD_FAILED = False


def ensure_sface_models(verbose=True):
    """Download YuNet + SFace if missing. Returns True when both files exist."""
    global _DOWNLOAD_FAILED
    os.makedirs(MODELS_DIR, exist_ok=True)
    if _DOWNLOAD_FAILED and not (os.path.exists(YUNET_PATH) and os.path.exists(SFACE_PATH)):
        return False
    ok = True
    for url, path in ((YUNET_URL, YUNET_PATH), (SFACE_URL, SFACE_PATH)):
        if os.path.exists(path) and os.path.getsize(path) > 10_000:
            continue
        try:
            if verbose:
                print(f"  downloading {os.path.basename(path)} ...")
            _download(url, path)
        except Exception as e:  # no internet, blocked, etc.
            if verbose:
                print(f"  could not download {os.path.basename(path)}: {e}")
            ok = False
            break
    _DOWNLOAD_FAILED = not ok
    return ok


def _confidence_from_similarity(sim, thr):
    """Map cosine similarity to a 0-100 % score. thr -> 70 %, 0.75 -> 100 %, 0.2 -> 0 %."""
    if sim >= thr:
        return 70.0 + 30.0 * min(1.0, (sim - thr) / max(1e-6, 0.75 - thr))
    return max(0.0, 70.0 * (sim - 0.2) / max(1e-6, thr - 0.2))


def _confidence_from_distance(d, thr):
    """Map LBPH distance to 0-100 %. 0 -> 100 %, thr -> 70 %, 2*thr -> 0 %."""
    if d <= thr:
        return 100.0 - 30.0 * d / thr
    return max(0.0, 70.0 * (2 * thr - d) / thr)


class FaceEngine:
    def __init__(self, prefer="auto"):
        self.prefer = prefer          # "auto" | "sface" | "lbph"
        self.backend = None           # set by load()/train()
        self.students = []
        self._lock = threading.Lock()
        # sface
        self._yunet = None
        self._sface = None
        self._gallery = {}            # name -> (N,128) normalised embeddings
        # lbph
        self._haar = None
        self._lbph = None
        self._labels = []
        self._lbph_thr = LBPH_ACCEPT
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------ setup
    def _init_sface_nets(self):
        if not (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")):
            raise RuntimeError("OpenCV too old (need >= 4.5.4)")
        self._yunet = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), YUNET_SCORE, 0.3, 500)
        self._sface = cv2.FaceRecognizerSF.create(SFACE_PATH, "")

    def _init_haar(self):
        if self._haar is None:
            self._haars = [cv2.CascadeClassifier(os.path.join(
                cv2.data.haarcascades, f"haarcascade_frontalface_{n}.xml")) for n in ("default", "alt2")]
            self._haar = True

    def _want_sface(self):
        if self.prefer == "lbph":
            return False
        if os.path.exists(YUNET_PATH) and os.path.exists(SFACE_PATH):
            return True
        if self.prefer == "sface":
            return ensure_sface_models()
        return ensure_sface_models()   # auto: try to download, silently fall back

    def is_trained(self):
        return os.path.exists(SFACE_MODEL_FILE) or os.path.exists(LBPH_MODEL_FILE)

    def load(self):
        """Load a previously trained model. Returns False when (re)training is needed."""
        if self._want_sface():
            if not os.path.exists(SFACE_MODEL_FILE):
                return False                     # nets available -> train the good back-end
            try:
                self._init_sface_nets()
                with open(SFACE_MODEL_FILE, "rb") as f:
                    data = pickle.load(f)
                self._gallery = data["gallery"]
                self.students = sorted(self._gallery)
                self.backend = "sface"
                return True
            except Exception as e:
                print("  SFace back-end failed to start:", e)
        if os.path.exists(LBPH_MODEL_FILE) and os.path.exists(LBPH_META_FILE):
            self._init_haar()
            self._lbph = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8)
            self._lbph.read(LBPH_MODEL_FILE)
            with open(LBPH_META_FILE) as f:
                meta = json.load(f)
            self._labels = meta["labels"]
            self._lbph_thr = LBPH_ACCEPT
            self.students = sorted(self._labels)
            self.backend = "lbph"
            return True
        return False

    # --------------------------------------------------------------- training
    def train(self, train_dir, val_dir=None, log=print):
        """Train on dataset/train/<Student>/*.jpg and evaluate on dataset/validation."""
        people = sorted(d for d in os.listdir(train_dir)
                        if os.path.isdir(os.path.join(train_dir, d)))
        if not people:
            raise RuntimeError(f"No student folders found in {train_dir}")
        os.makedirs(MODELS_DIR, exist_ok=True)

        use_sface = False
        if self.prefer != "lbph":
            try:
                if self._want_sface():
                    self._init_sface_nets()
                    use_sface = True
            except Exception as e:
                log(f"SFace not available ({e}); using offline LBPH back-end instead.")
        report = (self._train_sface if use_sface else self._train_lbph)(people, train_dir, log)
        report["students"] = people
        if val_dir and os.path.isdir(val_dir):
            report["validation"] = self._validate(val_dir, people, log)
        with open(REPORT_FILE, "w") as f:
            json.dump(report, f, indent=2)
        self.students = people
        return report

    def _train_sface(self, people, train_dir, log):
        log("Back-end: SFace deep embeddings (YuNet detector)")
        gallery, per_img, used, skipped = {}, [], 0, []
        for name in people:
            embs = []
            for p in _list_images(os.path.join(train_dir, name)):
                img = _load_resized(p)
                if img is None:
                    continue
                got = False
                for variant in (img, cv2.flip(img, 1)):       # original + mirror
                    faces = self._sface_detect(variant)
                    if not faces:
                        continue
                    f = max(faces, key=lambda r: r[2] * r[3])  # largest face
                    e = self._sface_embed(variant, f)
                    embs.append(e)
                    per_img.append((name, p, e))
                    got = True
                if got:
                    used += 1
                else:
                    skipped.append(os.path.relpath(p, train_dir))
            if not embs:
                raise RuntimeError(f"No usable face found for {name}")
            gallery[name] = np.vstack(embs).astype(np.float32)
            log(f"  {name:<12} {len(embs):>3} embeddings")
        with open(SFACE_MODEL_FILE, "wb") as f:
            pickle.dump({"gallery": gallery}, f)
        self._gallery, self.backend = gallery, "sface"

        # diagnostics: how separable are the students? (same photo + its mirror are excluded)
        gen, imp = [], []
        for n1, p1, e1 in per_img:
            for n2, p2, e2 in per_img:
                if p1 == p2:
                    continue
                (gen if n1 == n2 else imp).append(float(e1 @ e2))
        rep = {"backend": "sface", "images_used": used, "images_skipped": skipped,
               "accept_threshold": SFACE_ACCEPT}
        if gen and imp:
            rep["same_person_similarity"] = {"median": round(float(np.median(gen)), 3),
                                             "5th_percentile": round(float(np.percentile(gen, 5)), 3)}
            rep["different_person_similarity"] = {"median": round(float(np.median(imp)), 3),
                                                  "max": round(max(imp), 3)}
            log(f"  same-person similarity   median {np.median(gen):.2f}  (5th pct {np.percentile(gen, 5):.2f})")
            log(f"  different-person sim.    median {np.median(imp):.2f}  (max {max(imp):.2f})")
            log(f"  accept threshold         {SFACE_ACCEPT:.2f}   (set ATTENDANCE_ACCEPT to change)")
        return rep

    def _train_lbph(self, people, train_dir, log):
        log("Back-end: LBPH (offline) with Haar-cascade detector")
        self._init_haar()
        faces, labels, used, skipped = [], [], 0, []
        for li, name in enumerate(people):
            n = 0
            for p in _list_images(os.path.join(train_dir, name)):
                img = _load_resized(p)
                dets = self._haar_detect(img, 3) if img is not None else []
                if not dets:
                    skipped.append(os.path.relpath(p, train_dir))
                    continue
                d = max(dets, key=lambda r: r["box"][2] * r["box"][3])
                for crop in self._lbph_augment(d["img"], d["ibox"]):
                    faces.append(crop)
                    labels.append(li)
                    n += 1
                used += 1
            log(f"  {name:<12} {n:>3} training samples (with augmentation)")
        if not faces:
            raise RuntimeError("No faces detected in the training images")
        rec = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8)
        rec.train(faces, np.array(labels, dtype=np.int32))
        rec.write(LBPH_MODEL_FILE)
        with open(LBPH_META_FILE, "w") as f:
            json.dump({"labels": people, "threshold": LBPH_ACCEPT}, f)
        self._lbph, self._labels, self._lbph_thr, self.backend = rec, people, LBPH_ACCEPT, "lbph"
        return {"backend": "lbph", "images_used": used, "images_skipped": skipped,
                "training_samples": len(faces)}

    def _validate(self, val_dir, people, log):
        """Validation files are named like ABEL_1.jpeg -> the prefix is the student."""
        by_lower = {p.lower(): p for p in people}
        rows, correct = [], 0
        for p in _list_images(val_dir):
            stem = os.path.splitext(os.path.basename(p))[0]
            truth = by_lower.get(stem.split("_")[0].lower())
            img = _load_resized(p)
            res = self.identify(img) if img is not None else []
            if res:
                top = max(res, key=lambda r: r["box"][2] * r["box"][3])
                pred, conf = top["name"], top["confidence"]
            else:
                pred, conf = None, 0.0
            ok = (pred == truth)
            correct += ok
            rows.append({"file": os.path.basename(p), "expected": truth,
                         "predicted": pred, "confidence": round(conf, 1), "correct": ok})
            log(f"  {os.path.basename(p):<18} expected={truth!s:<8} got={pred!s:<8} "
                f"conf={conf:5.1f}%  {'OK' if ok else 'MISS'}")
        total = len(rows)
        return {"total": total, "correct": correct,
                "accuracy": round(100.0 * correct / total, 1) if total else None,
                "details": rows}

    # ------------------------------------------------------------ inference
    def identify(self, bgr):
        """Detect every face in a BGR image and identify each one.

        Returns a list of dicts:
          box [x, y, w, h], name (or "Unknown"), known (bool),
          confidence (0-100 recognition confidence), det_score (0-100)
        """
        with self._lock:
            if self.backend == "sface":
                return self._identify_sface(bgr)
            if self.backend == "lbph":
                return self._identify_lbph(bgr)
        raise RuntimeError("Model not loaded - run train.py first")

    # -- sface
    def _sface_detect(self, bgr):
        h, w = bgr.shape[:2]
        self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(bgr)
        return [] if faces is None else [f for f in faces]

    def _sface_embed(self, bgr, face_row):
        aligned = self._sface.alignCrop(bgr, face_row)
        feat = self._sface.feature(aligned).astype(np.float32).ravel()
        return feat / (np.linalg.norm(feat) + 1e-9)

    def _identify_sface(self, bgr):
        out = []
        for f in self._sface_detect(bgr):
            x, y, w, h = [int(round(v)) for v in f[:4]]
            emb = self._sface_embed(bgr, f)
            # score per student: mean of the top-3 similarities to that student's gallery
            scores = {}
            for name, g in self._gallery.items():
                sims = np.sort(g @ emb)[::-1]
                scores[name] = float(np.mean(sims[:3]))
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            best_name, best = ranked[0]
            runner = ranked[1][1] if len(ranked) > 1 else -1.0
            known = best >= SFACE_ACCEPT and (best - runner) >= SFACE_MARGIN
            out.append({
                "box": [x, y, w, h],
                "name": best_name if known else "Unknown",
                "known": bool(known),
                "confidence": round(_confidence_from_similarity(best, SFACE_ACCEPT) if known
                                    else min(69.0, _confidence_from_similarity(best, SFACE_ACCEPT)), 1),
                "similarity": round(best, 3),
                "det_score": round(float(f[14]) * 100, 1),
            })
        return out

    # -- lbph
    def _haar_detect(self, bgr, min_neighbors=4):
        """Tilt-tolerant Haar search. Returns [{box, img, ibox}] where `box` is in the
        original frame and (`img`, `ibox`) is the (possibly de-rotated) image + box to crop from."""
        self._init_haar()
        h, w = bgr.shape[:2]
        gray0 = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        ms = max(50, int(min(h, w) * 0.15))
        for ang in (0, -12, 12, -24, 24, -36, 36):
            if ang:
                M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
                gray = cv2.warpAffine(gray0, M, (w, h))
            else:
                M, gray = None, gray0
            found = []
            for casc in self._haars:
                found = casc.detectMultiScale(gray, 1.1, min_neighbors, minSize=(ms, ms))
                if len(found):
                    break
            if not len(found):
                continue
            res = []
            for (x, y, fw, fh) in found:
                if M is None:
                    img, box = bgr, (int(x), int(y), int(fw), int(fh))
                    obox = box
                else:
                    img = cv2.warpAffine(bgr, M, (w, h))
                    box = (int(x), int(y), int(fw), int(fh))
                    inv = cv2.invertAffineTransform(M)
                    cx, cy = inv @ np.array([x + fw / 2, y + fh / 2, 1.0])
                    obox = (int(cx - fw / 2), int(cy - fh / 2), int(fw), int(fh))
                res.append({"box": obox, "img": img, "ibox": box})
            return res
        return []

    def _lbph_prep(self, bgr, box):
        x, y, w, h = box
        # trim a little forehead/background so crops are consistent
        mx, my = int(w * 0.05), int(h * 0.05)
        x0, y0 = max(0, x + mx), max(0, y + my)
        x1, y1 = min(bgr.shape[1], x + w - mx), min(bgr.shape[0], y + h)
        crop = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, (LBPH_FACE, LBPH_FACE), interpolation=cv2.INTER_AREA)
        return self._clahe.apply(crop)

    def _lbph_augment(self, bgr, box):
        """Create several variations of one training face (shift / scale / flip / light)."""
        x, y, w, h = box
        samples = []
        for dx, dy, sc in [(0, 0, 1.0), (0.04, 0, 1.0), (-0.04, 0, 1.0), (0, 0.04, 1.0),
                           (0, -0.04, 1.0), (0, 0, 1.08), (0, 0, 0.93)]:
            nw, nh = int(w * sc), int(h * sc)
            nx = int(x + dx * w - (nw - w) / 2)
            ny = int(y + dy * h - (nh - h) / 2)
            nx, ny = max(0, nx), max(0, ny)
            crop = self._lbph_prep(bgr, (nx, ny, nw, nh))
            samples.append(crop)
            samples.append(cv2.flip(crop, 1))
            samples.append(cv2.convertScaleAbs(crop, alpha=0.8, beta=-10))
            samples.append(cv2.convertScaleAbs(crop, alpha=1.15, beta=15))
            samples.append(cv2.GaussianBlur(crop, (5, 5), 0))
        return samples

    def _identify_lbph(self, bgr):
        out = []
        for d in self._haar_detect(bgr):
            box = d["box"]
            crop = self._lbph_prep(d["img"], d["ibox"])
            label, dist = self._lbph.predict(crop)
            known = dist <= self._lbph_thr
            out.append({
                "box": list(box),
                "name": self._labels[label] if known else "Unknown",
                "known": bool(known),
                "confidence": round(_confidence_from_distance(dist, self._lbph_thr) if known
                                    else min(69.0, _confidence_from_distance(dist, self._lbph_thr)), 1),
                "similarity": round(float(dist), 1),
                "det_score": 100.0,
            })
        return out
