import logging
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

import config
import db
from detector import Detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": config.CORS_ORIGINS}})

detector = None
model_error = None


def load_model():
    global detector, model_error
    if not os.path.exists(config.MODEL_PATH):
        model_error = f"Model file not found at {config.MODEL_PATH}. Put your trained best.onnx in backend/models/."
        log.error(model_error)
        return
    try:
        detector = Detector(config.MODEL_PATH, config.CONF_THRESHOLD, config.IOU_THRESHOLD)
        log.info("Model loaded: %s (%dx%d)", config.MODEL_PATH, detector.in_w, detector.in_h)
    except Exception as exc:  # noqa: BLE001
        model_error = f"Could not load model: {exc}"
        log.exception(model_error)


load_model()
db.init_db()


def _count(dets, label):
    return sum(1 for d in dets if d["label"] == label)


@app.get("/")
def index():
    return jsonify(service="Real-time Object Detection API", docs="/api/health")


@app.get("/api/health")
def health():
    return jsonify(
        status="ok",
        model_loaded=detector is not None,
        model_error=model_error,
        db_configured=db.is_configured(),
        db_connected=db.ping() if db.is_configured() else False,
    )


@app.post("/api/detect")
def detect():
    if detector is None:
        return jsonify(error=model_error or "Model not loaded"), 503
    file = request.files.get("image")
    if file is None:
        return jsonify(error="Send the picture as multipart field 'image'"), 400

    arr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="Could not read that file as an image"), 400
    h, w = img.shape[:2]

    try:
        conf = float(request.form.get("conf", config.CONF_THRESHOLD))
    except ValueError:
        conf = config.CONF_THRESHOLD
    conf = min(max(conf, 0.05), 0.95)

    dets, ms = detector.detect(img, conf)
    counts = {"pedestrian": _count(dets, "pedestrian"), "car": _count(dets, "car")}

    saved, record_id, save_error = False, None, None
    if request.form.get("save", "false").lower() == "true":
        try:
            record_id = save_detection(
                request.form.get("source", "upload")[:20],
                (file.filename or "")[:255], w, h, counts, ms, dets)
            saved = True
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not save detection: %s", exc)
            save_error = "Detections were found but could not be saved to the database."

    return jsonify(
        detections=dets, counts=counts, image={"width": w, "height": h},
        inference_ms=round(ms, 1), saved=saved, record_id=record_id, save_error=save_error)


def save_detection(source, filename, w, h, counts, ms, dets):
    with db.cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO detections (source, filename, image_width, image_height, pedestrians, cars, inference_ms) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (source, filename, w, h, counts["pedestrian"], counts["car"], ms))
        rid = cur.lastrowid
        if dets:
            cur.executemany(
                "INSERT INTO detection_boxes (detection_id, label, confidence, x1, y1, x2, y2) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                [(rid, d["label"], d["confidence"], *d["box"]) for d in dets])
    return rid


def _db_error(exc):
    log.warning("DB error: %s", exc)
    return jsonify(error="Database is unavailable. Check the DB_* environment variables."), 503


@app.get("/api/history")
def history():
    page = max(int(request.args.get("page", 1)), 1)
    limit = min(max(int(request.args.get("limit", 10)), 1), 50)
    try:
        with db.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM detections")
            total = cur.fetchone()["n"]
            cur.execute("SELECT * FROM detections ORDER BY id DESC LIMIT %s OFFSET %s",
                        (limit, (page - 1) * limit))
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        return _db_error(exc)
    for r in rows:
        r["created_at"] = r["created_at"].isoformat() + "Z"
    return jsonify(items=rows, total=total, page=page, limit=limit)


@app.get("/api/history/<int:rid>")
def history_item(rid):
    try:
        with db.cursor() as cur:
            cur.execute("SELECT * FROM detections WHERE id=%s", (rid,))
            row = cur.fetchone()
            if not row:
                return jsonify(error="Not found"), 404
            cur.execute("SELECT label, confidence, x1, y1, x2, y2 FROM detection_boxes WHERE detection_id=%s", (rid,))
            row["boxes"] = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        return _db_error(exc)
    row["created_at"] = row["created_at"].isoformat() + "Z"
    return jsonify(row)


@app.delete("/api/history/<int:rid>")
def history_delete(rid):
    try:
        with db.cursor(commit=True) as cur:
            cur.execute("DELETE FROM detections WHERE id=%s", (rid,))
            deleted = cur.rowcount
    except Exception as exc:  # noqa: BLE001
        return _db_error(exc)
    return (jsonify(deleted=True), 200) if deleted else (jsonify(error="Not found"), 404)


@app.get("/api/stats")
def stats():
    try:
        with db.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS runs, COALESCE(SUM(pedestrians),0) AS pedestrians, "
                        "COALESCE(SUM(cars),0) AS cars, COALESCE(AVG(inference_ms),0) AS avg_ms FROM detections")
            row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001
        return _db_error(exc)
    row = {k: float(v) if k == "avg_ms" else int(v) for k, v in row.items()}
    row["avg_ms"] = round(row["avg_ms"], 1)
    return jsonify(row)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"Image is larger than {config.MAX_UPLOAD_MB} MB"), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
