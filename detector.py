"""ONNX YOLO detector for pedestrians and cars (CPU, Render-free friendly).

Works with:
  * your trained 2-class model  (class 0 = pedestrian, class 1 = car)
  * a stock 80-class COCO YOLOv8 export (keeps person -> pedestrian, car -> car)
Output tensor layout expected: Ultralytics YOLOv8/YOLO11 ONNX, (1, 4+nc, N).
"""
import json
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

COCO_KEEP = {0: "pedestrian", 2: "car"}
DEFAULT_2CLASS = ["pedestrian", "car"]


def letterbox(img, new_h, new_w):
    h, w = img.shape[:2]
    r = min(new_h / h, new_w / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new_h - nh) // 2, (new_w - nw) // 2
    canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


class Detector:
    def __init__(self, model_path, conf=0.35, iou=0.45):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "1"))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        self.in_h = h if isinstance(h, int) else 640
        self.in_w = w if isinstance(w, int) else 640
        self.conf = conf
        self.iou = iou
        self.names_file = os.path.join(os.path.dirname(model_path), "classes.json")

    # ---- class mapping -------------------------------------------------
    def _class_map(self, nc):
        if nc == 80:
            return dict(COCO_KEEP)
        if os.path.exists(self.names_file):
            with open(self.names_file) as f:
                names = json.load(f)
            return {i: n for i, n in enumerate(names)}
        if nc == 2:
            return {i: n for i, n in enumerate(DEFAULT_2CLASS)}
        return {i: f"class_{i}" for i in range(nc)}

    # ---- public API ----------------------------------------------------
    def detect(self, bgr, conf=None):
        h, w = bgr.shape[:2]
        canvas, r, left, top = letterbox(bgr, self.in_h, self.in_w)
        blob = canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        t0 = time.perf_counter()
        out = self.session.run(None, {self.input_name: np.ascontiguousarray(blob)})[0]
        ms = (time.perf_counter() - t0) * 1000
        dets = self.postprocess(out, r, left, top, w, h, conf if conf is not None else self.conf)
        return dets, ms

    def postprocess(self, out, r, left, top, w, h, conf):
        pred = np.squeeze(out, axis=0) if out.ndim == 3 else out
        if pred.shape[0] < pred.shape[1]:          # (4+nc, N) -> (N, 4+nc)
            pred = pred.T
        boxes, scores = pred[:, :4], pred[:, 4:]
        cmap = self._class_map(scores.shape[1])
        keep_idx = [i for i in cmap if i < scores.shape[1]]
        if not keep_idx:
            return []
        sub = scores[:, keep_idx]
        best = sub.argmax(axis=1)
        best_conf = sub.max(axis=1)
        mask = best_conf >= conf
        if not mask.any():
            return []
        boxes, best, best_conf = boxes[mask], best[mask], best_conf[mask]

        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = np.clip((cx - bw / 2 - left) / r, 0, w)
        y1 = np.clip((cy - bh / 2 - top) / r, 0, h)
        x2 = np.clip((cx + bw / 2 - left) / r, 0, w)
        y2 = np.clip((cy + bh / 2 - top) / r, 0, h)

        results = []
        for k, cls_idx in enumerate(keep_idx):
            sel = np.where(best == k)[0]
            if sel.size == 0:
                continue
            rects = [[float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])] for i in sel]
            confs = [float(best_conf[i]) for i in sel]
            for j in np.array(cv2.dnn.NMSBoxes(rects, confs, conf, self.iou)).flatten():
                i = sel[int(j)]
                results.append({
                    "label": cmap[cls_idx],
                    "confidence": round(float(best_conf[i]), 4),
                    "box": [round(float(x1[i]), 1), round(float(y1[i]), 1),
                            round(float(x2[i]), 1), round(float(y2[i]), 1)],
                })
        results.sort(key=lambda d: -d["confidence"])
        return results
