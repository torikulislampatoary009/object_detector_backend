import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from detector import Detector, letterbox

class Fake(Detector):
    def __init__(self):
        self.conf, self.iou = 0.35, 0.45
        self.names_file = "/nonexistent/classes.json"

def make(nc, n=8400):
    out = np.zeros((1, 4 + nc, n), np.float32)
    # a pedestrian at (320,320) 100x200, duplicate overlapping, and a car at (100,500) 120x80
    out[0, :4, 0] = [320, 320, 100, 200]; out[0, 4, 0] = 0.9
    out[0, :4, 1] = [322, 322, 100, 200]; out[0, 4, 1] = 0.8
    out[0, :4, 2] = [100, 500, 120, 80];  out[0, 4 + (1 if nc == 2 else 2), 2] = 0.7
    return out

d = Fake()
for nc in (2, 80):
    res = d.postprocess(make(nc), 1.0, 0, 0, 640, 640, 0.35)
    print(nc, res)
    assert len(res) == 2, res
    assert {r["label"] for r in res} == {"pedestrian", "car"}
img = np.zeros((480, 640, 3), np.uint8)
c, r, l, t = letterbox(img, 640, 640)
assert c.shape == (640, 640, 3) and r == 1.0 and t == 80
print("OK")
