"""Stage 1: find every bird in a photo with a COCO-pretrained YOLO model.

COCO already includes a "bird" class, so no box annotation is needed to start.
For large photos of flocks, --tile runs the detector on overlapping tiles so
small, distant birds are not missed, then merges duplicate boxes.

Note: Ultralytics YOLO is licensed AGPL-3.0. Fine for research and personal use;
a closed-source commercial app would need an Ultralytics enterprise license.
"""

from dataclasses import dataclass

import numpy as np

COCO_BIRD_CLASS = 14


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1


def _nms(dets, iou_threshold=0.5):
    """Merge overlapping boxes from neighbouring tiles (keep the highest score)."""
    if not dets:
        return dets
    boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets])
    scores = np.array([d.score for d in dets])
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_threshold]
    return [dets[i] for i in keep]


class BirdDetector:
    def __init__(self, weights="yolo11n.pt", conf=0.25, imgsz=640, max_det=1000):
        # PyTorch must be fully loaded before TensorFlow: loading TF first and then
        # torchvision crashes the process. Import torchvision and run one dummy
        # prediction now, so YOLO's lazy imports finish before the classifier loads.
        import torchvision  # noqa: F401
        from ultralytics import YOLO   # imported here so training doesn't need it
        self.model = YOLO(weights)
        self.model.predict(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
        self.conf = conf
        self.imgsz = imgsz
        self.max_det = max_det
        # a fine-tuned bird-only model has class 0; the COCO model uses class 14
        names = self.model.names
        self.bird_class = next((k for k, v in names.items() if str(v).lower() == "bird"),
                               COCO_BIRD_CLASS)

    def _run(self, image, offset=(0, 0)):
        res = self.model.predict(image, conf=self.conf, imgsz=self.imgsz,
                                 classes=[self.bird_class], max_det=self.max_det,
                                 verbose=False)[0]
        ox, oy = offset
        out = []
        for (x1, y1, x2, y2), s in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            out.append(Detection(x1 + ox, y1 + oy, x2 + ox, y2 + oy, float(s)))
        return out

    def detect(self, image, tile=False, tile_size=960, overlap=0.2):
        """image: RGB numpy array (H, W, 3). Returns a list of Detection, largest first."""
        h, w = image.shape[:2]
        dets = self._run(image)
        if tile and max(h, w) > tile_size * 1.3:
            step = int(tile_size * (1 - overlap))
            for y in range(0, max(h - tile_size, 0) + step, step):
                for x in range(0, max(w - tile_size, 0) + step, step):
                    crop = image[y:y + tile_size, x:x + tile_size]
                    dets += self._run(np.ascontiguousarray(crop), offset=(x, y))
            dets = _nms(dets)
        return sorted(dets, key=lambda d: d.width * d.height, reverse=True)
