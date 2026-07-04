from dataclasses import dataclass
from typing import Dict, List, Tuple
import cv2
import numpy as np
import yaml
from pathlib import Path

#
# core data struct
#
@dataclass
class Entity:
    name: str
    bbox: Tuple[int, int, int, int]  # x,y,w,h
    confidence: float

#
# sprite registration loader
#
class SpriteRegistry:
    def __init__(self, yaml_path: str):
        self.yaml_path = Path(yaml_path)
        self.data = self._load_yaml()
        self.templates = self._load_templates()

    def _load_yaml(self):
        with self.yaml_path.open("r") as f:
            return yaml.safe_load(f)

    def _load_templates(self) -> Dict[str, List[np.ndarray]]:
        # loads greyscale templates for matching
        templates: Dict[str, List[np.ndarray]] = {}

        def walk(node, prefix: str = ""):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "sprites" and isinstance(v, list):
                        key_name = prefix.rstrip(":")
                        for path in v:
                            full_path = self.yaml_path.parent / path
                            img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
                            #img = cv2.GaussianBlur(img, (3,3), 0)
                            if img is None:
                                continue
                            templates.setdefault(key_name, []).append(img)
                    else:
                        new_prefix = f"{prefix}{k}:"
                        walk(v, new_prefix)
            elif isinstance(node, list):
                for item in node:
                    walk(item, prefix)
            # ignore other node types

        walk(self.data)
        return templates

#
# perception engine
#
class PerceptionEngine:
    def __init__(self, registry: SpriteRegistry, threshold: float = 0.75):
        self.registry = registry
        self.threshold = float(threshold)

    def detect(self, frame: np.ndarray) -> List[Entity]:
        # returns detected entities in the frame
        if frame is None:
            return []

        if frame.ndim == 2:
            grey = frame
        else:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #grey = cv2.GaussianBlur(grey, (3,3), 0)
        detections: List[Entity] = []
        for name, templates in self.registry.templates.items():
            for template in templates:
                h, w = template.shape[:2]
                if grey.shape[0] < h or grey.shape[1] < w:
                    continue
                result = cv2.matchTemplate(grey, template, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(result >= self.threshold)
                for y, x in zip(ys, xs):
                    detections.append(
                        Entity(
                            name=name,
                            bbox=(int(x), int(y), int(w), int(h)),
                            confidence=float(result[y, x]),
                        )
                    )
        return detections