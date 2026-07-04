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
    bbox: Tuple[int, int, int, int] # x,y,w,h
    confidence: float





#
# sprite registration loader
#

class SpriteRegistry:
    def __init__(self, yaml_path:str):
        self.yaml_path = Path(yaml_path)
        self.data = self._load_yaml()
        self.templates = self._load_templates()
        
    def _load_yaml(self):
        with open(self.yaml_path, "r") as f:
            return yaml.full_load(f)
    
    def _load_templates(self) -> Dict[str, List[np.ndarray]]:
        # loads greyscale templates for matching
        
        templates = {}
        
        def walk(node, prefix=""):
            if isinstance(node, dict):
                for k,v in node.items():
                    if k == "sprites":
                        for path in v:
                            full_path = self.yaml_path.parent / path
                            img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
                            if img is None:
                                continue
                            templates[prefix.rstrip(":")] = templates.get(prefix.rstrip(":"), [])
                            templates[prefix.rstrip(":")].append(img)
                    else:
                        walk(v, prefix=f"{k}:")
        
        walk(self.data)
        return templates


#
# perception engine
#

class PerceptionEngine:
    def __init__(self, registry: SpriteRegistry, threshold: float = 0.75):
        self.registry = registry
        self.threshold = threshold
    
    def detect(self, frame: np.ndarray) -> List[Entity]:
        # returns detected entities in the frame
        
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detections = []
        
        for name, templates in self.registry.templates.items():
            for template in templates:
                h, w = template.shape[:2]
                
                result = cv2.matchTemplate(grey,template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(result >= self.threshold)
                
                for pt in zip(*loc[::-1]):
                    x,y = pt
                    
                    detections.append(
                        
                        Entity(
                            name=name,
                            bbox=(x,y,w,h),
                            confidence=float(result[y,x]),
                        )
                    )
        
        return detections
                