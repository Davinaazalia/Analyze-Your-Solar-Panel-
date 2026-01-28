#!/usr/bin/env python3
"""
Solar Panel Classification - Production Inference Script
Model: yolov8s-cls.pt
Classes: Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered
"""

from ultralytics import YOLO
from pathlib import Path

# Load model
model = YOLO('models/saved_models/best_solar_panel_classifier.pt')

# Classes
CLASSES = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

def predict(image_path, conf_threshold=0.5):
    """Predict solar panel class"""
    results = model.predict(source=image_path, verbose=False)
    result = results[0]

    top1_idx = int(result.probs.top1)
    top1_conf = float(result.probs.top1conf)

    if top1_conf >= conf_threshold:
        return CLASSES[top1_idx], top1_conf
    else:
        return "Low Confidence", top1_conf

# Example usage
if __name__ == "__main__":
    img_path = "path/to/your/image.jpg"
    class_name, confidence = predict(img_path)
    print(f"Predicted: {class_name} ({confidence:.2%})")
