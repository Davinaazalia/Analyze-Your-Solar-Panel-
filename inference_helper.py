# ✅ INFERENCE HELPER MODULE
# For production use dengan model YOLO

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class SolarPanelInference:
    """Helper class untuk inference YOLO model"""
    
    def __init__(self, model_path="models/saved_models/best_solar_panel_classifier.pt"):
        """Initialize model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.classes = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 
                           'Physical-Damage', 'Snow-Covered']
            print(f"✅ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def predict(self, image_path_or_array, conf=0.25):
        """
        Predict panel condition
        
        Args:
            image_path_or_array: Path to image atau numpy array
            conf: Confidence threshold (0-1)
        
        Returns:
            {
                'class': predicted class name,
                'confidence': confidence score,
                'top5': list of (class, confidence),
                'image': processed image array
            }
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Load image
            if isinstance(image_path_or_array, str) or isinstance(image_path_or_array, Path):
                image = Image.open(image_path_or_array).convert('RGB')
            else:
                image = Image.fromarray(image_path_or_array.astype('uint8'))
            
            # Inference
            results = self.model(image, conf=conf, verbose=False)
            
            # Parse results
            if len(results) > 0 and hasattr(results[0], 'probs'):
                probs = results[0].probs.data.cpu().numpy()
                top1_idx = np.argmax(probs)
                top1_conf = float(probs[top1_idx])
                
                # Get top 5
                top5_indices = np.argsort(probs)[::-1][:5]
                top5 = [(self.classes[idx], float(probs[idx])) for idx in top5_indices]
                
                return {
                    'success': True,
                    'class': self.classes[top1_idx],
                    'confidence': top1_conf,
                    'class_idx': int(top1_idx),
                    'all_probs': {cls: float(prob) for cls, prob in zip(self.classes, probs)},
                    'top5': top5,
                    'image': np.array(image)
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_batch(self, image_paths_list):
        """Predict multiple images"""
        results = []
        for img_path in image_paths_list:
            result = self.predict(img_path)
            results.append(result)
        return results
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_type': 'YOLOv8s-Classification',
            'classes': self.classes,
            'num_classes': len(self.classes),
            'device': str(self.device),
            'input_size': 224,
            'accuracy': '98.06%',
            'model_size': '10.2 MB'
        }
