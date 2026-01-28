"""
🌞 SOLAR PANEL API SERVER
REST API untuk integrasi YOLO dengan HTML Frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np
from inference_helper import SolarPanelInference
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS untuk akses dari HTML

# Load model saat startup
try:
    model = SolarPanelInference(model_path="models/saved_models/best_solar_panel_classifier.pt")
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = None

# Mapping kelas ke rekomendasi dan info
CLASS_INFO = {
    'Clean': {
        'status': 'GOOD',
        'color': '#27ae60',  # Green
        'description': 'Panel dalam kondisi sempurna',
        'urgency': 'Low',
        'maintenance': 'Lakukan pemeriksaan rutin setiap 3 bulan',
        'risk': 'Minimal'
    },
    'Dusty': {
        'status': 'WARNING',
        'color': '#f39c12',  # Orange
        'description': 'Panel tertutup debu/kotoran',
        'urgency': 'Medium',
        'maintenance': 'Bersihkan panel segera untuk efisiensi maksimal',
        'risk': 'Moderate - Mengurangi output hingga 25%'
    },
    'Bird-drop': {
        'status': 'WARNING',
        'color': '#f39c12',  # Orange
        'description': 'Terdapat kotoran burung',
        'urgency': 'High',
        'maintenance': 'Bersihkan segera untuk mencegah kerusakan',
        'risk': 'High - Dapat menyebabkan hot spots'
    },
    'Snow-Covered': {
        'status': 'CRITICAL',
        'color': '#e74c3c',  # Red
        'description': 'Panel tertutup salju',
        'urgency': 'Critical',
        'maintenance': 'Hapus salju dengan hati-hati menggunakan alat yang tepat',
        'risk': 'Critical - Output berkurang drastis'
    },
    'Electrical-damage': {
        'status': 'CRITICAL',
        'color': '#e74c3c',  # Red
        'description': 'Kerusakan elektrik terdeteksi',
        'urgency': 'Critical',
        'maintenance': 'Hubungi teknisi profesional segera untuk inspeksi',
        'risk': 'Critical - Risiko keselamatan'
    },
    'Physical-Damage': {
        'status': 'CRITICAL',
        'color': '#e74c3c',  # Red
        'description': 'Kerusakan fisik pada panel',
        'urgency': 'Critical',
        'maintenance': 'Hubungi layanan perbaikan profesional',
        'risk': 'Critical - Memerlukan penggantian'
    }
}


@app.route('/', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'service': 'SolarEye Panel Detection API'
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Expects: multipart/form-data dengan field 'image' berisi file gambar
    Returns: JSON dengan hasil prediksi
    """
    try:
        # Check if image exists
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Read dan process image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Save temporarily untuk inference
        temp_path = 'temp_input.jpg'
        image.save(temp_path)
        
        # Run prediction
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        result = model.predict(temp_path, conf=0.25)
        
        if not result.get('success', False):
            return jsonify({'error': result.get('error', 'Prediction failed')}), 400
        
        # Format response dengan informasi tambahan
        predicted_class = result['class']
        confidence = result['confidence']
        
        # Get class info
        info = CLASS_INFO.get(predicted_class, {})
        
        # Convert image to base64 untuk preview
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Check if confidence is low (possible non-panel image)
        confidence_percentage = round(confidence * 100, 2)
        low_confidence_warning = None
        
        if confidence < 0.70:  # Below 70% confidence
            low_confidence_warning = {
                'message': 'Low confidence detected',
                'details': 'The model is not very confident about this prediction. The image might not be a solar panel or might be unclear.',
                'suggestion': 'Please try uploading a clear, well-lit image of a solar panel',
                'severity': 'warning'
            }
        elif confidence < 0.50:  # Below 50% confidence
            low_confidence_warning = {
                'message': 'Very low confidence',
                'details': 'The model cannot reliably identify this as a solar panel image. This might not be a solar panel.',
                'suggestion': 'Please check the image and upload a clear solar panel image',
                'severity': 'error'
            }
        
        response = {
            'success': True,
            'prediction': {
                'class': predicted_class,
                'confidence': confidence_percentage,
                'class_idx': result['class_idx']
            },
            'all_probabilities': {
                cls: round(prob * 100, 2) 
                for cls, prob in result['all_probs'].items()
            },
            'top5': [
                {'class': cls, 'confidence': round(conf * 100, 2)} 
                for cls, conf in result['top5']
            ],
            'info': {
                'status': info.get('status', 'UNKNOWN'),
                'color': info.get('color', '#95a5a6'),
                'description': info.get('description', ''),
                'urgency': info.get('urgency', 'Unknown'),
                'maintenance': info.get('maintenance', ''),
                'risk': info.get('risk', '')
            },
            'image': f"data:image/jpeg;base64,{img_base64}",
            'warning': low_confidence_warning  # Add warning if exists
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    info = model.get_model_info()
    return jsonify(info), 200


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of classes dengan info"""
    classes_with_info = {
        cls: CLASS_INFO.get(cls, {}) 
        for cls in model.classes if model
    }
    return jsonify(classes_with_info), 200


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Predict multiple images
    Expects: JSON dengan array base64 images
    """
    try:
        data = request.get_json()
        
        if 'images' not in data:
            return jsonify({'error': 'No images provided'}), 400
        
        results = []
        
        for idx, img_data in enumerate(data['images']):
            try:
                # Decode base64
                if img_data.startswith('data:image'):
                    img_data = img_data.split(',')[1]
                
                image = Image.open(io.BytesIO(base64.b64decode(img_data))).convert('RGB')
                
                # Save temporarily
                temp_path = f'temp_batch_{idx}.jpg'
                image.save(temp_path)
                
                # Predict
                result = model.predict(temp_path, conf=0.25)
                
                if result.get('success', False):
                    predicted_class = result['class']
                    confidence = result['confidence']
                    info = CLASS_INFO.get(predicted_class, {})
                    
                    results.append({
                        'index': idx,
                        'success': True,
                        'prediction': predicted_class,
                        'confidence': round(confidence * 100, 2),
                        'status': info.get('status', 'UNKNOWN'),
                        'urgency': info.get('urgency', 'Unknown')
                    })
                else:
                    results.append({
                        'index': idx,
                        'success': False,
                        'error': result.get('error', 'Prediction failed')
                    })
            
            except Exception as e:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total': len(data['images']),
            'processed': len([r for r in results if r.get('success')]),
            'results': results
        }), 200
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


if __name__ == '__main__':
    # Development server
    app.run(debug=True, host='127.0.0.1', port=5000)
    
    # Untuk production:
    # app.run(debug=False, host='0.0.0.0', port=5000)
