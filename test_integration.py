"""
🌞 SOLAR PANEL INTEGRATION TEST
Script untuk test API dan integrasi
"""

import requests
import json
from pathlib import Path
import time

# Configuration
API_URL = "http://127.0.0.1:5000"
TIMEOUT = 30

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'

def print_section(title):
    print(f"\n{BLUE}{'='*50}{END}")
    print(f"{BLUE}{title}{END}")
    print(f"{BLUE}{'='*50}{END}\n")

def print_success(msg):
    print(f"{GREEN}✅ {msg}{END}")

def print_error(msg):
    print(f"{RED}❌ {msg}{END}")

def print_info(msg):
    print(f"{YELLOW}ℹ️  {msg}{END}")

# Test 1: Health Check
def test_health_check():
    print_section("Test 1: API Health Check")
    
    try:
        response = requests.get(f"{API_URL}/", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print_success("API is running")
            print(f"  Status: {data.get('status')}")
            print(f"  Model Loaded: {data.get('model_loaded')}")
            print(f"  Service: {data.get('service')}")
            return True
        else:
            print_error(f"API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API. Is it running?")
        print_info(f"Make sure to run: python api_server.py")
        return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# Test 2: Get Model Info
def test_model_info():
    print_section("Test 2: Get Model Information")
    
    try:
        response = requests.get(f"{API_URL}/api/model-info", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Model info retrieved")
            print(f"  Model Type: {data.get('model_type')}")
            print(f"  Classes: {len(data.get('classes', []))}")
            print(f"  Classes: {', '.join(data.get('classes', []))}")
            print(f"  Device: {data.get('device')}")
            print(f"  Accuracy: {data.get('accuracy')}")
            return True
        else:
            print_error(f"Failed to get model info: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# Test 3: Get Classes
def test_get_classes():
    print_section("Test 3: Get Classes Information")
    
    try:
        response = requests.get(f"{API_URL}/api/classes", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Classes info retrieved")
            
            for class_name, info in data.items():
                status = info.get('status', 'UNKNOWN')
                print(f"\n  {class_name}:")
                print(f"    Status: {status}")
                print(f"    Description: {info.get('description', 'N/A')}")
                print(f"    Urgency: {info.get('urgency', 'N/A')}")
            return True
        else:
            print_error(f"Failed to get classes: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# Test 4: Predict Image
def test_predict_image():
    print_section("Test 4: Predict Image")
    
    # Find a test image
    test_image_paths = [
        'test_image.jpg',
        'data/dataset/Clean/image1.jpg',
        'data/yolo_classify_dataset/test/Clean/image1.jpg'
    ]
    
    test_image = None
    for path in test_image_paths:
        if Path(path).exists():
            test_image = path
            break
    
    if not test_image:
        print_info("No test image found. Skipping prediction test.")
        print_info("To test prediction, place image at root or in data folder")
        return None
    
    try:
        with open(test_image, 'rb') as f:
            files = {'image': f}
            response = requests.post(
                f"{API_URL}/api/predict", 
                files=files, 
                timeout=TIMEOUT
            )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                print_success("Prediction successful")
                pred = data.get('prediction', {})
                info = data.get('info', {})
                
                print(f"\n  Predicted Class: {pred.get('class')}")
                print(f"  Confidence: {pred.get('confidence')}%")
                print(f"  Status: {info.get('status')}")
                print(f"  Description: {info.get('description')}")
                print(f"  Urgency: {info.get('urgency')}")
                print(f"  Risk: {info.get('risk')}")
                print(f"  Maintenance: {info.get('maintenance')}")
                
                print(f"\n  All Probabilities:")
                probs = data.get('all_probabilities', {})
                for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {cls}: {prob:.2f}%")
                
                return True
            else:
                print_error(f"Prediction failed: {data.get('error')}")
                return False
        else:
            print_error(f"API returned status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# Test 5: Frontend Integration
def test_frontend_integration():
    print_section("Test 5: Frontend Files Check")
    
    required_files = [
        'Web_Implementation/index.html',
        'Web_Implementation/panel-predictor.js',
        'Web_Implementation/panel-styles.css',
        'api_server.py',
        'inference_helper.py',
        'models/saved_models/best_solar_panel_classifier.pt'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print_success(f"Found: {file_path}")
        else:
            print_error(f"Missing: {file_path}")
            all_exist = False
    
    return all_exist

# Main test routine
def main():
    print(f"\n{BLUE}{'='*50}{END}")
    print(f"{YELLOW} 🌞 SolarEye Integration Test Suite{END}")
    print(f"{BLUE}{'='*50}{END}")
    
    results = {
        'Health Check': False,
        'Model Info': False,
        'Classes': False,
        'Prediction': None,
        'Frontend': False
    }
    
    # Run tests
    results['Health Check'] = test_health_check()
    if not results['Health Check']:
        print_error("\nCannot proceed with other tests. API is not running.")
        print_info("Run: python api_server.py")
        return
    
    results['Model Info'] = test_model_info()
    results['Classes'] = test_get_classes()
    results['Prediction'] = test_predict_image()
    results['Frontend'] = test_frontend_integration()
    
    # Summary
    print_section("Test Summary")
    
    for test_name, result in results.items():
        if result is None:
            status = f"{YELLOW}⊘ SKIPPED{END}"
        elif result:
            status = f"{GREEN}✅ PASSED{END}"
        else:
            status = f"{RED}❌ FAILED{END}"
        
        print(f"{test_name}: {status}")
    
    # Recommendations
    print_section("Next Steps")
    
    if results['Health Check'] and results['Model Info']:
        print_success("API is working correctly!")
        print_info("You can now use the web interface at:")
        print_info("  - Web_Implementation/index-integrated.html")
        print_info("  - Or integrate with your existing HTML")
        print("\nIntegration steps:")
        print("  1. Add to <head>: <link rel=\"stylesheet\" href=\"./panel-styles.css\">")
        print("  2. Add before </body>: <script src=\"./panel-predictor.js\"></script>")
        print("  3. Add upload zone with id=\"uploadZone\" and input id=\"imageInput\"")
        print("  4. Add result container with id=\"resultContainer\"")
    else:
        print_error("Some tests failed. Check errors above.")
        print_info("Common issues:")
        print_info("  - API server not running")
        print_info("  - Model file not found")
        print_info("  - Dependencies not installed (pip install -r requirements.txt)")
    
    print()

if __name__ == '__main__':
    main()
