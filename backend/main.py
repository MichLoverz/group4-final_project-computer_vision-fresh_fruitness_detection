import cv2
import numpy as np
import onnxruntime as ort
import base64
import io
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# =========================================================
# CONFIG
# =========================================================

IMG_SIZE = 128
CLASSES = ['fresh', 'rotten']
MAX_KEYPOINTS = 100
MIN_MASK_RATIO = 0.30

# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "fruit_model.onnx")
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

app = FastAPI(title="Fruit Freshness API")

# =========================================================
# PREPROCESSING (sama persis dengan model.py)
# =========================================================

def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return img

# =========================================================
# SEGMENTASI (sama persis dengan model.py)
# =========================================================

def _create_center_crop_mask(shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = int(min(h, w) * 0.40)
    cv2.circle(mask, center, radius, 255, thickness=cv2.FILLED)
    return mask

def segment_fruit(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 15, 15])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fruit_mask = None
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        fruit_mask = np.zeros_like(mask)
        cv2.drawContours(fruit_mask, [largest], -1, 255, thickness=cv2.FILLED)
    
    if fruit_mask is None or (fruit_mask > 0).sum() / fruit_mask.size < MIN_MASK_RATIO:
        fruit_mask = _create_center_crop_mask(img.shape)
    
    segmented = cv2.bitwise_and(img, img, mask=fruit_mask)
    return segmented, fruit_mask

# =========================================================
# FEATURE EXTRACTION (sama persis dengan model.py)
# =========================================================

def extract_color_features(img, mask=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    mean_hsv = cv2.mean(hsv, mask=mask)[:3]
    return np.hstack([hist, mean_hsv])

def extract_orb_features(img, mask=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=MAX_KEYPOINTS)
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    if descriptors is None:
        return np.zeros(32), [], gray
    return descriptors.mean(axis=0), keypoints, gray

def extract_features(img):
    preprocessed = preprocess_image(img)
    segmented, mask = segment_fruit(preprocessed)
    color_features = extract_color_features(segmented, mask)
    orb_features, keypoints, gray = extract_orb_features(segmented, mask)
    features = np.hstack([color_features, orb_features]).astype(np.float32)
    return features, preprocessed, segmented, mask, keypoints

# =========================================================
# DETEKSI BUAH (apakah gambar berisi buah)
# =========================================================

def detect_fruit_presence(img):
    """Cek apakah gambar berisi buah berdasarkan segmentasi."""
    preprocessed = preprocess_image(img)
    hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)
    
    # Threshold saturasi — buah punya saturasi tinggi
    # Screenshot/teks biasanya saturasi rendah
    s_channel = hsv[:, :, 1]
    high_sat_ratio = (s_channel > 30).sum() / s_channel.size
    
    # Threshold warna buah
    lower = np.array([0, 40, 40])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    fruit_ratio = (mask > 0).sum() / mask.size
    
    # Harus punya saturasi cukup tinggi DAN area warna buah cukup besar
    return high_sat_ratio > 0.15 and fruit_ratio > 0.10

# =========================================================
# PREDICT
# =========================================================

def predict_image(img):
    features, preprocessed, segmented, mask, keypoints = extract_features(img)
    features_input = features.reshape(1, -1)
    results = session.run(output_names, {input_name: features_input})
    
    prediction = int(results[0][0])
    
    # Confidence dari probability map
    confidence = 0.0
    if len(results) > 1:
        prob_list = results[1]
        prob_dict = prob_list[0] if isinstance(prob_list, list) else prob_list[0]
        if isinstance(prob_dict, dict):
            confidence = prob_dict[prediction] * 100
    
    return CLASSES[prediction], confidence, preprocessed, segmented, mask, keypoints

# =========================================================
# HELPERS
# =========================================================

def mat_to_base64_png(img):
    """Convert cv2 image to base64 PNG string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def draw_keypoints_image(original_resized, keypoints):
    """Draw ORB keypoints (titik merah) pada gambar."""
    result = original_resized.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(result, (x, y), 3, (0, 0, 255), -1)
    return result

# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/")
async def root():
    return {"status": "ok", "message": "Fruit Freshness API"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "fruit_model.onnx"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Terima foto buah, return prediksi + visualisasi.
    
    Response:
    - result: "fresh" atau "rotten"
    - confidence: persentase (0-100)
    - has_fruit: apakah buah terdeteksi
    - preprocessed_image: base64 PNG gambar setelah preprocessing
    - keypoints_image: base64 PNG gambar dengan ORB keypoints (titik merah)
    """
    try:
        # Baca image dari upload
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Gagal membaca gambar"}
            )
        
        # Cek apakah ada buah
        has_fruit = detect_fruit_presence(img)
        
        if not has_fruit:
            return JSONResponse(content={
                "has_fruit": False,
                "result": None,
                "confidence": 0,
                "message": "Buah tidak terdeteksi dalam gambar. Pastikan foto berisi buah dengan pencahayaan cukup.",
                "preprocessed_image": None,
                "keypoints_image": None,
            })
        
        # Predict
        label, confidence, preprocessed, segmented, mask, keypoints = predict_image(img)
        
        # Generate visualisasi
        preprocessed_b64 = mat_to_base64_png(segmented)
        
        # ORB keypoints pada gambar original (resized)
        original_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        keypoints_img = draw_keypoints_image(original_resized, keypoints)
        keypoints_b64 = mat_to_base64_png(keypoints_img)
        
        return JSONResponse(content={
            "has_fruit": True,
            "result": label,
            "confidence": round(confidence, 1),
            "message": None,
            "preprocessed_image": preprocessed_b64,
            "keypoints_image": keypoints_b64,
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Server error: {str(e)}"}
        )
