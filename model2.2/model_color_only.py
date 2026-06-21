"""
Retrain model tanpa ORB features (515 color features only).
ORB tidak tersedia di Android ARM64 opencv_dart.
"""
import cv2
import os
import numpy as np
import struct

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# =========================================================
# CONFIG
# =========================================================

TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"
IMG_SIZE = 128
CLASSES = ['fresh', 'rotten']

# =========================================================
# PREPROCESSING (identik dengan model.py)
# =========================================================

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Gagal membaca: {image_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return img

def preprocess_img_direct(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return img

# =========================================================
# SEGMENTASI (identik dengan model.py)
# =========================================================

MIN_MASK_RATIO = 0.30

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
# FEATURE EXTRACTION - COLOR ONLY (515 features)
# =========================================================

def extract_color_features(img, mask=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    mean_hsv = cv2.mean(hsv, mask=mask)[:3]
    return np.hstack([hist, mean_hsv])

def extract_features(image_path):
    img = preprocess_image(image_path)
    segmented, mask = segment_fruit(img)
    color_features = extract_color_features(segmented, mask)
    return color_features  # 515 features only

def extract_features_from_img(img):
    segmented, mask = segment_fruit(img)
    color_features = extract_color_features(segmented, mask)
    return color_features

# =========================================================
# AUGMENTASI (identik dengan model.py)
# =========================================================

def augment_image(img):
    augmented = []
    h, w = img.shape[:2]
    factor = np.random.uniform(0.5, 1.5)
    augmented.append(cv2.convertScaleAbs(img, alpha=factor, beta=0))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.7, 1.3), 0, 255)
    augmented.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    temp_img = img.copy().astype(np.float32)
    temp_img[:, :, 0] *= np.random.uniform(0.85, 1.0)
    temp_img[:, :, 2] *= np.random.uniform(1.0, 1.15)
    augmented.append(np.clip(temp_img, 0, 255).astype(np.uint8))
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    augmented.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    augmented.append(cv2.flip(img, 1))
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    augmented.append(cv2.add(img, noise))
    contrast = np.random.uniform(0.7, 1.3)
    mean = img.mean()
    augmented.append(cv2.convertScaleAbs(img, alpha=contrast, beta=mean * (1 - contrast)))
    return augmented

# =========================================================
# LOAD DATASET
# =========================================================

def get_label(folder_name):
    folder = folder_name.lower()
    if "fresh" in folder: return 0
    elif "rotten" in folder: return 1
    return None

def load_dataset(base_path, augment=False):
    X, y = [], []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path): continue
        label = get_label(folder)
        if label is None: continue
        print(f"  Loading: {folder}")
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            try:
                features = extract_features(img_path)
                X.append(features)
                y.append(label)
                if augment:
                    img = cv2.imread(img_path)
                    if img is None: continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    for aug_img in augment_image(img):
                        try:
                            processed = preprocess_img_direct(aug_img)
                            feat = extract_features_from_img(processed)
                            X.append(feat)
                            y.append(label)
                        except: pass
            except Exception as e:
                print(f"  Error: {img_path}: {e}")
    return np.array(X), np.array(y)

# =========================================================
# TRAIN
# =========================================================

print("Loading TRAIN data (with augmentation)...")
X_train, y_train = load_dataset(TRAIN_PATH, augment=True)
print(f"\nLoading TEST data...")
X_test, y_test = load_dataset(TEST_PATH, augment=False)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
print(f"Features per sample: {X_train.shape[1]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining SVM (color features only)...")
model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
model.fit(X_train_scaled, y_train)
print("Training done!")

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=CLASSES))

# =========================================================
# EXPORT TO BINARY (for Dart)
# =========================================================

n_sv = model.n_support_.sum()
n_features = 515
gamma = model._gamma  # actual gamma used
dual_coef = model.dual_coef_.flatten().astype(np.float32)
support_vectors = model.support_vectors_.astype(np.float32)
intercept = model.intercept_.astype(np.float32)
scaler_mean = scaler.mean_.astype(np.float32)
scaler_scale = (1.0 / scaler.scale_).astype(np.float32)  # ONNX Scaler format: scale = 1/std

# Platt scaling params
from sklearn.calibration import CalibratedClassifierCV
prob_a = np.float32(0.0)
prob_b = np.float32(0.0)
if hasattr(model, '_predict_proba_lr'):
    # Get probA and probB from libsvm
    # For binary SVM, probA and probB are used for Platt scaling
    pass

# Test to find probA/probB by fitting sigmoid
from scipy.optimize import minimize_scalar
decisions = model.decision_function(X_test_scaled)
# Platt scaling: P(y=1|f) = 1 / (1 + exp(A*f + B))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(decisions.reshape(-1, 1), y_test)
prob_a_val = np.float32(-lr.coef_[0][0])  # negative because Platt uses -coef
prob_b_val = np.float32(-lr.intercept_[0])

print(f"\nModel info:")
print(f"  Support vectors: {n_sv}")
print(f"  Features: {n_features}")
print(f"  Gamma: {gamma}")
print(f"  Intercept (rho): {intercept[0]}")
print(f"  Prob A: {prob_a_val}, Prob B: {prob_b_val}")

out = '../fruit_freshness_app/assets/model_params.bin'
with open(out, 'wb') as f:
    f.write(struct.pack('<I', int(n_sv)))
    f.write(struct.pack('<I', n_features))
    f.write(scaler_mean.tobytes())      # 515 floats
    f.write(scaler_scale.tobytes())     # 515 floats
    f.write(dual_coef.tobytes())        # n_sv floats
    f.write(support_vectors.tobytes())  # n_sv * 515 floats
    f.write(intercept.tobytes())        # 1 float (rho = -intercept for ONNX convention)
    f.write(np.float32(gamma).tobytes())# 1 float
    f.write(prob_a_val.tobytes())       # 1 float
    f.write(prob_b_val.tobytes())       # 1 float

print(f"  Binary size: {os.path.getsize(out)/1024/1024:.1f} MB")

# =========================================================
# VERIFY: manual inference matches sklearn
# =========================================================

print("\nVerifying manual inference...")
match = 0
total = min(100, len(X_test))
for i in range(total):
    features = X_test[i]
    scaled = (features - scaler.mean_) / scaler.scale_
    
    # Manual SVM decision
    decision = 0.0
    for j in range(int(n_sv)):
        diff = scaled - support_vectors[j]
        sq_dist = np.dot(diff, diff)
        k = np.exp(-gamma * sq_dist)
        decision += dual_coef[j] * k
    decision += intercept[0]  # sklearn uses +intercept, not -rho
    
    manual_label = 1 if decision > 0 else 0
    sklearn_label = model.predict(X_test_scaled[i:i+1])[0]
    
    if manual_label == sklearn_label:
        match += 1

print(f"Match: {match}/{total}")

# Verify with testdata
print("\nPredicting testdata...")
test_dir = "testdata"
if os.path.exists(test_dir):
    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith(('.jpg', '.jpeg', '.png')): continue
        path = os.path.join(test_dir, fname)
        feat = extract_features(path)
        feat_scaled = scaler.transform(feat.reshape(1, -1))
        pred = model.predict(feat_scaled)[0]
        prob = model.predict_proba(feat_scaled)[0]
        print(f"  {fname}: {CLASSES[pred]} (fresh={prob[0]*100:.1f}%, rotten={prob[1]*100:.1f}%)")
