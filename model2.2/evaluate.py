import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import onnxruntime as ort
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# CONFIG
# =========================================================

TEST_PATH = "dataset/test"
IMG_SIZE = 128
CLASSES = ['fresh', 'rotten']
MAX_KEYPOINTS = 100
MIN_MASK_RATIO = 0.30

# =========================================================
# LOAD MODEL ONNX
# =========================================================

ONNX_MODEL_PATH = "fruit_model.onnx"
session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

print(f"Model loaded: {ONNX_MODEL_PATH}")

# =========================================================
# PIPELINE (sama persis dengan model.py & predict.py)
# =========================================================

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Gagal membaca image: {image_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return img

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
        return np.zeros(32)
    return descriptors.mean(axis=0)

def extract_features(image_path):
    img = preprocess_image(image_path)
    segmented, mask = segment_fruit(img)
    color_features = extract_color_features(segmented, mask)
    orb_features = extract_orb_features(segmented, mask)
    return np.hstack([color_features, orb_features])

def get_label(folder_name):
    folder = folder_name.lower()
    if "fresh" in folder:
        return 0
    elif "rotten" in folder:
        return 1
    return None

# =========================================================
# LOAD TEST DATA & PREDICT
# =========================================================

print("\nLoading test data & predicting...")

y_true = []
y_pred = []

for folder in os.listdir(TEST_PATH):
    folder_path = os.path.join(TEST_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    label = get_label(folder)
    if label is None:
        continue

    print(f"  Processing: {folder}")

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        try:
            features = extract_features(img_path)
            features_input = features.astype(np.float32).reshape(1, -1)
            results = session.run(output_names, {input_name: features_input})
            prediction = int(results[0][0])

            y_true.append(label)
            y_pred.append(prediction)
        except:
            pass

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(f"\nTotal test: {len(y_true)} gambar")

# =========================================================
# EVALUASI
# =========================================================

acc = accuracy_score(y_true, y_pred)

print("\n" + "=" * 40)
print("HASIL EVALUASI MODEL")
print("=" * 40)
print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES,
    annot_kws={"size": 16}
)
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nConfusion Matrix tersimpan: confusion_matrix.png")
plt.show()

# =========================================================
# DISTRIBUSI DATA
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Test distribution
unique_true, counts_true = np.unique(y_true, return_counts=True)
labels_true = [CLASSES[i] for i in unique_true]
axes[0].bar(labels_true, counts_true, color=['green', 'red'])
axes[0].set_title("Test Data Distribution", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts_true):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

# Prediction distribution
unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
labels_pred = [CLASSES[i] for i in unique_pred]
axes[1].bar(labels_pred, counts_pred, color=['green', 'red'])
axes[1].set_title("Prediction Distribution", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Count")
for i, v in enumerate(counts_pred):
    axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("distribution.png", dpi=150)
print("Distribution tersimpan: distribution.png")
plt.show()

# =========================================================
# ACCURACY PER CLASS
# =========================================================

plt.figure(figsize=(7, 5))
per_class_acc = cm.diagonal() / cm.sum(axis=1)
bars = plt.bar(CLASSES, per_class_acc * 100, color=['green', 'red'])
plt.ylim(0, 105)
plt.title("Accuracy Per Class", fontsize=14, fontweight='bold')
plt.xlabel("Class", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
for bar, acc_val in zip(bars, per_class_acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{acc_val*100:.1f}%", ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig("accuracy_per_class.png", dpi=150)
print("Accuracy per class tersimpan: accuracy_per_class.png")
plt.show()

print("\n" + "=" * 40)
print("SEMUA VISUALISASI TERSIMPAN!")
print("=" * 40)
print("  - confusion_matrix.png")
print("  - distribution.png")
print("  - accuracy_per_class.png")
