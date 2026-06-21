import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# =========================================================
# CONFIG
# =========================================================

# Dataset:
# https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"

IMG_SIZE = 128

# LABEL OUTPUT
CLASSES = ['fresh', 'rotten']

# ORB CONFIG
MAX_KEYPOINTS = 100

# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise Exception(f"Gagal membaca image: {image_path}")

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Gaussian Blur (noise reduction)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Contrast Enhancement (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    cl = clahe.apply(l)

    enhanced = cv2.merge((cl, a, b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return img


# =========================================================
# SEGMENTASI BUAH
# =========================================================

MIN_MASK_RATIO = 0.30  # Jika mask < 30%, fallback ke center crop

def _create_center_crop_mask(shape):
    """Buat mask lingkaran di tengah gambar (fallback)"""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = int(min(h, w) * 0.40)
    cv2.circle(mask, center, radius, 255, thickness=cv2.FILLED)
    return mask

def segment_fruit(img):
    """Segmentasi buah. Return (segmented_img, mask)"""

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold warna buah (diturunkan dari S=30 ke S=15)
    lower = np.array([0, 15, 15])
    upper = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Noise removal
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Contour detection
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    fruit_mask = None

    if len(contours) > 0:

        largest = max(contours, key=cv2.contourArea)

        fruit_mask = np.zeros_like(mask)

        cv2.drawContours(
            fruit_mask,
            [largest],
            -1,
            255,
            thickness=cv2.FILLED
        )

    # Cek apakah mask cukup besar
    if fruit_mask is None or (fruit_mask > 0).sum() / fruit_mask.size < MIN_MASK_RATIO:
        # Fallback: center crop (asumsi buah di tengah frame)
        fruit_mask = _create_center_crop_mask(img.shape)

    segmented = cv2.bitwise_and(img, img, mask=fruit_mask)

    return segmented, fruit_mask


# =========================================================
# FEATURE EXTRACTION - WARNA
# =========================================================

def extract_color_features(img, mask=None):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Histogram HSV (pakai mask agar hanya pixel buah)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        mask,
        [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )

    hist = cv2.normalize(hist, hist).flatten()

    # Mean color (hanya dari area buah)
    mean_hsv = cv2.mean(hsv, mask=mask)[:3]

    return np.hstack([hist, mean_hsv])


# =========================================================
# FEATURE EXTRACTION - TEKSTUR (ORB)
# =========================================================

def extract_orb_features(img, mask=None):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=MAX_KEYPOINTS)

    # Pakai mask agar keypoints hanya di area buah
    keypoints, descriptors = orb.detectAndCompute(gray, mask)

    # Kalau descriptor kosong
    if descriptors is None:
        return np.zeros(32)

    # Rata-rata descriptor
    orb_feature = descriptors.mean(axis=0)

    return orb_feature


# =========================================================
# GABUNG SEMUA FEATURE
# =========================================================

def extract_features(image_path):

    # Preprocessing
    img = preprocess_image(image_path)

    # Segmentasi (return segmented + mask)
    segmented, mask = segment_fruit(img)

    # Feature warna (pakai mask untuk akurasi)
    color_features = extract_color_features(segmented, mask)

    # Feature tekstur ORB (pakai mask)
    orb_features = extract_orb_features(segmented, mask)

    # Gabung feature vector
    feature_vector = np.hstack([
        color_features,
        orb_features
    ])

    return feature_vector


# =========================================================
# AUTO LABEL
# =========================================================

def get_label(folder_name):

    folder = folder_name.lower()

    if "fresh" in folder:
        return 0

    elif "rotten" in folder:
        return 1

    else:
        return None


# =========================================================
# AUGMENTASI (simulasi kondisi kamera HP)
# =========================================================

def augment_image(img):
    """Generate beberapa versi augmented dari 1 gambar (simulasi kamera HP)"""
    augmented = []

    h, w = img.shape[:2]

    # 1. Random brightness (simulasi lighting bervariasi)
    factor = np.random.uniform(0.5, 1.5)
    bright = cv2.convertScaleAbs(img, alpha=factor, beta=0)
    augmented.append(bright)

    # 2. Random saturation & hue jitter (simulasi white balance HP)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-10, 10)) % 180  # hue shift
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.7, 1.3), 0, 255)  # sat
    sat_jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    augmented.append(sat_jittered)

    # 3. Color temperature shift (simulasi lampu kuning indoor)
    temp_img = img.copy().astype(np.float32)
    temp_img[:, :, 0] *= np.random.uniform(0.85, 1.0)   # kurangi blue
    temp_img[:, :, 2] *= np.random.uniform(1.0, 1.15)    # tambah red
    temp_img = np.clip(temp_img, 0, 255).astype(np.uint8)
    augmented.append(temp_img)

    # 4. Random rotation ±15 derajat
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    augmented.append(rotated)

    # 5. Horizontal flip
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)

    # 6. Gaussian noise (simulasi noise kamera HP)
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    augmented.append(noisy)

    # 7. Random contrast
    contrast = np.random.uniform(0.7, 1.3)
    mean = img.mean()
    contrasted = cv2.convertScaleAbs(img, alpha=contrast, beta=mean * (1 - contrast))
    augmented.append(contrasted)

    return augmented


# =========================================================
# PREPROCESSING DARI GAMBAR (bukan path, untuk augmentasi)
# =========================================================

def preprocess_img_direct(img):
    """Preprocessing langsung dari gambar (sudah di-resize)"""

    img = cv2.GaussianBlur(img, (5, 5), 0)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    enhanced = cv2.merge((cl, a, b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return img


# =========================================================
# EXTRACT FEATURES DARI GAMBAR (bukan path)
# =========================================================

def extract_features_from_img(img):
    """Extract features langsung dari gambar yang sudah dipreprocess"""

    segmented, mask = segment_fruit(img)
    color_features = extract_color_features(segmented, mask)
    orb_features = extract_orb_features(segmented, mask)

    return np.hstack([color_features, orb_features])


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(base_path, augment=False):

    X = []
    y = []

    for folder in os.listdir(base_path):

        folder_path = os.path.join(base_path, folder)

        if not os.path.isdir(folder_path):
            continue

        label = get_label(folder)

        if label is None:
            continue

        print(f"Loading folder: {folder}")

        for file in os.listdir(folder_path):

            img_path = os.path.join(folder_path, file)

            try:
                # Original
                features = extract_features(img_path)
                X.append(features)
                y.append(label)

                # Augmented (hanya untuk training)
                if augment:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                    for aug_img in augment_image(img):
                        try:
                            processed = preprocess_img_direct(aug_img)
                            feat = extract_features_from_img(processed)
                            X.append(feat)
                            y.append(label)
                        except:
                            pass

            except Exception as e:
                print("Error:", img_path)

    return np.array(X), np.array(y)


# =========================================================
# LOAD TRAIN & TEST
# =========================================================

print("Loading TRAIN data (dengan augmentasi)...")
X_train, y_train = load_dataset(TRAIN_PATH, augment=True)

print("\nLoading TEST data (tanpa augmentasi)...")
X_test, y_test = load_dataset(TEST_PATH, augment=False)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))


# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================================================
# TRAINING SVM
# =========================================================

print("\nTraining SVM Model...")

model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True
)

model.fit(X_train, y_train)

print("Training selesai!")


# =========================================================
# EVALUASI
# =========================================================

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\n==============================")
print("HASIL EVALUASI")
print("==============================")

print(f"Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASSES
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# =========================================================
# DISTRIBUSI DATA
# =========================================================

def plot_distribution(y, title):

    unique, counts = np.unique(y, return_counts=True)

    labels = [CLASSES[i] for i in unique]

    plt.figure(figsize=(5, 4))

    plt.bar(labels, counts)

    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.show()


plot_distribution(y_train, "Train Distribution")
plot_distribution(y_test, "Test Distribution")


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(model, "model_svm.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel berhasil disimpan!")


# =========================================================
# EXPORT KE ONNX (untuk Flutter)
# =========================================================

print("\nExporting ke ONNX...")

full_pipeline = Pipeline([
    ('scaler', scaler),
    ('svm', model)
])

initial_type = [('input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(full_pipeline, initial_types=initial_type)

with open("fruit_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model berhasil disimpan! (fruit_model.onnx)")


# =========================================================
# PREDIKSI GAMBAR BARU
# =========================================================

def predict_image(image_path):

    features = extract_features(image_path)

    features = scaler.transform([features])

    prediction = model.predict(features)[0]

    return CLASSES[prediction]


# =========================================================
# TEST MANUAL
# =========================================================

test_img = "test.jpg"

if os.path.exists(test_img):

    result = predict_image(test_img)

    img = cv2.imread(test_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(5, 5))

    plt.imshow(img)

    plt.title(f"Prediction: {result}")

    plt.axis("off")

    plt.show()

else:
    print("\nFile test.jpg tidak ditemukan")


# =========================================================
# OPTIONAL - REALTIME CAMERA TEST
# =========================================================

def realtime_detection():

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Resize supaya realtime lebih ringan
        frame = cv2.resize(frame, (320, 240))

        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, frame)

        try:
            result = predict_image(temp_path)

            cv2.putText(
                frame,
                f"Prediction: {result}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        except:
            pass

        cv2.imshow("Fruit Freshness Detection", frame)

        # Tekan Q untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()