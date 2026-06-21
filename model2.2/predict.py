import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import os

# =========================================================
# CONFIG
# =========================================================

IMG_SIZE = 128
CLASSES = ['fresh', 'rotten']
MAX_KEYPOINTS = 100

# =========================================================
# LOAD MODEL ONNX (scaler + SVM sudah terintegrasi)
# =========================================================

ONNX_MODEL_PATH = "fruit_model.onnx"
session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

print(f"Model loaded: {ONNX_MODEL_PATH}")
print(f"Input: {input_name}")
print(f"Outputs: {output_names}")

# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise Exception("Image tidak ditemukan")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = cv2.GaussianBlur(img, (5, 5), 0)

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
# SEGMENTASI
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

    if fruit_mask is None or (fruit_mask > 0).sum() / fruit_mask.size < MIN_MASK_RATIO:
        fruit_mask = _create_center_crop_mask(img.shape)

    segmented = cv2.bitwise_and(img, img, mask=fruit_mask)

    return segmented, fruit_mask


# =========================================================
# FEATURE WARNA
# =========================================================

def extract_color_features(img, mask=None):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        mask,
        [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )

    hist = cv2.normalize(hist, hist).flatten()

    mean_hsv = cv2.mean(hsv, mask=mask)[:3]

    return np.hstack([hist, mean_hsv])


# =========================================================
# FEATURE ORB
# =========================================================

def extract_orb_features(img, mask=None):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=MAX_KEYPOINTS)

    keypoints, descriptors = orb.detectAndCompute(gray, mask)

    if descriptors is None:
        return np.zeros(32)

    return descriptors.mean(axis=0)


# =========================================================
# FEATURE VECTOR
# =========================================================

def extract_features(image_path):

    img = preprocess_image(image_path)

    segmented, mask = segment_fruit(img)

    color_features = extract_color_features(segmented, mask)

    orb_features = extract_orb_features(segmented, mask)

    return np.hstack([
        color_features,
        orb_features
    ])


# =========================================================
# PREDICT IMAGE (menggunakan ONNX model)
# =========================================================

def predict_image(image_path):

    features = extract_features(image_path)

    # ONNX inference (scaler + SVM dalam 1 pipeline)
    features_input = features.astype(np.float32).reshape(1, -1)

    results = session.run(output_names, {input_name: features_input})

    prediction = int(results[0][0])  # label

    # Confidence dari probability map (output kedua dari skl2onnx)
    if len(results) > 1:
        prob_list = results[1]  # list of dicts: [{0: prob, 1: prob}]
        if isinstance(prob_list, list):
            prob_dict = prob_list[0]
        else:
            prob_dict = prob_list[0]
        if isinstance(prob_dict, dict):
            confidence = prob_dict[prediction] * 100
        else:
            confidence = float(prob_dict[prediction]) * 100
        return CLASSES[prediction], confidence

    return CLASSES[prediction], None



# =========================================================
# TEST MANUAL
# =========================================================

folder = "testdata"

images = [
    f for f in os.listdir(folder)
    if f.endswith((".jpg", ".jpeg", ".png"))
]

cols = 5
rows = (len(images) + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(12, 8))

axes = axes.flatten()

for ax, test_img in zip(axes, images):

    path = os.path.join(folder, test_img)

    result, conf = predict_image(path)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax.imshow(img)
    title = f"{test_img}\n{result}"
    if conf: title += f" ({conf:.1f}%)"
    ax.set_title(title, fontsize=10)
    ax.axis("off")

# hapus subplot kosong
for ax in axes[len(images):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("predict_result.png", dpi=100)
print("\nHasil prediksi tersimpan di: predict_result.png")
plt.show()