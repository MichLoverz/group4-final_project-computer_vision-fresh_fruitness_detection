# Fruit Freshness Detection

Sistem deteksi kesegaran buah (**fresh** vs **rotten**) berbasis *classical computer vision* —
fitur warna (histogram HSV) dan tekstur (ORB) — dengan klasifikasi **Support Vector Machine (SVM)**.

Final Project — **Computer Vision (COMP7116001)**, Kelompok 4.

## Anggota
- Christian — 2802446812
- Fransiskus Owen Ladjuardi — 2802446762
- Michael Peterson — 2802451541
- Nicholas Lee — 2802450721

## Struktur Repositori
- **`model2.2/`** — pipeline ML: preprocessing (resize, Gaussian blur, CLAHE) → segmentasi HSV →
  ekstraksi fitur (histogram HSV + ORB, 547 dimensi) → training SVM → evaluasi → export ONNX.
- **`backend/`** — REST API (FastAPI) yang melayani model `fruit_model.onnx` untuk prediksi.
- **`fruit_freshness_app/`** — aplikasi *mobile* (Flutter): ambil foto buah → kirim ke backend →
  tampilkan hasil (SEGAR/BUSUK + confidence + visualisasi).
- **`Fruit_Freshness_Demo.ipynb`** — notebook demo *end-to-end* (preprocessing → fitur → training → evaluasi).

## Dataset
[Fruits Fresh and Rotten for Classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
(Kaggle) — apel, pisang, jeruk × fresh/rotten. Dataset tidak disertakan di repo; unduh dari Kaggle
dan letakkan di `model2.2/dataset/{train,test}/`.

## Menjalankan Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Hasil
Akurasi **98,48%** pada 2.698 citra uji, dengan *precision*–*recall* seimbang pada kedua kelas (0,98–0,99).
