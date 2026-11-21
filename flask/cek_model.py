import joblib
import os

path = "models/knn_model.pkl"

if os.path.exists(path):
    print(f"✅ File ditemukan: {path}")
    try:
        model = joblib.load(path)
        print("✅ Model berhasil dimuat!")
    except Exception as e:
        print("❌ Gagal memuat model:", e)
else:
    print(f"❌ File tidak ditemukan: {path}")