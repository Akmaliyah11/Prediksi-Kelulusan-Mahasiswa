# build_model.py
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("Data_Mahasiswa_Bersih.csv", sep=';')
df.drop(columns=['nama'], inplace=True)

# Encoding dan Preprocessing
le = LabelEncoder()
for col in ['jenis_kelamin', 'status_menikah']:
    df[col] = le.fit_transform(df[col])
df['ipk'] = df['ipk'].astype(str).str.replace(',', '.', regex=False).astype(float)
df['status_akademik'] = df['status_akademik'].map({'Lulus': 1, 'Tidak': 0})

X = df.drop(columns=['status_akademik'])
y = df['status_akademik']

# Buat folder models jika belum ada
os.makedirs("models", exist_ok=True)

# Inisialisasi model terbaik (KNN, SVM, Random Forest)
best_knn = KNeighborsClassifier(n_neighbors=1)
best_svm = SVC(kernel='rbf', C=5, gamma='scale', probability=True, class_weight='balanced')  # ✅ updated
best_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Pipeline + Training + Simpan model
pipeline_knn = Pipeline([("scaler", StandardScaler()), ("knn", best_knn)])
pipeline_knn.fit(X, y)
joblib.dump(pipeline_knn, "models/knn_model.pkl")

pipeline_svm = Pipeline([("scaler", StandardScaler()), ("svm", best_svm)])
pipeline_svm.fit(X, y)
joblib.dump(pipeline_svm, "models/svm_model.pkl")

pipeline_rf = Pipeline([("scaler", StandardScaler()), ("rf", best_rf)])
pipeline_rf.fit(X, y)
joblib.dump(pipeline_rf, "models/rf_model.pkl")

print("✅ Semua model berhasil dilatih dan disimpan di folder 'models/'")