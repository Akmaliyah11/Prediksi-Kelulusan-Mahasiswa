from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
models = {
    'KNN': joblib.load('models/knn_model.pkl'),
    'SVM': joblib.load('models/svm_model.pkl'),
    'Random Forest': joblib.load('models/rf_model.pkl'),
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form
    jenis_kelamin = request.form['jenis_kelamin']
    umur = request.form['umur']
    status_menikah = request.form['status_menikah']
    kehadiran = request.form['kehadiran']
    diskusi = request.form['diskusi']
    tugas = request.form['tugas']
    elearning = request.form['elearning']
    ipk = request.form['ipk']
    metode = request.form['metode']

    # Konversi ke tipe data yang sesuai
    input_data = np.array([[int(jenis_kelamin), float(umur), int(status_menikah),
                            float(kehadiran), float(diskusi), float(tugas),
                            float(elearning), float(ipk)]])

    # Prediksi
    model = models.get(metode)
    prediction = model.predict(input_data)
    hasil = "Lulus" if prediction[0] == 1 else "Tidak Lulus"
    warna = "green" if hasil == "Lulus" else "red"

    return render_template('index.html',
                           prediction_text=hasil,
                           metode=metode,
                           jenis_kelamin=jenis_kelamin,
                           umur=umur,
                           status_menikah=status_menikah,
                           kehadiran=kehadiran,
                           diskusi=diskusi,
                           tugas=tugas,
                           elearning=elearning,
                           ipk=ipk,
                           warna=warna)

if __name__ == '__main__':
    app.run(debug=True)