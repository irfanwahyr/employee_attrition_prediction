# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Maju merupakan salah satu perusahaan multinasional yang telah berdiri sejak tahun 2000. Ia memiliki lebih dari 1000 karyawan yang tersebar di seluruh penjuru negeri.
Walaupun telah menjadi menjadi perusahaan yang cukup besar, Jaya Jaya Maju masih cukup kesulitan dalam mengelola karyawan. Hal ini berimbas tingginya attrition rate (rasio jumlah karyawan yang keluar dengan total karyawan keseluruhan) hingga lebih dari 10%. tentu saja ini tidak bagus untuk kelangsungan perusahaan.

### Permasalahan Bisnis

* tingginya attrition rate (lebih dari 10%)
* faktor yang mempengaruhi tingginya attrition rate

### Cakupan Proyek

Cakupan proyek kali ni akan melakukan Employee segemntation dengan machine learning yaang akan menggunakan algoritma yang sesuai dataaset. akan dianalisis juga dataset tersebut dengan EDA sehingga gambaran permasalahan akan jelas. sehingga berdasarkan cakupan tersebut, maka akan digunakan tools sebagai berikut.
* [dataset](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee)
* bahasa pemrograman Python
* metabase dan supabase untuk visualisasi dataset serta sebagai database online untuk analisis data
* serta library python pendukung pemrosesan serta untuk machine learning

### Persiapan

Sumber data: https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee

Setup environment:

1. Buka terminal atau PowerShell.
2. Buat sebuah folder baru bernama submission dengan menjalankan perintah berikut.
```
mkdir submission
```

3. Pindah ke folder terbaru tersebut menggunakan perintah berikut.
```
cd submission
```

4. Buat sebuah virtual environment dengan menjalankan perintah berikut.
```
pipenv install
```

5. Aktifkan virtual environment dengan menjalankan perintah berikut.
```
pipenv shell
```

6. Instal semua library yang dibutuhkan menggunakan perintah berikut.
```
pip install numpy pandas scipy matplotlib seaborn jupyter sqlalchemy scikit-learn joblib streamlit
```

7. Buka jupyter-notebook dengan menjalankan perintah berikut.
```
jupyter-notebook .
```

## Business Dashboard

dashboard dapat diakses pada [link](http://localhost:3000/public/dashboard/a3525560-f079-44d1-be85-49c40e8c95e0)
dashboard ini digunakan untuk analisis data usia dan gender dari karyawan yang menunjukkan keterhungan pada pengunduran diri. berikut rincian keterangan dari dashboard
* dari disitribusi karyawan terlihat lebih banyak karyawan laki-laki daripada perempuan
* sebaran karyawan paling banyak terdapat di umur 35 tahun
* keterhubungan data usia dan gender adalah lebih banyak laki-laki pada setiap gender yang terdata dalam data
* sebaran karyawan yang melakukan pengunduran diri juga lebih sedikit dari pada yang tidak, serta ada yang tidak mau menyebutkan juga lumayan (null)
* untuk gender juga ternyata lebih banyak laki-laki yang melakukan pengunduran diri, walau tak significant
* dan terakhir sebaran usia yang melakukan pengunduran diri.

## Conclusion

dari analisis dan modelling dari pengaruh usia dan jenis kelamin terhadap pengunduran diri karyawan adalah ada sedikit hubungan antara hal tersebut. dari beberapa cluster yang ada, terdapat cluster yang harus diperhatikan, karena memiliki persentase hasil dari perhitungan model terlihat ada kemungkinan karyawan yang berada di cluster itu bisa saja melakukan pengunduran diri

### Rekomendasi Action Items (Optional)

Berikut beberapa rekomendasi action items yang harus dilakukan perusahaan guna menyelesaikan permasalahan atau mencapai target.

- lebih memperhatikan usia pekerja beserta gender mereka, karena bisa sedikit mempengaruhi pengunduran diri pekerja tersebut
- tidak hanya terpaku pada suatu cluster saja yang memiliki hasil berkemungkinan melakukan pengunduran diri, tapi juga memperhatikan cluster lain untuk mengantisipasi pengunduran diri yang bisa saja terjadi.

## Cara predict dengan Model yang dibuat
model yang dipakai 'kmeans_model,joblib'
untuk penggunaan tinggal run code seperti biasa dengan menginputkan parameter
    'EmployeeId',
    'Age',
    'Gender',
    'Attrition'
