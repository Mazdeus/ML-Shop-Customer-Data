"""
1. PERKENALAN DATASET

Pada submisiion kali ini saya menggunakan dataset tanpa label "Shop Customer Data". Dataset ini merupakan sumber dataset yang disarankan
pada Instruksi Submission -> Lainnya. Dataset ini berisi informasi pelanggan toko sebagai berikut:
- CustomerID: ID unik untuk setiap pelanggan.
- Gender: Jenis kelamin pelanggan.
- Age: Usia pelanggan.
- Annual Income ($): Pendapatan tahunan pelanggan.
- Spending Score (1-100): Skor pengeluaran pelanggan berdasarkan perilaku belanja.
- Profession: Pekerjaan pelanggan.
- Work Experience: Pengalaman kerja pelanggan dalam tahun.
- Family Size: Ukuran keluarga pelanggan.

Dataset ini memiliki 2001 baris dan mencakup data numerik (misal: Age, Annual Income) dan kategorikal (misal: Gender, Profession).

"""

"""
2. IMPORT LIBRARY
Pada tahap ini, Anda perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

"""
3. Memuat Data
"""

# Memuat Dataset
df = pd.read_csv(r"Data\Customers.csv")

# Melihat beberapa baris pertama untuk memastikan data telah dimuat dengan benar
df.head()

"""
4. Exploratory Data Analysis (EDA)
Pada tahap ini, Anda akan melakukan Exploratory Data Analysis (EDA) untuk memahami karakteristik dataset. EDA bertujuan untuk:

a. Memahami Struktur Data
Tinjau jumlah baris dan kolom dalam dataset.
Tinjau jenis data di setiap kolom (numerikal atau kategorikal).

b. Menangani Data yang Hilang
Identifikasi dan analisis data yang hilang (missing values). Tentukan langkah-langkah yang diperlukan untuk menangani data yang hilang, seperti pengisian atau penghapusan data tersebut.

c. Analisis Distribusi dan Korelasi
Analisis distribusi variabel numerik dengan statistik deskriptif dan visualisasi seperti histogram atau boxplot.
Periksa hubungan antara variabel menggunakan matriks korelasi atau scatter plot.

d. Visualisasi Data
Buat visualisasi dasar seperti grafik distribusi dan diagram batang untuk variabel kategorikal.
Gunakan heatmap atau pairplot untuk menganalisis korelasi antar variabel.

Tujuan dari EDA adalah untuk memperoleh wawasan awal yang mendalam mengenai data dan menentukan langkah selanjutnya dalam analisis atau pemodelan.
"""

# Menampilkan informasi tentang dataset, termasuk jumlah baris, kolom, tipe data, dan jumlah nilai non-null
df.info()

# Mengecek missing values jika ada
df.isnull().sum()

# Menampilkan statistik deskriptif dari dataset untuk kolo numerik
df.describe()

# Visualisasi distribusi variabel numerik
num_features = df.select_dtypes(include=[np.number])
plt.figure(figsize=(14, 10))
for i, column in enumerate(num_features.columns, 1):
    plt.subplot(3,4,i)
    sns.histplot(df[column], bins=30, kde=True, color='blue')
    plt.title(f'Distribusi {column}')
plt.tight_layout()
plt.show()

# Distribusi fitur kategorikal
cat_features = df.select_dtypes(include=[object])
plt.figure(figsize=(14, 8))
for i, column in enumerate(cat_features.columns, 1):
    plt.subplot(2, 4, i)
    # sns.countplot(y=data[column], palette='viridis')
    sns.countplot(y=df[column], palette='viridis', hue=df[column], legend=False)
    plt.title(f'Distribusi {column}')
plt.tight_layout()
plt.show()

# Heatmap korelasi untuk fitur numerik
plt.figure(figsize=(12, 10))
correlation_matrix = num_features.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap Korelasi')
plt.show()

# Pairplot untuk fitur numerik
sns.pairplot(num_features)
plt.show()

"""
5. Data Preprocessing
Pada tahap ini, data preprocessing adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model machine learning. Data mentah sering kali mengandung nilai kosong, duplikasi, atau rentang nilai yang tidak konsisten, yang dapat memengaruhi kinerja model. Oleh karena itu, proses ini bertujuan untuk membersihkan dan mempersiapkan data agar analisis berjalan optimal.

Berikut adalah tahapan-tahapan yang bisa dilakukan, tetapi tidak terbatas pada:
- Menghapus atau Menangani Data Kosong (Missing Values)
- Menghapus Data Duplikat
- Normalisasi atau Standarisasi Fitur
- Deteksi dan Penanganan Outlier
- Encoding Data Kategorikal
- Binning (Pengelompokan Data)
"""

# Mengisi missing values pada kolom 'Profession' dengan modus (nilai yang paling sering muncul)
df['Profession'] = df['Profession'].fillna(df['Profession'].mode()[0])

# Cek Kembali
df.isnull().sum()