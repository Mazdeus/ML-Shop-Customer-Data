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
num_features = df.select_dtypes(include=[np.number]).columns

plt.figure(figsize=(14, 10))
for i, column in enumerate(num_features, 1):
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
correlation_matrix = df[num_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap Korelasi')
plt.show()

# Pairplot untuk fitur numerik
sns.pairplot(df[num_features])
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

"""
Missing Values
"""
# Mengisi missing values pada kolom 'Profession' dengan modus (nilai yang paling sering muncul)
df['Profession'] = df['Profession'].fillna(df['Profession'].mode()[0])

# Cek Kembali
# df.isnull().sum()

"""
Data duplikat
"""
# Mengidentifikasi baris duplikat
duplicates = df.duplicated()

# Menghapus Data Duplikat
df.drop_duplicates(inplace=True)
print(f"Number of duplicates after removal: {df.duplicated().sum()}")

"""
Standarisasi Fitur Numerik
"""
# Histogram sebelum standarisasi
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# sns.histplot(df['Annual Income ($)'], kde=True)
# plt.title("Histogram Sebelum Standarisasi")
# plt.show()

# Normalisasi atau Standarisasi Fitur
num_features = df.select_dtypes(include=np.number).columns
scaler = MinMaxScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# Histogram setelah standarisasi
plt.subplot(1,2,2)
sns.histplot(df['Annual Income ($)'], kde=True)
plt.title("Histogram Setelah Standarisasi")
plt.show()

"""
Outliners
"""
# Deteksi dan Penanganan Outlier (using IQR method)
for features in num_features:
    Q1 = df[features].quantile(0.25)
    Q3 = df[features].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with bounds
    df[features] = np.where(df[features] < lower_bound, lower_bound, df[features])
    df[features] = np.where(df[features] > upper_bound, upper_bound, df[features])

"""
Encoding data kategorikal
"""
# Encoding Data Kategorikal
cat_features = df.select_dtypes(include='object').columns
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Menghapus kolom CustomerID, tidak diperlukan untuk analisis
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

print(df.head())
print(df.info())

"""
6. Pembangunan Model Clustering
"""

# Inisialisasi model KMeans tanpa parameter awal
kmeans = KMeans()

# Inisialisasi visualizer Kelbow untuk menentukan jumlah cluster optimal
visualizer = KElbowVisualizer(kmeans, k=(1,10))

# Fit visualizer dgn data untuk menemukan jumlah cluster optimal
visualizer.fit(df)

# Menampilkan grafik elbow untuk analisis
visualizer.show()

# Inisialisasi dan melatih model Kmearns dengan jumlah cluster = 2
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(df)

# Mendapatkan label cluster
labels = kmeans.labels_

# Menambahkan hasil cluster ke dataframe
df['Cluster'] = labels

# Menampilkan hasil model clustering
print(df.head())

# Mendapatkan jumlah cluster
k = 2

# Menghitung Silhouette Score untuk berbagai jumlah cluster
sil_scores = []
k_range = range(2, 11)  # Coba jumlah cluster dari 2 sampai 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df)
    labels = kmeans.labels_

    # Menghitung Silhouette Score untuk cluster tersebut
    sil_score = silhouette_score(df, labels)
    sil_scores.append(sil_score)
    print(f"Jumlah Cluster: {k}, Silhouette Score: {sil_score:.2f}")

# Visualisasi Silhouette Scores untuk jumlah cluster yang berbeda
plt.plot(k_range, sil_scores, marker='o')
plt.title('Evaluasi Silhouette Score untuk Berbagai Jumlah Cluster')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Silhouette Score')
plt.show()

# Memilih jumlah cluster dengan Silhouette Score tertinggi
optimal_k = k_range[sil_scores.index(max(sil_scores))]
print(f"Jumlah Cluster Optimal berdasarkan Silhouette Score: {optimal_k}")

# Update num_features to exclude 'CustomerID'
num_features = df.select_dtypes(include=np.number).columns.drop('CustomerID', errors='ignore')  # Handle potential KeyError

# Feature Selection - Variance Threshold
# Menggunakan VarianceThreshold untuk memilih fitur yang memiliki variansi lebih tinggi dari threshold tertentu
# Misalnya, kita hanya akan mempertahankan fitur yang memiliki variansi lebih tinggi dari 0.1
selector = VarianceThreshold(threshold=0.1)
df_selected_variance = df[df.columns[selector.fit(df).get_support()]]

# Feature Selection - PCA (Principal Component Analysis)
# Mengurangi dimensi data dengan PCA, misalnya kita memilih 3 komponen utama
pca = PCA(n_components=3)
df_selected_pca = pca.fit_transform(df[num_features])

# Melatih Model Clustering Sebelum Feature Selection
kmeans_before = KMeans(n_clusters=3, random_state=0)
kmeans_before.fit(df[num_features])
labels_before = kmeans_before.labels_

# Menghitung Silhouette Score sebelum feature selection
sil_score_before = silhouette_score(df[num_features], labels_before)
print(f"Silhouette Score sebelum Feature Selection: {sil_score_before:.2f}")

# Melatih Model Clustering Setelah Variance Threshold Feature Selection
kmeans_variance = KMeans(n_clusters=3, random_state=0)
kmeans_variance.fit(df_selected_variance)
labels_variance = kmeans_variance.labels_

# Menghitung Silhouette Score setelah Variance Threshold Feature Selection
sil_score_variance = silhouette_score(df_selected_variance, labels_variance)
print(f"Silhouette Score setelah Variance Threshold Feature Selection: {sil_score_variance:.2f}")

# 6. Melatih Model Clustering Setelah PCA Feature Selection
kmeans_pca = KMeans(n_clusters=3, random_state=0)
kmeans_pca.fit(df_selected_pca)
labels_pca = kmeans_pca.labels_

# Menghitung Silhouette Score setelah PCA Feature Selection
sil_score_pca = silhouette_score(df_selected_pca, labels_pca)
print(f"Silhouette Score setelah PCA Feature Selection: {sil_score_pca:.2f}")

# Visualisasi Hasil Clustering
# Menggunakan PCA untuk mereduksi dimensi data menjadi 2D
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[num_features])  # Menggunakan fitur numerik yang sudah distandarisasi

# Menambahkan hasil PCA ke dalam dataframe untuk visualisasi
df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
df_pca['Cluster'] = df['Cluster']  # Menambahkan label cluster ke data PCA

# Plotting hasil clustering dalam 2D menggunakan scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100, edgecolor='black', alpha=0.7)
plt.title('Visualisasi Hasil Clustering (2D PCA Projection)', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Cluster')
plt.show()

# Analisis Karakteristik Tiap Cluster
print("Jumlah data pada setiap cluster:")
print(df['Cluster'].value_counts())  # Menampilkan jumlah data per cluster

# Deskripsi Statistik untuk Tiap Cluster
print("\nDeskripsi statistik tiap cluster:")
for cluster in sorted(df['Cluster'].unique()):
    print(f"\nCluster {cluster}:")
    cluster_data = df[df['Cluster'] == cluster]
    print(cluster_data.describe())

# Visualisasi Distribusi Fitur Numerik untuk Tiap Cluster
num_features = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(15, 8))
for i, feature in enumerate(num_features, 1):
    plt.subplot(2, len(num_features)//2, i)
    for cluster in sorted(df['Cluster'].unique()):
        sns.kdeplot(df[df['Cluster'] == cluster][feature], label=f'Cluster {cluster}', fill=True, alpha=0.4)
    plt.title(f'Distribusi {feature} per Cluster')
    plt.legend()
plt.tight_layout()
plt.show()

# Visualisasi Fitur Numerik Menggunakan Boxplot
plt.figure(figsize=(15, 8))
for i, feature in enumerate(num_features, 1):
    plt.subplot(2, len(num_features)//2, i)
    sns.boxplot(x='Cluster', y=feature, data=df)
    plt.title(f'Boxplot {feature} per Cluster')
plt.tight_layout()
plt.show()

# Mengeksport Data Hasil Clustering

# Menyimpan hasil clustering ke dalam file CSV
output_file = "hasil_clustering.csv"  # Nama file output
df.to_csv(output_file, index=False)

print(f"Data hasil clustering berhasil disimpan ke dalam file: {output_file}")
