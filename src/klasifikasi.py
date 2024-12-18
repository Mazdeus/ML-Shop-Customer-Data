#Impoert Library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Memuat Dataset
df = pd.read_csv(r"Data\hasil_clustering.csv")

# Melihat beberapa baris pertama untuk memastikan data telah dimuat dengan benar
df.head()

# Data Splitting

# Menentukan fitur (X) dan target (y)
X = df.drop(columns=['Cluster'])  # Fitur: Semua kolom kecuali kolom target 'Cluster'
y = df['Cluster']                # Target: Kolom Cluster

# Mengecek data X dan y
print("Fitur (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())

# Memisahkan data menjadi Training Set dan Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Menampilkan jumlah data pada Training Set dan Test Set
print("\nJumlah data dalam Training dan Test Set:")
print(f"Training Set: {X_train.shape[0]} sampel")
print(f"Test Set: {X_test.shape[0]} sampel")

# Mengecek distribusi target pada Training dan Test Set
print("\nDistribusi target pada Training Set:")
print(y_train.value_counts())
print("\nDistribusi target pada Test Set:")
print(y_test.value_counts())

# Inisialisasi Model
# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)

# Melatih Model dengan Data Latih
print("Melatih model Random Forest...")
rf_model.fit(X_train, y_train)

print("Melatih model K-Nearest Neighbors...")
knn_model.fit(X_train, y_train)

# Prediksi pada Data Uji
print("\nMelakukan prediksi pada data uji...")
y_pred_rf = rf_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

# Evaluasi Model
# Evaluasi Random Forest
print("\n===== Evaluasi Model: Random Forest =====")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_rf, average='weighted'):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_rf, average='weighted'):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf, average='weighted'):.2f}")
print("\nClassification Report Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Visualisasi Confusion Matrix Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Evaluasi KNN
print("\n===== Evaluasi Model: K-Nearest Neighbors (KNN) =====")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_knn, average='weighted'):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_knn, average='weighted'):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred_knn, average='weighted'):.2f}")
print("\nClassification Report KNN:")
print(classification_report(y_test, y_pred_knn))

# Visualisasi Confusion Matrix KNN
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_knn, annot=True, fmt="d", cmap="Greens", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

