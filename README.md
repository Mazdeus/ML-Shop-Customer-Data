# ML-Shop-Customer-Data

## Overview
This repository contains the implementation of clustering and classification algorithms on a customer dataset. The goal is to segment customers into distinct groups and classify them based on their characteristics. The dataset used in this project is the "Shop Customer Data," which includes various features such as age, annual income, spending score, profession, work experience, and family size.

## Project Structure 📁
The project is organized into the following directories and files:

## Dataset 📊
The dataset consists of the following features:
- **CustomerID**: Unique ID for each customer.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Annual Income ($)**: Annual income of the customer.
- **Spending Score (1-100)**: Spending score based on customer behavior.
- **Profession**: Profession of the customer.
- **Work Experience**: Work experience of the customer in years.
- **Family Size**: Family size of the customer.

## Clustering 🔍
The clustering process involves segmenting customers into distinct groups based on their characteristics. The following steps are performed in the clustering process:
1. **Data Preprocessing**: Handling missing values, removing duplicates, normalizing features, and encoding categorical data.
2. **Feature Selection**: Using techniques like Variance Threshold and PCA to select important features.
3. **Model Training**: Training the KMeans clustering model with different numbers of clusters.
4. **Evaluation**: Evaluating the model using Silhouette Score and visualizing the results.

The clustering implementation can be found in the [clustering.py](src/clustering.py) script.

## Classification 🧩
The classification process involves predicting the cluster of a customer based on their characteristics. The following steps are performed in the classification process:
1. **Data Splitting**: Splitting the dataset into training and testing sets.
2. **Model Training**: Training classification models like Random Forest and K-Nearest Neighbors (KNN).
3. **Evaluation**: Evaluating the models using metrics like accuracy, precision, recall, and F1-score. Visualizing the results using confusion matrices.

The classification implementation can be found in the [klasifikasi.py](src/klasifikasi.py) script.

## Notebooks 📓
The project includes Jupyter notebooks that provide detailed explanations and visualizations of the clustering and classification processes:
- [Clustering Notebook](notebooks/[Clustering]_Submission_Akhir_BMLP_Mohammad_Amadeus_Andika_Fadhil (1).ipynb)
- [Classification Notebook](notebooks/[Klasifikasi]_Submission_Akhir_BMLP_Mohammad_Amadeus_Andika_Fadhil.ipynb)

## Results
The results of the clustering and classification processes are saved in the `hasil_clustering.csv` file. The notebooks provide detailed visualizations and analysis of the results.

Conclusion 🏁
This project demonstrates the application of clustering and classification algorithms on a customer dataset. The results provide valuable insights into customer segmentation and classification, which can be used for targeted marketing and personalized customer experiences.