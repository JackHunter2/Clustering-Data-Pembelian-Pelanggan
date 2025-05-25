from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import uuid
import glob

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f'Gagal menghapus {f}: {e}')
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'Tidak ada file yang diunggah'
    file = request.files['file']
    if file.filename == '':
        return 'File belum dipilih'

    unique_id = str(uuid.uuid4())
    filename = os.path.join(UPLOAD_FOLDER, unique_id + '_' + file.filename)
    file.save(filename)

    try:
        df = pd.read_csv(filename, encoding='ISO-8859-1')
    except Exception as e:
        return f'Gagal membaca file: {e}'

    expected_cols = {"InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"}
    if not expected_cols.issubset(df.columns):
        return f"Dataset harus memiliki kolom: {', '.join(expected_cols)}"

    df = df.dropna(subset=['CustomerID'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    customer_df = df.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'TotalPrice': 'sum'
    }).rename(columns={
        'InvoiceNo': 'TotalTransaksi',
        'Quantity': 'TotalItem',
        'TotalPrice': 'TotalBelanja'
    }).reset_index()

    features = ['TotalTransaksi', 'TotalItem', 'TotalBelanja']
    X = customer_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    original_table = customer_df.to_html(classes='table table-bordered', index=False)

    inertia = []
    max_k = min(10, len(customer_df))
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    elbow_path = os.path.join('static', 'uploads', f'elbow_{unique_id}.png')
    plt.figure()
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.xlabel('Jumlah Klaster (k)')
    plt.ylabel('WCSS')
    plt.title('Metode Elbow')
    plt.savefig(elbow_path)
    plt.close()

    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    silhouette_path = os.path.join('static', 'uploads', f'silhouette_{unique_id}.png')
    plt.figure()
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    plt.xlabel('Jumlah Klaster (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Metode Silhouette Score')
    plt.savefig(silhouette_path)
    plt.close()

    final_k = 3 if len(customer_df) >= 3 else len(customer_df)
    kmeans = KMeans(n_clusters=final_k, random_state=42)
    customer_df['Cluster'] = kmeans.fit_predict(X_scaled)

    cluster_path = os.path.join('static', 'uploads', f'cluster_{unique_id}.png')
    plt.figure()
    sns.scatterplot(x=customer_df['TotalItem'], y=customer_df['TotalBelanja'], hue=customer_df['Cluster'], palette='viridis')
    plt.title('Segmentasi Pelanggan')
    plt.xlabel('Total Item')
    plt.ylabel('Total Belanja')
    plt.savefig(cluster_path)
    plt.close()

    result_table = customer_df.to_html(classes='table table-striped', index=False)

    return render_template('result.html',
                           original=original_table,
                           result=result_table,
                           elbow='/' + elbow_path,
                           silhouette='/' + silhouette_path,
                           cluster='/' + cluster_path)

if __name__ == '__main__':
    app.run(debug=True)
