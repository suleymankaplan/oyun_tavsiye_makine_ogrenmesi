import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

# 1. VERİYİ YÜKLE
try:
    df = pd.read_csv("oyun_projesi_final_veri.csv")
    print(f"✅ Veri yüklendi. Toplam Oyun: {len(df)}")
except FileNotFoundError:
    print("❌ Hata: Dosya bulunamadı.")
    exit()

# --- ADIM 2: ÖZELLİK SEÇİMİ (X MATRİSİ) ---
features = [col for col in df.columns if col.startswith(('gen_', 'cat_', 'is_', 'dev_'))]

if 'lang_turkish' in df.columns: features.append('lang_turkish')
if 'norm_reviews' in df.columns: features.append('norm_reviews')

X = df[features]
print(f"🤖 Model {len(features)} özellik ile eğitilecek.")


# --- TEKNİK 1: K-MEANS CLUSTERING (Ana Gruplama) ---
K = 40 
print(f"\n🚀 TEKNİK 1: K-Means (K={K}) çalıştırılıyor...")
kmeans = KMeans(n_clusters=K, init='k-means++', random_state=42, n_init=10)
kmeans.fit(X)
df['cluster_label'] = kmeans.labels_


# --- TEKNİK 2: K-NEAREST NEIGHBORS (Hassas Tavsiye) ---
print("\n🚀 TEKNİK 2: k-NN (Nearest Neighbors) eğitiliyor...")
knn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
knn_model.fit(X)


# --- TEKNİK 3: PCA (Boyut İndirgeme ve Görselleştirme) ---
print("\n🚀 TEKNİK 3: PCA (Feature Extraction) uygulanıyor...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
df['pca_x'] = pca_result[:, 0]
df['pca_y'] = pca_result[:, 1]


# --- TEKNİK 4: VALIDASYON (Silhouette Score) ---
print("\n📊 ANALİZ: Silhouette Skoru hesaplanıyor (Kümeleme Başarısı)...")
sample_X = X.sample(n=min(2000, len(X)), random_state=42)
sample_labels = kmeans.predict(sample_X)
score = silhouette_score(sample_X, sample_labels)
print(f"   👉 Silhouette Score: {score:.4f}")
print("   (Not: 1'e ne kadar yakınsa kümeler o kadar net ayrışmış demektir.)")


# --- TEKNİK 5: HİYERARŞİK KÜMELEME (Dendrogram - Rapor İçin) ---
print("\n📊 ANALİZ: Hiyerarşik Kümeleme Dendrogramı oluşturuluyor (Rapor Görseli)...")
top_50_indices = df.nlargest(50, 'num_reviews_total').index
X_subset = X.loc[top_50_indices]
names_subset = df.loc[top_50_indices, 'final_name'].values

plt.figure(figsize=(10, 7))
plt.title("Oyunların Hiyerarşik İlişkisi (Top 50)")
dend = shc.dendrogram(shc.linkage(X_subset, method='ward'), labels=names_subset, leaf_rotation=90)
plt.tight_layout()
plt.savefig("dendrogram.png")
print("   👉 'dendrogram.png' kaydedildi.")


# --- KAYDETME ---
with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn_model, f)

df.to_csv("oyun_projesi_clustered.csv", index=False)

print("\n💾 KAYIT BAŞARILI:")
print("   1. kmeans_model.pkl (Gruplama Modeli)")
print("   2. knn_model.pkl (Benzerlik/Tavsiye Modeli)")
print("   3. oyun_projesi_clustered.csv (İşlenmiş Veri)")