import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle

# --- 1. VERİYİ YÜKLEME ---
filename = "oyun_projesi_final_veri.csv"
try:
    df = pd.read_csv(filename)
    print(f"✅ Veri başarıyla yüklendi: {len(df)} satır")
except FileNotFoundError:
    try:
        df = pd.read_csv("oyun_tavsiye_makine_ogrenmesi/" + filename)
        print(f"✅ Veri başarıyla yüklendi (Alt klasörden): {len(df)} satır")
    except FileNotFoundError:
        print("❌ Hata: CSV dosyası bulunamadı!")
        exit()

# --- 2. HEDEF BELİRLEME (HIT PREDICTION) ---
# Medyan değerin üzerindekilere "1" (Hit), altına "0" (Niche) diyelim.
threshold = df['num_reviews_total'].median()
df['is_hit'] = (df['num_reviews_total'] > threshold).astype(int)

print(f"\n🎯 HEDEF: Popülerlik Tahmini (Eşik Değeri: {threshold:.0f} inceleme)")
print(f"   Hit Olanlar (1): {df['is_hit'].sum()}")
print(f"   Niche Olanlar (0): {len(df) - df['is_hit'].sum()}")

# Model Girdileri (Features)
drop_cols = ['final_name', 'header_image', 'cluster_label', 'pca_x', 'pca_y', 
             'num_reviews_total', 'norm_reviews', 'is_hit']
features = [col for col in df.columns if col not in drop_cols and df[col].dtype in [np.float64, np.int64]]

X = df[features]
y = df['is_hit']

# Veriyi Bölme (%70 Eğitim, %30 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# --- 3. KORELASYON ANALİZİ (GÖREV 1) ---
print("\n📊 1. KORELASYON MATRİSİ OLUŞTURULUYOR...")

# Analiz edilecek sayısal sütunlar (Fiyat, Puan, Tür İlişkisi)
corr_cols = ['final_price', 'metacritic_score', 
             'gen_action', 'gen_rpg', 'gen_indie', 'cat_multiplayer', 'is_recent','is_hit']

# Sadece veri setinde mevcut olanları al
existing_cols = [c for c in corr_cols if c in df.columns]
corr_matrix = df[existing_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Oyun Özellikleri Arasındaki Korelasyon")
plt.tight_layout()
plt.savefig("korelasyon_matrisi.png")
print("   ✅ 'korelasyon_matrisi.png' kaydedildi.")




# --- 4. SINIFLANDIRMA VE CONFUSION MATRIX (GÖREV 2 & 3) ---
print("\n🤖 2. SINIFLANDIRMA MODELİ (Decision Tree - Derinlik 8)...")

# Tek bir model eğitiyoruz (Confusion Matrix için)
clf_fixed = DecisionTreeClassifier(max_depth=8, random_state=42)
clf_fixed.fit(X_train, y_train)
y_test_pred = clf_fixed.predict(X_test)

# Confusion Matrix Çizimi
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Niche', 'Hit'], 
            yticklabels=['Niche', 'Hit'])
plt.title("Confusion Matrix (Popülerlik Tahmini)")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Durum")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("   ✅ 'confusion_matrix.png' kaydedildi.")

# Metrik Raporu
print("\n   📄 Sınıflandırma Raporu:")
print(classification_report(y_test, y_test_pred))


# --- 5. DETAYLI OVERFITTING ANALİZİ VE GRAFİĞİ (GÖREV 4) ---
print("\n📈 3. OVERFITTING GRAFİĞİ (Complexity Curve) ÇİZİLİYOR...")

depths = range(1, 21)
train_scores = []
test_scores = []

# Döngü ile her derinliği test et
for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    
    train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
    test_scores.append(accuracy_score(y_test, clf.predict(X_test)))

# Grafiği Çiz
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'bo-', label='Eğitim Başarısı (Train Accuracy)')
plt.plot(depths, test_scores, 'ro-', label='Test Başarısı (Test Accuracy)')

# Optimal noktayı bul ve işaretle
optimal_idx = np.argmax(test_scores)
optimal_depth = depths[optimal_idx]
max_test_score = test_scores[optimal_idx]

plt.axvline(x=optimal_depth, color='green', linestyle='--', label=f'En İyi Derinlik ({optimal_depth})')
plt.title('Overfitting Analizi: Ağaç Derinliği vs Başarı', fontsize=14)
plt.xlabel('Ağaç Derinliği (Max Depth)', fontsize=12)
plt.ylabel('Doğruluk (Accuracy)', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(depths)

plt.tight_layout()
plt.savefig("overfitting_analizi_grafigi.png")
print(f"   ✅ 'overfitting_analizi_grafigi.png' kaydedildi.")
print(f"   👉 En iyi Test Başarısı: %{max_test_score*100:.2f} (Derinlik {optimal_depth})")

with open("hit_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("\n✅ TÜM İŞLEMLER TAMAMLANDI. Rapor için 3 adet PNG dosyası hazır.")