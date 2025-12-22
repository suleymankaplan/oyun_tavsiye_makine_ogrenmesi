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
# %25'lik dilimi (Top 25%) Hit kabul ediyoruz
threshold = df['num_reviews_total'].quantile(0.75)
df['is_hit'] = (df['num_reviews_total'] > threshold).astype(int)

print(f"\n🎯 HEDEF: Popülerlik Tahmini (Eşik Değeri: {threshold:.0f} inceleme)")
print(f"   Orijinal Hit Sayısı: {df['is_hit'].sum()}")
print(f"   Orijinal Niche Sayısı: {len(df) - df['is_hit'].sum()}")


# --- 3. VERİ DENGELEME (UNDERSAMPLING) ---
# İsteğin üzerine: Hit sayısı kadar Niche seçip durumu %50-%50 eşitliyoruz.
df_hit = df[df['is_hit'] == 1]
df_niche = df[df['is_hit'] == 0]

# Niche olanlardan rastgele Hit sayısı kadar al
df_niche_balanced = df_niche.sample(n=len(df_hit), random_state=42)

# İkisini birleştir (Artık elimizde dengeli bir veri seti var)
df_balanced = pd.concat([df_hit, df_niche_balanced])

print(f"\n⚖️  VERİ DENGELENDİ (UNDERSAMPLING):")
print(f"   Yeni Veri Seti Boyutu: {len(df_balanced)}")
print(f"   Hit Sayısı: {df_balanced['is_hit'].sum()}")
print(f"   Niche Sayısı: {len(df_balanced) - df_balanced['is_hit'].sum()}")
print("   (Model artık %50 Hit - %50 Niche verisiyle eğitilecek.)")


# --- 4. KORELASYON ANALİZİ ---
# Dengeli veri seti üzerinden korelasyona bakmak daha mantıklıdır.
print("\n📊 1. KORELASYON MATRİSİ OLUŞTURULUYOR...")

corr_cols = ['is_hit', 'final_price', 'metacritic_score', 
             'gen_action', 'gen_rpg', 'gen_indie', 'cat_multiplayer', 'is_recent']

# Sadece veri setinde mevcut olanları al
existing_cols = [c for c in corr_cols if c in df_balanced.columns]
corr_matrix = df_balanced[existing_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Oyun Özellikleri Arasındaki Korelasyon (Dengeli Veri)")
plt.tight_layout()
plt.savefig("korelasyon_matrisi.png")
print("   ✅ 'korelasyon_matrisi.png' kaydedildi.")


# --- 5. MODEL HAZIRLIĞI ---
# Modelin kopya çekmesini (Data Leakage) engellemek için sızıntı yapan sütunları atıyoruz.
drop_cols = [
    'final_name', 'header_image', 'cluster_label', 'pca_x', 'pca_y', 
    'is_hit',             # Hedef
    'num_reviews_total',  # Hedefin kendisi
    'norm_reviews',       
    'num_reviews_recent', # ⚠️ KOPYA: Son incelemeler
    'estimated_owners'    # ⚠️ KOPYA: Sahip sayısı
]
features = [col for col in df_balanced.columns if col not in drop_cols and df_balanced[col].dtype in [np.float64, np.int64]]

X = df_balanced[features]
y = df_balanced['is_hit']

# Veriyi Bölme (%70 Eğitim, %30 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# --- 6. SINIFLANDIRMA VE CONFUSION MATRIX ---
print("\n🤖 2. SINIFLANDIRMA MODELİ (Decision Tree - Derinlik 8)...")

# Sabit derinlikte bir model eğitelim
clf_fixed = DecisionTreeClassifier(max_depth=6, random_state=42)
clf_fixed.fit(X_train, y_train)
y_test_pred = clf_fixed.predict(X_test)

# Confusion Matrix Çizimi
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Niche', 'Hit'], 
            yticklabels=['Niche', 'Hit'])
plt.title("Confusion Matrix (Dengeli 50/50)")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Durum")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("   ✅ 'confusion_matrix.png' kaydedildi.")

# Metrik Raporu
print("\n   📄 Sınıflandırma Raporu:")
print(classification_report(y_test, y_test_pred))


# --- 7. OVERFITTING ANALİZİ ---
print("\n📈 3. OVERFITTING GRAFİĞİ (Complexity Curve) ÇİZİLİYOR...")

depths = range(1, 21)
train_scores = []
test_scores = []

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
plt.title('Overfitting Analizi: Ağaç Derinliği vs Başarı (Dengeli Veri)', fontsize=14)
plt.xlabel('Ağaç Derinliği (Max Depth)', fontsize=12)
plt.ylabel('Doğruluk (Accuracy)', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(depths)
# Dengeli olduğu için %50'den başlaması beklenir
plt.ylim(0.9, 1.0) 

plt.tight_layout()
plt.savefig("overfitting_analizi_grafigi.png")
print(f"   ✅ 'overfitting_analizi_grafigi.png' kaydedildi.")
print(f"   👉 En iyi Test Başarısı: %{max_test_score*100:.2f} (Derinlik {optimal_depth})")


# --- 8. MODELİ KAYDETME ---
print("\n💾 4. MODEL KAYDEDİLİYOR...")

# DİKKAT: Döngüdeki son model (depth=20) yerine, 
# Overfitting yapmayan ideal derinlikteki (örn: 8) modeli tüm dengeli veriyle eğitip kaydediyoruz.
final_clf = DecisionTreeClassifier(max_depth=6, random_state=42)
final_clf.fit(X, y)

with open("hit_model.pkl", "wb") as f:
    pickle.dump(final_clf, f)

print(f"✅ 'hit_model.pkl' başarıyla kaydedildi.")