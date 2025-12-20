import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
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

# --- DÜZELTME: EKSİK VERİLERİ DOLDURMA ---
X = X.fillna(X.mean())
print(f"   🛠️ Eksik veriler (NaN) ortalama değerlerle dolduruldu.")

y = df['is_hit']

# Veriyi Bölme (%70 Eğitim, %30 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# --- 3. KORELASYON ANALİZİ (GÖREV 1) ---
print("\n📊 1. KORELASYON MATRİSİ OLUŞTURULUYOR...")

# Analiz edilecek sayısal sütunlar
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


# --- 4. MODEL KARŞILAŞTIRMA (GÖREV 2: EN AZ 3 MODEL) ---
print("\n🤖 2. MODELLER EĞİTİLİYOR VE KARŞILAŞTIRILIYOR...")

# Kullanılacak Modeller
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB()
}

results = {}
best_model_name = ""
best_score = 0
best_model_obj = None

# Modelleri döngü ile eğitip test edelim
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print(f"   👉 {name} Başarısı: %{acc*100:.2f}")
        
        # En iyiyi kaydet
        if acc > best_score:
            best_score = acc
            best_model_name = name
            best_model_obj = model
            
    except Exception as e:
        print(f"   ❌ {name} çalışırken hata oluştu: {e}")

# Karşılaştırma Grafiği
if results:
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylim(0, 1.05)
    plt.ylabel('Doğruluk (Accuracy)')
    plt.title('Sınıflandırma Modellerinin Başarı Karşılaştırması')
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.02, f"%{v*100:.1f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig("model_karsilastirma.png")
    print("   ✅ 'model_karsilastirma.png' kaydedildi.")


# --- 5. EN İYİ MODELİN DETAYLARI VE KAYDEDİLMESİ ---
if best_model_obj:
    print(f"\n🏆 KAZANAN MODEL: {best_model_name} (Skor: %{best_score*100:.2f})")

    y_pred_best = best_model_obj.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Niche', 'Hit'], 
                yticklabels=['Niche', 'Hit'])
    plt.title(f"Confusion Matrix ({best_model_name})")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek Durum")
    plt.tight_layout()
    plt.savefig("confusion_matrix_best.png")
    print(f"   ✅ 'confusion_matrix_best.png' kaydedildi ({best_model_name} için).")

    # Modeli Kaydet
    with open("hit_model.pkl", "wb") as f:
        pickle.dump(best_model_obj, f)
    print(f"   💾 '{best_model_name}' başarıyla 'hit_model.pkl' dosyasına kaydedildi.")


# --- 6. OVERFITTING ANALİZİ (Sadece Decision Tree İçin) ---
print("\n📈 3. OVERFITTING GRAFİĞİ (Decision Tree Derinlik Analizi)...")

depths = range(1, 16)
train_scores = []
test_scores = []

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_tr, y_tr)
    train_scores.append(accuracy_score(y_tr, clf.predict(X_tr)))
    test_scores.append(accuracy_score(y_te, clf.predict(X_te)))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'bo-', label='Eğitim Başarısı')
plt.plot(depths, test_scores, 'ro-', label='Test Başarısı')
plt.title('Overfitting Analizi: Ağaç Derinliği vs Başarı')
plt.xlabel('Derinlik')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True)
plt.savefig("overfitting_analizi_grafigi.png")
print(f"   ✅ 'overfitting_analizi_grafigi.png' kaydedildi.")

print("\n✅ TÜM İŞLEMLER TAMAMLANDI.")