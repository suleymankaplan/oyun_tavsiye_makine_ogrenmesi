import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
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

# --- 2. HEDEF BELİRLEME VE VERİ DENGELEME ---
threshold = df['num_reviews_total'].quantile(0.75)
df_hits = df[df['num_reviews_total'] > threshold].copy()
df_niche_candidates = df[df['num_reviews_total'] <= threshold].copy()

df_hits['is_hit'] = 1
df_niche_candidates['is_hit'] = 0

print(f"\n🎯 HEDEF: Popülerlik Tahmini (Eşik: {threshold:.0f} inceleme)")

# Dengeleme
if len(df_niche_candidates) >= len(df_hits):
    df_niche_balanced = df_niche_candidates.sample(n=len(df_hits), random_state=42)
else:
    df_niche_balanced = df_niche_candidates

df = pd.concat([df_hits, df_niche_balanced])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n⚖️ VERİ SETİ DENGELENDİ (50/50)")
print(f"   Analiz Edilecek Toplam Veri: {len(df)}")

# Model Girdileri
drop_cols = ['final_name', 'header_image', 'cluster_label', 'pca_x', 'pca_y', 
             'num_reviews_total', 'norm_reviews', 'is_hit']
features = [col for col in df.columns if col not in drop_cols and df[col].dtype in [np.float64, np.int64]]

X = df[features]
X = X.fillna(X.mean())
y = df['is_hit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 2.1 VERİ DENGESİ GRAFİĞİ ---
plt.figure(figsize=(6, 5))
ax = sns.countplot(x=df['is_hit'], palette='viridis')
plt.title('Dengelenmiş Veri Dağılımı')
plt.xticks([0, 1], ['Niche (0)', 'Hit (1)'])
for container in ax.containers:
    ax.bar_label(container, fmt='%d', padding=3, fontweight='bold')
plt.tight_layout()
plt.savefig("veri_dengesi_grafigi.png")
print("   ✅ 'veri_dengesi_grafigi.png' kaydedildi.")


# --- 3. KORELASYON GRAFİĞİ ---
print("\n📊 1. KORELASYON GRAFİĞİ OLUŞTURULUYOR...")
corr_cols = ['final_price', 'metacritic_score', 
             'gen_action', 'gen_rpg', 'gen_indie', 'cat_multiplayer', 'is_recent','is_hit']
existing_cols = [c for c in corr_cols if c in df.columns]
corr_matrix = df[existing_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Özellik Korelasyonları")
plt.tight_layout()
plt.savefig("korelasyon_grafigi.png")
print("   ✅ 'korelasyon_grafigi.png' kaydedildi.")


# --- 4. MODEL ANALİZİ (GRAFİKLERLE) ---
print("\n🤖 2. MODELLERİN GRAFİKSEL ANALİZİ...")

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,max_depth=10),
    "Naive Bayes": GaussianNB()
}

performance_summary = []
best_model_name = ""
best_test_f1 = 0
best_model_obj = None

for name, model in models.items():
    print(f"\n   👉 {name} Grafikleri Hazırlanıyor...")
    safe_name = name.replace(" ", "_")
    
    # 1. Eğitim
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 2. Metrikler
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred)
    
    # --- 4.1 SINIFLANDIRMA RAPORU GRAFİĞİ (HEATMAP) ---
    report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    
    df_plot = df_report.loc[['0', '1'], ['precision', 'recall', 'f1-score']]
    df_plot.index = ['Niche', 'Hit']
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_plot, annot=True, cmap='RdYlGn', fmt='.2f', vmin=0, vmax=1)
    plt.title(f"Detaylı Performans Analizi - {name}")
    plt.tight_layout()
    plt.savefig(f"performans_grafigi_{safe_name}.png")
    plt.close()
    print(f"      ✅ 'performans_grafigi_{safe_name}.png' kaydedildi.")

    # --- 4.2 CONFUSION MATRIX GRAFİĞİ ---
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Niche', 'Hit'], yticklabels=['Niche', 'Hit'])
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{safe_name}.png")
    plt.close()
    
    performance_summary.append({
        'Model': name,
        'Train Acc': acc_train,
        'Test Acc': acc_test
    })
    
    if f1_test > best_test_f1:
        best_test_f1 = f1_test
        best_model_name = name
        best_model_obj = model

# --- 5. KARŞILAŞTIRMA GRAFİĞİ ---
print("\n📊 3. GENEL KARŞILAŞTIRMA GRAFİĞİ...")
df_summary = pd.DataFrame(performance_summary)
df_melted = df_summary.melt(id_vars="Model", value_vars=["Train Acc", "Test Acc"], 
                            var_name="Veri Seti", value_name="Başarı")

plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x="Model", y="Başarı", hue="Veri Seti", palette="viridis")
plt.ylim(0, 1.1)
plt.title("Model Başarı Karşılaştırması (Train vs Test)")
plt.ylabel("Accuracy Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("basari_karsilastirma_grafigi.png")
print("   ✅ 'basari_karsilastirma_grafigi.png' kaydedildi.")


# --- 6. MODEL KAYDETME ---
if best_model_obj:
    print(f"\n🏆 KAZANAN MODEL: {best_model_name} (F1: %{best_test_f1*100:.2f})")
    with open("hit_model.pkl", "wb") as f:
        pickle.dump(best_model_obj, f)
    print(f"   💾 Model 'hit_model.pkl' olarak kaydedildi.")


# --- 7. DERİNLİK ANALİZİ GRAFİĞİ (Decision Tree) ---
print("\n📈 4. DECISION TREE ANALİZİ (Derinlik vs Başarı)...")
depths = range(1, 16)
tr_sc, te_sc = [], []
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_tr, y_tr)
    tr_sc.append(accuracy_score(y_tr, clf.predict(X_tr)))
    te_sc.append(accuracy_score(y_te, clf.predict(X_te)))

plt.figure(figsize=(10, 6))
plt.plot(depths, tr_sc, 'bo-', label='Eğitim (Train)')
plt.plot(depths, te_sc, 'ro-', label='Test')
plt.title('Overfitting Analizi: Decision Tree (Derinlik)')
plt.xlabel('Ağaç Derinliği')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("dt_derinlik_grafigi.png")
print(f"   ✅ 'dt_derinlik_grafigi.png' kaydedildi.")


# --- 8. YENİ: RANDOM FOREST N_ESTIMATORS ANALİZİ ---
print("\n🌲 5. RANDOM FOREST ANALİZİ (Ağaç Sayısı vs Başarı)...")
# Ağaç sayılarını belirle (10'dan 200'e kadar)
estimator_range = [10, 30, 50, 80, 100, 150, 200]
tr_sc_rf, te_sc_rf = [], []

for n in estimator_range:
    # random_state sabit ki karşılaştırma adil olsun
    rf = RandomForestClassifier(n_estimators=n, random_state=42,max_depth=10) 
    rf.fit(X_tr, y_tr)
    
    tr_sc_rf.append(accuracy_score(y_tr, rf.predict(X_tr)))
    te_sc_rf.append(accuracy_score(y_te, rf.predict(X_te)))

plt.figure(figsize=(10, 6))
plt.plot(estimator_range, tr_sc_rf, 'bo-', label='Eğitim (Train)')
plt.plot(estimator_range, te_sc_rf, 'ro-', label='Test')
plt.title('Overfitting Analizi: Random Forest (n_estimators)')
plt.xlabel('Ağaç Sayısı (n_estimators)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("rf_estimator_grafigi.png")
print(f"   ✅ 'rf_estimator_grafigi.png' kaydedildi.")

print("\n✅ TÜM İŞLEMLER TAMAMLANDI.")