import pandas as pd
import os

# Dosya yolu
DATA_PATH = "oyun_projesi_clustered.csv"

def main():
    if not os.path.exists(DATA_PATH):
        print("❌ Hata: CSV dosyası bulunamadı.")
        return

    print("⏳ Veri yükleniyor...")
    df = pd.read_csv(DATA_PATH)
    print("✅ Hazır! (Çıkmak için 'q' yazın)\n")

    while True:
        game_name = input("🔎 Oyun Adı Girin: ").strip()
        
        if game_name.lower() == 'q':
            break

        # İsmi küçük harfe çevirip arayalım (Case insensitive)
        row = df[df['final_name'].str.lower() == game_name.lower()]

        if row.empty:
            print("❌ Oyun bulunamadı. Tam ismini yazdığınızdan emin olun.")
            continue
        
        # İlk eşleşen satırı al
        data = row.iloc[0]
        
        print(f"\n🎯 OYUN: {data['final_name']}")
        print(f"💰 Fiyat: {data['final_price']}")
        print(f"📊 İnceleme Sayısı: {data['num_reviews_total']}")
        print(f"📈 Normalize Puan (0-1): {data.get('norm_reviews', 'Yok')}")
        print("-" * 40)
        print("AKTİF ÖZELLİKLER (1 OLANLAR):")
        
        # Tüm sütunları gez, 1 olan feature'ları yazdır
        found_features = False
        for col in df.columns:
            # Sadece bizim teknik sütunlara bakıyoruz
            if col.startswith(('gen_', 'cat_', 'is_', 'dev_', 'lang_')):
                if data[col] == 1:
                    print(f"  ✅ {col}")
                    found_features = True
        
        if not found_features:
            print("  ⚠️ Hiçbir özellik işaretli değil (Hepsi 0).")
            
        print("-" * 40 + "\n")

if __name__ == "__main__":
    main()