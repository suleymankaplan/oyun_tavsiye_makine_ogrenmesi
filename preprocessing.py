import pandas as pd
import numpy as np
import re

# 1. VERİLERİ YÜKLE
try:
    data_steam = pd.read_csv("./data/steam_games.csv")
    data_epic = pd.read_csv("./data/epic_games.csv")
except FileNotFoundError:
    print("Hata: Dosyalar bulunamadı.")
    exit()

df_steam = pd.DataFrame(data_steam)
df_epic = pd.DataFrame(data_epic)

print(f"Ham Veri - Steam: {len(df_steam)}, Epic: {len(df_epic)}")

# --- YARDIMCI FONKSİYONLAR ---

def is_bundle_or_junk(text):
    """
    Hem Paketleri (Bundle) hem de Yan Ürünleri (Prologue, Playtest) tespit eder.
    Burası senin bahsettiğin 'Anonymous Hacker Simulator: Prologue' sorununu çözer.
    """
    if not isinstance(text, str): return False
    text = text.lower()
    
    # 1. Paket Anahtar Kelimeleri
    bundle_keywords = ['bundle', ' pack', 'collection', 'edition', 'season pass', 'dlc', 'franchise']
    
    # 2. "Çöp" (Yan Ürün) Anahtar Kelimeleri (Prologue, Playtest vb.)
    junk_keywords = [
        'prologue', 'playtest', 'demo', 'soundtrack', ' artbook', 
        'server', 'beta', 'test branch', 'public test'
    ]
    
    # Birleştirilmiş liste
    all_bad_keywords = bundle_keywords + junk_keywords
    
    # "Standard Edition" hariç diğerlerini silmek için kontrol
    # Ama içinde 'prologue' geçiyorsa direkt sil
    if any(keyword in text for keyword in all_bad_keywords):
        # İstisna: Bazen oyunun adı "Packet" olabilir, onu korumak zor ama 
        # genel filtre için bu yeterli.
        return True
    return False

def advanced_clean_name(text):
    """İsimleri standartlaştırır."""
    if not isinstance(text, str): return None
    text = text.lower()
    text = re.sub(r'[®™©]', '', text)
    junk_words = [
        r'standard edition', r'deluxe edition', r'gold edition', r'ultimate edition',
        r'game of the year edition', r'goty', r'directors cut', r'remastered', 
        r'anniversary edition', r'complete edition'
    ]
    for junk in junk_words:
        text = re.sub(junk, '', text)
    return re.sub(r'[^a-z0-9]', '', text)


# --- ADIM 1: STEAM ÖN İŞLEME VE TEMİZLİK ---

# A. Bundle ve Prologue/Playtest Temizliği (YENİ EKLENDİ)
# İsimlerinde yasaklı kelime geçenleri atıyoruz
df_steam = df_steam[~df_steam['name'].apply(is_bundle_or_junk)]

# B. Tarihi Kurtar
df_steam['release_year'] = pd.to_datetime(df_steam['release_date'], errors='coerce').dt.year

# C. İnceleme Sayısı Filtresi
df_steam = df_steam[df_steam["num_reviews_total"] >= 500]

# D. Akıllı Tekilleştirme (Deduplication) - (YENİ)
# Aynı isme sahip (Clean name) birden fazla oyun varsa, en çok incelemesi olanı tut.
# Örn: "Hacker Sim" (1000 yorum) vs "Hacker Sim Deluxe" (50 yorum) -> Deluxe silinir.
df_steam['temp_clean'] = df_steam['name'].apply(advanced_clean_name)
df_steam = df_steam.sort_values('num_reviews_total', ascending=False) # En popüler en üste
df_steam = df_steam.drop_duplicates(subset=['temp_clean'], keep='first') # İlkini (popüleri) tut
df_steam = df_steam.drop(columns=['temp_clean'])

# E. Gereksiz Sütunları Sil
cols_to_drop_steam = [
    "reviews", "dlc_count", "release_date",
    "detailed_description", "about_the_game", "short_description",
    "website", "support_url", "support_email", "metacritic_url",
    "achievements", "notes", "packages", "screenshots", "movies"
]
df_steam = df_steam.drop(columns=[c for c in cols_to_drop_steam if c in df_steam.columns], axis=1)


# --- ADIM 2: EPIC ÖN İŞLEME ---

# A. Bundle Temizliği (Epic için de uyguluyoruz)
df_epic = df_epic[~df_epic['name'].apply(is_bundle_or_junk)]

# B. Fiyat ve Tarih
df_epic['price'] = pd.to_numeric(df_epic['price'], errors='coerce').fillna(0) / 100
df_epic['release_year'] = pd.to_datetime(df_epic['release_date'], errors='coerce').dt.year

# C. Sütun İsimleri
df_epic = df_epic.rename(columns={
    'name': 'name_epic', 'price': 'price_epic', 'genres': 'genres_epic',
    'release_year': 'release_year_epic', 'developer': 'developer_epic'
})

# D. Tekilleştirme (Epic'te inceleme sayısı yoksa fiyata göre yapalım)
df_epic['temp_clean'] = df_epic['name_epic'].apply(advanced_clean_name)
df_epic = df_epic.sort_values('price_epic') # En ucuz (muhtemelen ana oyun) üste
df_epic = df_epic.drop_duplicates(subset=['temp_clean'], keep='first')
df_epic = df_epic.drop(columns=['temp_clean'])

# Sadece gerekli sütunlar
epic_keep_cols = ['name_epic', 'price_epic', 'genres_epic', 'release_year_epic', 'developer_epic']
df_epic = df_epic[[c for c in epic_keep_cols if c in df_epic.columns]]


# --- ADIM 3: BİRLEŞTİRME (MERGE) ---

df_steam['merge_key'] = df_steam['name'].apply(advanced_clean_name)
df_epic['merge_key'] = df_epic['name_epic'].apply(advanced_clean_name)

# Outer Join
df_final = pd.merge(df_steam, df_epic, on='merge_key', how='outer')


# --- ADIM 4: VERİ DOLDURMA (COALESCE) ---

df_final['final_name'] = df_final['name'].combine_first(df_final['name_epic'])
df_final['final_price'] = df_final['price'].combine_first(df_final['price_epic'])
df_final['genres'] = df_final['genres'].combine_first(df_final['genres_epic'])
df_final['release_year'] = df_final['release_year'].combine_first(df_final['release_year_epic'])
df_final['release_year'] = df_final['release_year'].fillna(df_final['release_year'].median())

# Platform Bilgisi
df_final['on_steam'] = df_final['name'].notna().astype(int)
df_final['on_epic'] = df_final['name_epic'].notna().astype(int)


# --- ADIM 5: MANUEL OYUN EKLEME ---

manual_games = [
    {
        'final_name': 'League of Legends', 'final_price': 0.0,
        'genres': 'MOBA, Strategy, RPG', 'release_year': 2009,
        'on_steam': 0, 'on_epic': 1, 'num_reviews_total': 999999,
        'header_image': 'https://cdn2.unrealengine.com/league-of-legends-artwork-1920x1080-254070248.jpg'
    },
    {
        'final_name': 'Valorant', 'final_price': 0.0,
        'genres': 'FPS, Tactical, Action', 'release_year': 2020,
        'on_steam': 0, 'on_epic': 1, 'num_reviews_total': 500000,
        'header_image': 'https://cdn2.unrealengine.com/valorant-artwork-1920x1080-1a8971e3b683.jpg'
    },
    {
        'final_name': 'Minecraft', 'final_price': 29.99,
        'genres': 'Sandbox, Survival, Adventure', 'release_year': 2011,
        'on_steam': 0, 'on_epic': 0, 'num_reviews_total': 1000000,
        'header_image': 'https://image.api.playstation.com/vulcan/img/rnd/202010/2119/UtZW37q1Q06a5961Q8k5s583.png'
    }
]
df_final = pd.concat([df_final, pd.DataFrame(manual_games)], ignore_index=True)


# --- ADIM 6: GLOBAL TEMİZLİK (SANSÜR & DİL) ---

# A. Asian & Cyrillic Filter
def has_non_latin(text):
    if not isinstance(text, str): return False
    return bool(re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u0400-\u04ff]', text))

df_final = df_final[~df_final['final_name'].apply(has_non_latin)]

# B. NSFW Filter
banned_keywords = ['sexual content', 'nudity', 'hentai', 'mature', 'nsfw', 'adult', 'sex', 'erotic']
df_final['text_search'] = (df_final['final_name'].astype(str) + " " + df_final['genres'].astype(str)).str.lower()
mask_banned = df_final['text_search'].apply(lambda x: any(ban in x for ban in banned_keywords))
df_final = df_final[~mask_banned]


# --- SON RÖTUŞLAR ---
cols_to_clean = ['merge_key', 'name', 'name_epic', 'price', 'price_epic', 
                 'genres_epic', 'release_year_epic', 'developer_epic', 'text_search']
df_final = df_final.drop(columns=[c for c in cols_to_clean if c in df_final.columns])

df_final['num_reviews_total'] = df_final['num_reviews_total'].fillna(0)
df_final['metacritic_score'] = df_final['metacritic_score'].fillna(0)

# Son bir duplicate kontrolü (Manuel ekleme çakışmasın diye)
df_final = df_final.drop_duplicates(subset=['final_name'])

print(f"İşlem Tamam! Final Veri Sayısı: {len(df_final)}")
df_final.to_csv("oyun_projesi_final_veri.csv", index=False)
print("✅ Kayıt Başarılı.")