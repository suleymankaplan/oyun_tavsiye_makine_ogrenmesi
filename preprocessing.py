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
    if not isinstance(text, str): return False
    text = text.lower()
    bundle_keywords = ['bundle', ' pack', 'collection', 'edition', 'season pass', 'dlc', 'franchise']
    junk_keywords = ['prologue', 'playtest', 'demo', 'soundtrack', ' artbook', 'server', 'beta', 'test branch']
    all_bad_keywords = bundle_keywords + junk_keywords
    if any(keyword in text for keyword in all_bad_keywords):
        return True
    return False

def advanced_clean_name(text):
    if not isinstance(text, str): return None
    text = text.lower()
    text = re.sub(r'[®™©]', '', text)
    junk_words = [r'standard edition', r'deluxe edition', r'gold edition', r'ultimate edition',
                  r'game of the year edition', r'goty', r'directors cut', r'remastered', r'anniversary edition']
    for junk in junk_words:
        text = re.sub(junk, '', text)
    return re.sub(r'[^a-z0-9]', '', text)


# --- ADIM 1: STEAM ÖN İŞLEME ---

# A. Bundle Temizliği
df_steam = df_steam[~df_steam['name'].apply(is_bundle_or_junk)]

# B. Tarih ve Platform Düzeltme
df_steam['release_year'] = pd.to_datetime(df_steam['release_date'], errors='coerce').dt.year

# Steam'deki True/False sütunlarını 1/0'a çevirelim (Garanti olsun)
for plat in ['windows', 'mac', 'linux']:
    if plat in df_steam.columns:
        df_steam[plat] = df_steam[plat].astype(int)
    else:
        df_steam[plat] = 0 # Eğer sütun yoksa 0 yap

# C. Filtreleme
df_steam = df_steam[df_steam["num_reviews_total"] >= 500]

# D. Tekilleştirme
df_steam['temp_clean'] = df_steam['name'].apply(advanced_clean_name)
df_steam = df_steam.sort_values('num_reviews_total', ascending=False)
df_steam = df_steam.drop_duplicates(subset=['temp_clean'], keep='first')
df_steam = df_steam.drop(columns=['temp_clean'])

# E. Gereksizleri Sil (Platform sütunlarını SİLMİYORUZ!)
cols_to_drop_steam = [
    "reviews", "dlc_count", "release_date",
    "detailed_description", "about_the_game", "short_description",
    "website", "support_url", "support_email", "metacritic_url",
    "achievements", "notes", "packages", "screenshots", "movies","full_audio_languages", 
    "discount","user_score","score_rank","estimated_owners"
]
df_steam = df_steam.drop(columns=[c for c in cols_to_drop_steam if c in df_steam.columns], axis=1)


# --- ADIM 2: EPIC ÖN İŞLEME ---

# A. Bundle Temizliği
df_epic = df_epic[~df_epic['name'].apply(is_bundle_or_junk)]

# B. Fiyat ve Tarih
df_epic['price'] = pd.to_numeric(df_epic['price'], errors='coerce').fillna(0) / 100
df_epic['release_year'] = pd.to_datetime(df_epic['release_date'], errors='coerce').dt.year

# C. PLATFORM AYRIŞTIRMA (YENİ EKLENDİ)
# Epic'te 'platform' sütunu "Windows, Mac OS" gibi metin içeriyor.
# Bunu Steam formatına (windows, mac, linux) çeviriyoruz.
df_epic['platform'] = df_epic['platform'].fillna('Windows') # Boşsa Windows varsay
df_epic['windows_epic'] = df_epic['platform'].astype(str).str.contains('Windows', case=False).astype(int)
df_epic['mac_epic'] = df_epic['platform'].astype(str).str.contains('Mac', case=False).astype(int)
# Epic'te Linux genelde yazmaz ama varsa alalım
df_epic['linux_epic'] = df_epic['platform'].astype(str).str.contains('Linux', case=False).astype(int)

# D. Sütun İsimleri ve Tekilleştirme
df_epic = df_epic.rename(columns={
    'name': 'name_epic', 'price': 'price_epic', 'genres': 'genres_epic',
    'release_year': 'release_year_epic', 'developer': 'developer_epic'
})

df_epic['temp_clean'] = df_epic['name_epic'].apply(advanced_clean_name)
df_epic = df_epic.sort_values('price_epic')
df_epic = df_epic.drop_duplicates(subset=['temp_clean'], keep='first')
df_epic = df_epic.drop(columns=['temp_clean'])

# Sütun seçimi (Platformları ekledik)
epic_keep_cols = ['name_epic', 'price_epic', 'genres_epic', 'release_year_epic', 
                  'developer_epic', 'windows_epic', 'mac_epic', 'linux_epic']
df_epic = df_epic[[c for c in epic_keep_cols if c in df_epic.columns]]


# --- ADIM 3: BİRLEŞTİRME ---

df_steam['merge_key'] = df_steam['name'].apply(advanced_clean_name)
df_epic['merge_key'] = df_epic['name_epic'].apply(advanced_clean_name)

df_final = pd.merge(df_steam, df_epic, on='merge_key', how='outer')


# --- ADIM 4: VERİ DOLDURMA (COALESCE) ---

df_final['final_name'] = df_final['name'].combine_first(df_final['name_epic'])
df_final['final_price'] = df_final['price'].combine_first(df_final['price_epic'])
df_final['genres'] = df_final['genres'].combine_first(df_final['genres_epic'])
df_final['release_year'] = df_final['release_year'].combine_first(df_final['release_year_epic']).fillna(2020)

# PLATFORM BİRLEŞTİRME (YENİ)
# Steam verisi yoksa Epic verisini kullan
df_final['windows'] = df_final['windows'].combine_first(df_final['windows_epic']).fillna(1) # PC oyunu olduğu için 1 varsayılabilir
df_final['mac'] = df_final['mac'].combine_first(df_final['mac_epic']).fillna(0)
df_final['linux'] = df_final['linux'].combine_first(df_final['linux_epic']).fillna(0)

# Platform Kaynağı
df_final['on_steam'] = df_final['name'].notna().astype(int)
df_final['on_epic'] = df_final['name_epic'].notna().astype(int)

# --- ADIM: VR ONLY OYUNLARI SİLME ---

# 1. Kategoriler sütununu string yapalım (Hata almamak için)
df_final['categories'] = df_final['categories'].fillna('').astype(str)

# 2. İçinde 'VR Only' veya 'Tracked Motion Controller Support' geçenleri buluyoruz.
vr_keywords = 'VR Only'

# 3. Tilde (~) işareti ile "Bunları İçermeyenleri" alıyoruz.
df_final = df_final[~df_final['categories'].str.contains(vr_keywords, case=False, na=False)]

# --- ADIM 5: MANUEL OYUN EKLEME (PLATFORMLAR DAHİL) ---

manual_games = [
    {
        'final_name': 'League of Legends', 'final_price': 0.0,
        'genres': 'MOBA, Strategy, RPG', 'release_year': 2009,
        'on_steam': 0, 'on_epic': 1, 'num_reviews_total': 999999,
        'windows': 1, 'mac': 1, 'linux': 0, # LoL Mac'te var
        'header_image': 'https://cdn2.unrealengine.com/league-of-legends-artwork-1920x1080-254070248.jpg'
    },
    {
        'final_name': 'Valorant', 'final_price': 0.0,
        'genres': 'FPS, Tactical, Action', 'release_year': 2020,
        'on_steam': 0, 'on_epic': 1, 'num_reviews_total': 500000,
        'windows': 1, 'mac': 0, 'linux': 0, # Valorant sadece Windows
        'header_image': 'https://cdn2.unrealengine.com/valorant-artwork-1920x1080-1a8971e3b683.jpg'
    },
    {
        'final_name': 'Minecraft', 'final_price': 29.99,
        'genres': 'Sandbox, Survival, Adventure', 'release_year': 2011,
        'on_steam': 0, 'on_epic': 0, 'num_reviews_total': 1000000,
        'windows': 1, 'mac': 1, 'linux': 1, # Minecraft her yerde var
        'header_image': 'https://image.api.playstation.com/vulcan/img/rnd/202010/2119/UtZW37q1Q06a5961Q8k5s583.png'
    }
]
df_final = pd.concat([df_final, pd.DataFrame(manual_games)], ignore_index=True)


# --- ADIM 6: GLOBAL TEMİZLİK ---

def has_non_latin(text):
    if not isinstance(text, str): return False
    return bool(re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u0400-\u04ff]', text))

df_final = df_final[~df_final['final_name'].apply(has_non_latin)]

banned_keywords = ['sexual content', 'nudity', 'hentai', 'mature', 'nsfw', 'adult', 'sex', 'erotic']
df_final['text_search'] = (df_final['final_name'].astype(str) + " " + df_final['genres'].astype(str)).str.lower()
mask_banned = df_final['text_search'].apply(lambda x: any(ban in x for ban in banned_keywords))
df_final = df_final[~mask_banned]

# --- ADIM 1: METİN HAVUZU OLUŞTURMA (GENRES + TAGS) ---
# İkisini string'e çevir, yan yana ekle ve küçük harf yap.
# Örnek Sonuç: "action, indie, rpg {'action': 500, 'story rich': 200}"
df_final['combined_text'] = (df_final['genres'].fillna('') + " " + df_final['tags'].fillna('')).astype(str).str.lower()

# --- ADIM 2: HEDEF TÜRLERİ BELİRLEME VE SÜTUNLAŞTIRMA ---
# Hangi türleri kolon yapmak istiyoruz?
# Buraya hem Ana Türleri (Action) hem de Kritik Etiketleri (Open World) ekliyoruz.

target_genres = {
    # ANA TÜRLER
    'gen_action': ['action', 'shooter', 'fps', 'tps'], # Shooter görürse de Action say
    'gen_adventure': ['adventure', 'exploration'],
    'gen_rpg': ['rpg', 'role-playing', 'role playing'],
    'gen_simulation': ['simulation', 'sim'],
    'gen_strategy': ['strategy', 'rts', 'turn-based', 'tactical'],
    'gen_sports_racing': ['sports', 'racing'], # Genelde az oldukları için birleştirilebilir
    'gen_horror': ['horror', 'survival horror'], # Taglerden gelir
    
    # GÖRSEL / TEKNİK
    'gen_2d': ['2d', 'pixel', 'platformer'],
    'gen_3d': ['3d', 'realistic'],
    'gen_anime': ['anime', 'visual novel', 'jrpg'],
    
    # OYNANIŞ (TAG'lerden gelenler)
    'gen_open_world': ['open world', 'sandbox'],
    'gen_rogue': ['rogue-like', 'rogue-lite', 'roguelike'],
    'gen_scifi': ['sci-fi', 'space', 'cyberpunk', 'futuristic'],
    'gen_survival': ['survival'],
    'gen_puzzle': ['puzzle', 'logic'],
    'gen_arcade': ['arcade', 'casual'],
    'gen_story': ['story rich', 'narrative', 'visual novel']
}

# Döngü ile sütunları oluştur
print("Türler birleştiriliyor ve sütunlaştırılıyor...")


# --- DİL SÜTUNU TEMİZLİĞİ ---

# 1. Önce sütunu string yap ve küçük harfe çevir
# fillna('english') yapıyoruz çünkü Epic oyunlarında bu veri yok.
# Verisi olmayan oyunun en azından İngilizce olduğunu varsaymak güvenlidir.
df_final['supported_languages'] = df_final['supported_languages'].fillna('english').astype(str).str.lower()

# 2. İNGİLİZCE VAR MI?
# "english" kelimesi geçiyorsa 1, yoksa 0
df_final['lang_english'] = df_final['supported_languages'].apply(lambda x: 1 if 'english' in x else 0)

# 3. TÜRKÇE VAR MI?
# Hem "turkish" hem de "türkçe" kelimelerini arayalım (Bazen yerel dilde yazıyorlar)
df_final['lang_turkish'] = df_final['supported_languages'].apply(lambda x: 1 if 'turkish' in x or 'türkçe' in x else 0)

# 4. TEMİZLİK
# Orijinal karmaşık sütunu siliyoruz
df_final = df_final.drop(columns=['supported_languages'])

# --- KONTROL ---
print("✅ Dil ayrıştırma tamamlandı.")
print(df_final[['final_name', 'lang_english', 'lang_turkish']].head())

# Türkçe desteği olan kaç oyun var bakalım?
print(f"\nTürkçe Destekli Oyun Sayısı: {df_final['lang_turkish'].sum()}")

for col_name, keywords in target_genres.items():
    # Lambda: Metin havuzunda bu anahtar kelimelerden HERHANGİ BİRİ var mı?
    df_final[col_name] = df_final['combined_text'].apply(lambda x: 1 if any(k in x for k in keywords) else 0)

# --- ADIM 3: TEMİZLİK ---
# Artık genres, tags ve geçici combined_text sütunlarına ihtiyacımız kalmadı.
# Hepsini tek potada erittik.
cols_to_drop = ['genres', 'tags', 'combined_text']
df_final = df_final.drop(columns=cols_to_drop)

print("✅ İşlem Tamam! Genres ve Tags birleştirildi.")
print(f"Yeni Sütun Sayısı: {len(df_final.columns)}")

# Kontrol
check_cols = ['final_name', 'gen_action', 'gen_rpg', 'gen_horror', 'gen_open_world']
print(df_final[check_cols].head())



# --- SON RÖTUŞLAR ---
cols_to_clean = ['merge_key', 'name', 'name_epic', 'price', 'price_epic', 
                 'genres_epic', 'release_year_epic', 'developer_epic', 
                 'windows_epic', 'mac_epic', 'linux_epic', 'text_search'] # Ara sütunları temizle
df_final = df_final.drop(columns=[c for c in cols_to_clean if c in df_final.columns])

df_final['num_reviews_total'] = df_final['num_reviews_total'].fillna(0)
df_final['metacritic_score'] = df_final['metacritic_score'].fillna(0)
# Platform sayılarını int yap (görüntü kirliliğini önle)
for col in ['windows', 'mac', 'linux']:
    df_final[col] = df_final[col].fillna(0).astype(int)

df_final = df_final.drop_duplicates(subset=['final_name'])

import pandas as pd

# Varsayalım ki df_final yüklü.

# 1. Önce sütunu string yap ve temizle
df_final['categories'] = df_final['categories'].fillna('').astype(str)

# --- İSTENEN KATEGORİLERİ SÜTUNLAŞTIRMA ---

# 1. Single-player
df_final['cat_singleplayer'] = df_final['categories'].apply(lambda x: 1 if 'Single-player' in x else 0)

# 2. Controller (Full, Partial ve Tracked hepsini birleştir)
# Mantık: İçinde 'Controller' kelimesi geçen her şeyi 1 yap.
df_final['cat_controller'] = df_final['categories'].str.contains('Controller', case=False).astype(int)

# 3. Multiplayer (Sadece kelime bazlı)
df_final['cat_multiplayer'] = df_final['categories'].apply(lambda x: 1 if 'Multi-player' in x else 0)

# 4. Co-op (Genel)
# Not: 'Online Co-op' da içinde 'Co-op' barındırdığı için burası 1 olacaktır.
df_final['cat_coop'] = df_final['categories'].apply(lambda x: 1 if 'Co-op' in x else 0)

# 5. Online Co-op (Özel)
df_final['cat_online_coop'] = df_final['categories'].apply(lambda x: 1 if 'Online Co-op' in x else 0)

# 6. PvP
df_final['cat_pvp'] = df_final['categories'].apply(lambda x: 1 if 'PvP' in x else 0)

# 7. Shared/Split Screen
df_final['cat_split_screen'] = df_final['categories'].apply(lambda x: 1 if 'Shared/Split Screen' in x else 0)

# 8. MMO (Listendeki 'memo' -> Verideki 'MMO')
df_final['cat_mmo'] = df_final['categories'].apply(lambda x: 1 if 'MMO' in x else 0)


# --- TEMİZLİK ---
# İşini bitirdiğimiz orijinal 'categories' sütununu siliyoruz
df_final = df_final.drop(columns=['categories'])

# KONTROL
print("✅ Kategori ayrıştırma tamamlandı.")
print(df_final[['final_name', 'cat_singleplayer', 'cat_multiplayer', 'cat_controller', 'cat_mmo']].head())

# Gereksiz verileri manuel silme işlemi
df_final = df_final[df_final['final_name'] != 'VALORANT']
df_final = df_final[df_final['final_name'] != 'OHDcore Mod Kit']
df_final = df_final[df_final['final_name'] != 'iHeart: Radio, Music, Podcast']
df_final = df_final[df_final['final_name'] != 'Discord']
df_final = df_final[df_final['final_name'] != 'DCL The Game - Track Editor']
df_final = df_final[df_final['final_name'] != 'Brave']
df_final = df_final[df_final['final_name'] != 'Bus Simulator 21 - Modding Kit']
df_final = df_final[df_final['final_name'] != 'Opera GX - The First Browser for Gamers']
df_final = df_final[df_final['final_name'] != 'Itch.io']
df_final = df_final[df_final['final_name'] != 'Itch.io']
df_final = df_final[df_final['final_name'] != 'Itch.io']
df_final = df_final[df_final['final_name'] != 'Itch.io']
df_final = df_final[df_final['final_name'] != 'Itch.io']
df_final = df_final[df_final['final_name'] != 'Itch.io']
df_final = df_final[df_final['final_name'] != 'Itch.io']
df_final = df_final[df_final['final_name'] != 'Itch.io']
df_final = df_final[df_final['final_name'] != 'Itch.io']


print(f"İşlem Tamam! Final Veri Sayısı: {len(df_final)}")
# Kontrol: Sadece Epic'ten gelen (Steam'de olmayan) bir oyunun platform bilgisine bakalım
print("\nSadece Epic'ten gelen örnek oyun (Platform Kontrolü):")
only_epic = df_final[(df_final['on_epic']==1) & (df_final['on_steam']==0)].head(1)
print(only_epic[['final_name', 'windows', 'mac']])

df_final.to_csv("oyun_projesi_final_veri.csv", index=False)