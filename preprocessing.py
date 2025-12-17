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
    junk_keywords = ['prologue', 'playtest', 'demo', 'soundtrack', ' artbook', 'server', 'beta', 'test branch','modkit']
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

for plat in ['windows', 'mac', 'linux']:
    if plat in df_steam.columns:
        df_steam[plat] = df_steam[plat].astype(int)
    else:
        df_steam[plat] = 0 

# C. Filtreleme
df_steam = df_steam[df_steam["num_reviews_total"] >= 500]

# D. Tekilleştirme
df_steam['temp_clean'] = df_steam['name'].apply(advanced_clean_name)
df_steam = df_steam.sort_values('num_reviews_total', ascending=False)
df_steam = df_steam.drop_duplicates(subset=['temp_clean'], keep='first')
df_steam = df_steam.drop(columns=['temp_clean'])

# E. Gereksizleri Sil 
# NOT: 'pct_pos_total' (inceleme puanı) silinmemeli, analiz için çok önemli!
cols_to_drop_steam = [
    "reviews", "dlc_count", "release_date",
    "detailed_description", "about_the_game", "short_description",
    "website", "support_url", "support_email", "metacritic_url",
    "achievements", "notes", "packages", "screenshots", "movies","full_audio_languages", 
    "discount","user_score","score_rank","required_age","peak_ccu"
]
df_steam = df_steam.drop(columns=[c for c in cols_to_drop_steam if c in df_steam.columns], axis=1)


# --- ADIM 2: EPIC ÖN İŞLEME ---

df_epic = df_epic[~df_epic['name'].apply(is_bundle_or_junk)]

df_epic['price'] = pd.to_numeric(df_epic['price'], errors='coerce').fillna(0) / 100
df_epic['release_year'] = pd.to_datetime(df_epic['release_date'], errors='coerce').dt.year

df_epic['platform'] = df_epic['platform'].fillna('Windows')
df_epic['windows_epic'] = df_epic['platform'].astype(str).str.contains('Windows', case=False).astype(int)
df_epic['mac_epic'] = df_epic['platform'].astype(str).str.contains('Mac', case=False).astype(int)
df_epic['linux_epic'] = df_epic['platform'].astype(str).str.contains('Linux', case=False).astype(int)

df_epic = df_epic.rename(columns={
    'name': 'name_epic', 'price': 'price_epic', 'genres': 'genres_epic',
    'release_year': 'release_year_epic', 'developer': 'developer_epic'
})

df_epic['temp_clean'] = df_epic['name_epic'].apply(advanced_clean_name)
df_epic = df_epic.sort_values('price_epic')
df_epic = df_epic.drop_duplicates(subset=['temp_clean'], keep='first')
df_epic = df_epic.drop(columns=['temp_clean'])

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

# Puan Doldurma (Kritik): Epic oyunlarında puan yoksa ortalama (70) veya steam ortalamasını verelim
if 'pct_pos_total' in df_final.columns:
    avg_score = df_final['pct_pos_total'].mean()
    df_final['pct_pos_total'] = df_final['pct_pos_total'].fillna(avg_score)
else:
    df_final['pct_pos_total'] = 70 # Sütun yoksa varsayılan oluştur

df_final['windows'] = df_final['windows'].combine_first(df_final['windows_epic']).fillna(1)
df_final['mac'] = df_final['mac'].combine_first(df_final['mac_epic']).fillna(0)
df_final['linux'] = df_final['linux'].combine_first(df_final['linux_epic']).fillna(0)

df_final['on_steam'] = df_final['name'].notna().astype(int)
df_final['on_epic'] = df_final['name_epic'].notna().astype(int)


# --- ADIM 5: VR TEMİZLİĞİ ---
# Categories işlemi en son yapılmalı, burada sadece filtreliyoruz
df_final['categories'] = df_final['categories'].fillna('').astype(str)
vr_keywords = 'VR Only'
df_final = df_final[~df_final['categories'].str.contains(vr_keywords, case=False, na=False)]


# --- ADIM 6: MANUEL OYUN EKLEME ---

manual_games = [
{
        'final_name': 'Minecraft', 'final_price': 29.99,
        'genres': 'Sandbox, Survival, Adventure', 
        # Minecraft için kritik özellikler eklendi:
        'categories': 'Single-player, Multi-player, Co-op, Online Co-op, Shared/Split Screen, Full controller support',
        'release_year': 2011,
        'on_steam': 0, 'on_epic': 0, 'num_reviews_total': 1000000, 'pct_pos_total': 95,
        'windows': 1, 'mac': 1, 'linux': 1,
        'header_image': 'https://image.api.playstation.com/vulcan/img/rnd/202010/2119/UtZW37q1Q06a5961Q8k5s583.png'
    }
]
df_final = pd.concat([df_final, pd.DataFrame(manual_games)], ignore_index=True)


# --- ADIM 7: GLOBAL TEMİZLİK (SANSÜR & DİL) ---

def has_non_latin(text):
    if not isinstance(text, str): return False
    return bool(re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u0400-\u04ff]', text))

df_final = df_final[~df_final['final_name'].apply(has_non_latin)]

banned_keywords = ['sexual content', 'nudity', 'hentai', 'mature', 'nsfw', 'adult', 'sex', 'erotic']
df_final['text_search'] = (df_final['final_name'].astype(str) + " " + df_final['genres'].astype(str)).str.lower()
mask_banned = df_final['text_search'].apply(lambda x: any(ban in x for ban in banned_keywords))
df_final = df_final[~mask_banned]


# --- ADIM 8: GENRE + TAG BİRLEŞTİRME (One-Hot) ---

df_final['combined_text'] = (df_final['genres'].fillna('') + " " + df_final['tags'].fillna('')).astype(str).str.lower()

target_genres = {
    'gen_action': ['action', 'shooter','action rpg','character action game','action roguelike','action-adventure'],
    'gen_adventure': ['adventure', 'exploration','action-adventure'],
    'gen_rpg': ['rpg', 'role-playing', 'role playing','jrpg','tactical rpg','crpg','action rpg','choose your own adventure'],
    'gen_simulation': ['simulation', 'life sim','automobile sim','colony sim','space sim','walking simulator','job simulator','medical sim','farming sim'],
    'gen_strategy': ['strategy', 'rts', 'turn-based strategy', 'tactical','grand strategy',],
    'gen_sports_racing': ['sports', 'racing'],
    'gen_horror': ['horror', 'survival horror','psychological horror',],
    'gen_2d': ['2d', '2d platformer','2d fighter'],
    'gen_3d': ['3d', '3d platformer','realistic'],
    'gen_anime': ['anime', 'visual novel', 'jrpg'],
    'gen_open_world': ['open world','open world survival craft'],
    'gen_sandbox':['sandbox'],
    'gen_rogue': ['rogue-like', 'rogue-lite', 'roguelike','Action Roguelike'],
    'gen_scifi': ['sci-fi', 'space', 'cyberpunk', 'futuristic', 'supernatural','spaceships'],
    'gen_survival': ['survival','open world survival craft','survival horror'],
    'gen_indie': ['indie'],
    'gen_puzzle': ['puzzle', 'logic','puzzle-platformer'],
    'gen_arcade': ['arcade', 'casual'],
    'gen_story': ['story rich','narrative', 'visual novel'],
    'gen_fps':['fps','shooter','arena shooter','boomer shooter','first-person']
    
}

for col_name, keywords in target_genres.items():
    df_final[col_name] = df_final['combined_text'].apply(lambda x: 1 if any(k in x for k in keywords) else 0)

cols_to_drop = ['genres', 'tags', 'combined_text']
df_final = df_final.drop(columns=cols_to_drop)


# --- ADIM 9: DİL DESTEĞİ ---

df_final['supported_languages'] = df_final['supported_languages'].fillna('english').astype(str).str.lower()
df_final['lang_english'] = df_final['supported_languages'].apply(lambda x: 1 if 'english' in x else 0)
df_final['lang_turkish'] = df_final['supported_languages'].apply(lambda x: 1 if 'turkish' in x or 'türkçe' in x else 0)
df_final = df_final.drop(columns=['supported_languages'])


# --- ADIM 10: KATEGORİ AYRIŞTIRMA (SON AŞAMA) ---

# DÜZELTME BURADA: Manuel eklenen oyunlardan gelen NaN (float) değerleri temizliyoruz.
# Bu satır olmadan kod LoL veya Valorant satırına gelince hata verir.
df_final['categories'] = df_final['categories'].fillna('').astype(str)

df_final['cat_singleplayer'] = df_final['categories'].apply(lambda x: 1 if 'Single-player' in x else 0)
df_final['cat_controller'] = df_final['categories'].str.contains('Controller', case=False).astype(int)
df_final['cat_multiplayer'] = df_final['categories'].apply(lambda x: 1 if 'Multi-player' in x else 0)
df_final['cat_coop'] = df_final['categories'].apply(lambda x: 1 if 'Co-op' in x else 0)
df_final['cat_online_coop'] = df_final['categories'].apply(lambda x: 1 if 'Online Co-op' in x else 0)
df_final['cat_pvp'] = df_final['categories'].apply(lambda x: 1 if 'PvP' in x else 0)
df_final['cat_split_screen'] = df_final['categories'].apply(lambda x: 1 if 'Shared/Split Screen' in x else 0)
df_final['cat_mmo'] = df_final['categories'].apply(lambda x: 1 if 'MMO' in x else 0)

df_final = df_final.drop(columns=['categories'])

# --- ADIM 11: YILI DÖNEMLERE AYIRMA (BINNING) ---

# 1. is_retro: 2010 ve öncesi
df_final['is_retro'] = df_final['release_year'].apply(lambda x: 1 if x <= 2010 else 0)

# 2. is_mid_era: 2011 ile 2019 arası
df_final['is_mid_era'] = df_final['release_year'].apply(lambda x: 1 if 2010 < x < 2020 else 0)

# 3. is_recent: 2020 ve sonrası
df_final['is_recent'] = df_final['release_year'].apply(lambda x: 1 if x >= 2020 else 0)

# ÖNEMLİ: Orijinal 'release_year' sütununu SİLMİYORUZ!
# Çünkü kullanıcıya sonucu gösterirken "Bu oyun 2015 yapımı" diye göstermemiz gerekecek.
# Ama modeli eğitirken (scaling aşamasında) 'release_year'ı kullanmayıp bu 3 yeni sütunu kullanacağız.


# --- ADIM 12: ÖNEMLİ GELİŞTİRİCİLERİ MODEL İÇİN İŞARETLEME ---

# Geliştirici ve Yayıncı sütunlarını birleştirip arama yapacağız
# (Bazen Rockstar hem yapımcı hem yayıncıdır, ikisine de bakmak lazım)
df_final['dev_pub_combined'] = (df_final['developers'].fillna('') + " " + df_final['publishers'].fillna('')).astype(str).str.lower()

# Oyun dünyasında tarzı en belirgin olan devleri seçelim:
target_devs = {
    'dev_rockstar': ['rockstar games', 'rockstar north'],
    'dev_ubisoft': ['ubisoft', 'ubisoft montreal'],
    'dev_valve': ['valve'], # Half-Life, Portal, L4D hissi başkadır
    'dev_bethesda': ['bethesda', 'bethesda softworks'], # Skyrim, Fallout tarzı
    'dev_ea': ['electronic arts', 'ea sports', 'dice', 'bioware'], # FIFA, BF, Mass Effect
    'dev_square_enix': ['square enix'], # JRPG ve Final Fantasy tarzı
    'dev_capcom': ['capcom'], # Resident Evil, DMC, Street Fighter
    'dev_fromsoftware': ['fromsoftware'], # Souls oyunları (Çok kritik)
    'dev_cdprojekt': ['cd projekt red'], # Witcher, Cyberpunk
    'dev_sony': ['playstation', 'sony interactive', 'naughty dog', 'santa monica'] # God of War, Uncharted kalitesi
}

print("Önemli Geliştiriciler İşaretleniyor...")

for col_name, keywords in target_devs.items():
    df_final[col_name] = df_final['dev_pub_combined'].apply(lambda x: 1 if any(k in x for k in keywords) else 0)

# Geçici sütunu siliyoruz
df_final = df_final.drop(columns=['dev_pub_combined'])

# Kontrol
print(df_final[['final_name', 'dev_rockstar', 'dev_valve', 'dev_ubisoft']].head())

# Rockstar oyunlarına bakalım çalışmış mı?
print("\nRockstar Oyunları:")
print(df_final[df_final['dev_rockstar'] == 1]['final_name'].head())


# # --- ADIM 13: EPIC GAMES TEMİZLİĞİ (ÇÖP VERİLERİ SİLME) ---

# 1. KORUNACAK VIP OYUNLAR (Whitelist)
# Buraya senin manuel eklediklerini ve silinmesini istemediğin büyük oyunları yaz.
vip_epic_games = [
    'Crysis Remastered',
    'Crysis 2 Remastered',
    'Dying Light 2 Stay Human',
    'HITMAN 3',
    'League of Legends',
    'Legends of Runeterra',
    'LEGO® Batman™: The Videogame',
    "Marvel's Guardians of the Galaxy",
    'NBA 2K21',
    'VALORANT', 
    'Splinter Cell Chaos Theory',
    "Tom Clancy's Splinter Cell",
    'Minecraft', 
    'Fortnite', 
    'Rocket League', 
    'Genshin Impact', 
    'Alan Wake 2',
    'Fall Guys',
    'Control'
]

# 2. SİLME MANTIĞI
# Koşul: (Sadece Epic'te olsun) VE (Steam'de olmasın) VE (İsmi VIP listede OLMASIN)
mask_trash_epic = (df_final['on_epic'] == 1) & \
                  (df_final['on_steam'] == 0) & \
                  (~df_final['final_name'].isin(vip_epic_games))

# Silmeden önce kaç tane gidecek görelim
print(f"Silinecek Niteliksiz Epic Oyunu Sayısı: {mask_trash_epic.sum()}")

# Temizlik
df_final = df_final[~mask_trash_epic]

print(f"Temizlik Sonrası Toplam Oyun: {len(df_final)}")

# KONTROL: VIP oyunlar duruyor mu?
print("\nVIP Oyun Kontrolü:")
print(df_final[df_final['final_name'].isin(vip_epic_games)]['final_name'].unique())


# --- ADIM: EPIC GAMES ÖZEL OYUNLARINI MANUEL DOLDURMA ---
# Bu blok, sadece Epic'te olan ama verileri eksik gelen dev oyunları tamir eder.

print("🛠️ Epic Games Özel Oyunları Manuel Olarak Dolduruluyor...")

# Oyunların özellik haritası
# 'reviews': Tahmini inceleme sayısı (Steam standartlarına göre popülerlik)
# 'traits': İşaretlenecek sütunlar (1 yapılacaklar)
epic_manual_fix = {
    "League of Legends": {
        "reviews": 15000000, 
        # LoL 3D'dir (İzometrik), Klavye/Mouse oynanır (Controller yok)
        "traits": ["gen_rpg", "gen_strategy", "gen_3d", "cat_multiplayer", "cat_mmo", "cat_coop", "cat_pvp", "is_recent", "dev_riot"]
    },
    "VALORANT": {
        "reviews": 10000000, 
        # Valorant 3D FPS'tir. PC'de resmi controller desteği yoktur (Rekabetçi yapı gereği).
        "traits": ["gen_action", "gen_3d", "cat_multiplayer", "cat_pvp", "cat_coop", "is_recent", "dev_riot","gen_fps"]
    },
    "Fortnite": {
        "reviews": 12000000, 
        # Fortnite tam controller desteği sunar.
        "traits": ["gen_action", "gen_survival", "gen_3d", "cat_controller", "cat_multiplayer", "cat_coop", "cat_pvp", "gen_open_world", "is_recent","gen_sandbox"]
    },
    "Genshin Impact": {
        "reviews": 8000000, 
        "traits": ["gen_rpg", "gen_open_world", "gen_anime", "gen_adventure", "gen_3d", "cat_controller", "cat_singleplayer", "cat_coop", "is_recent"]
    },
    "Minecraft": {
        "reviews": 20000000,
        "traits": ["gen_adventure", "gen_simulation", "gen_open_world", "gen_survival", "gen_3d", "cat_controller", "cat_singleplayer", "cat_multiplayer", "cat_coop", "is_mid_era","gen_sandbox","gen_fps"]
    },
    "Tom Clancy's Splinter Cell": {
        "reviews": 20000, 
        "traits": ["gen_action", "gen_adventure", "gen_3d", "cat_controller", "cat_singleplayer", "is_retro", "dev_ubisoft"]
    },
    "Splinter Cell Chaos Theory": {
        "reviews": 15000, 
        "traits": ["gen_action", "gen_adventure", "gen_3d", "cat_controller", "cat_singleplayer", "cat_coop", "is_retro", "dev_ubisoft"]
    },
    "Marvel's Guardians of the Galaxy": {
        "reviews": 30000, 
        "traits": ["gen_action", "gen_adventure", "gen_3d", "cat_controller", "cat_singleplayer", "gen_scifi", "gen_story", "is_recent", "dev_square_enix"]
    },
    "LEGO® Batman™: The Videogame": {
        "reviews": 10000, 
        "traits": ["gen_action", "gen_adventure", "gen_3d", "cat_controller", "cat_singleplayer", "cat_coop", "cat_split_screen", "is_retro"]
    },
    "HITMAN 3": {
        "reviews": 40000, 
        "traits": ["gen_action", "gen_adventure", "gen_3d", "cat_controller", "cat_singleplayer", "gen_puzzle", "is_recent"]
    },
    "Dying Light 2 Stay Human": {
        "reviews": 120000, 
        "traits": ["gen_action", "gen_rpg", "gen_survival", "gen_open_world", "gen_horror", "gen_3d", "cat_controller", "cat_singleplayer", "cat_coop", "is_recent","gen_fps"]
    },
    "Crysis Remastered": {
        "reviews": 15000, 
        "traits": ["gen_action", "gen_scifi", "gen_3d", "cat_controller", "cat_singleplayer", "is_recent", "dev_ea","gen_fps","gen_story"]
    },
    "Crysis 2 Remastered": {
        "reviews": 12000, 
        "traits": ["gen_action", "gen_scifi", "gen_3d", "cat_controller", "cat_singleplayer", "is_recent", "dev_ea"]
    },
    "Control": {
        "reviews": 70000, 
        "traits": ["gen_action", "gen_adventure", "gen_3d", "cat_controller", "cat_singleplayer", "gen_scifi", "gen_story", "is_recent","gen_story"]
    },
    "NBA 2K21": {
        "reviews": 45000, 
        "traits": ["gen_sports_racing", "gen_simulation", "gen_3d", "cat_controller", "cat_singleplayer", "cat_multiplayer", "cat_pvp", "is_recent"]
    },
    "Legends of Runeterra": {
        "reviews": 50000, 
        # Kart oyunu olduğu için 2D ağırlıklı kabul edilir.
        "traits": ["gen_strategy", "gen_2d", "cat_multiplayer", "cat_pvp", "is_recent", "dev_riot"]
    }
}

# Döngü ile verileri güncelleme
for game_name, data in epic_manual_fix.items():
    # 1. İsmi DataFrame'de bul (Birebir eşleşme veya 'contains' ile)
    # Not: final_name temizlenmiş veya orijinal olabilir, en güvenlisi doğrudan eşleşme aramaktır.
    mask = df_final['final_name'] == game_name
    
    # Eğer birebir bulamazsa, temizlenmiş isimlerde ara (case insensitive)
    if not mask.any():
        mask = df_final['final_name'].str.lower() == game_name.lower()
    
    if mask.any():
        # A. İnceleme Sayısını Güncelle (Ham veriyi düzeltiyoruz)
        df_final.loc[mask, 'num_reviews_total'] = data['reviews']
        
        # B. Özellikleri (Traits) Güncelle
        for trait in data['traits']:
            # Eğer sütun varsa 1 yap
            if trait in df_final.columns:
                df_final.loc[mask, trait] = 1
            else:
                # Sütun yoksa (Örn: dev_riot listemizde yoktu) pas geç
                pass
        
        # C. Yıl Bilgisi (Era) Çakışmasını Önle
        # Eğer manuel olarak 'is_recent' dediysek, 'is_retro'yu 0 yapmalıyız.
        if 'is_recent' in data['traits']:
            df_final.loc[mask, ['is_retro', 'is_mid_era']] = 0
        elif 'is_retro' in data['traits']:
            df_final.loc[mask, ['is_recent', 'is_mid_era']] = 0
            
        print(f"   ✅ {game_name} güncellendi.")
    else:
        print(f"   ⚠️ {game_name} veri setinde bulunamadı! (İsim eşleşmedi)")


# --- SON ADIM: NORM_REVIEWS HESABINI GÜNCELLEME ---
# Manuel olarak review sayılarını değiştirdiğimiz için, normalizasyonu tekrar yapmalıyız.
# Aksi takdirde LoL'ün inceleme sayısı 15 milyon olur ama norm_reviews eski (düşük) kalır.

print("🔄 Normalizasyon yeniden hesaplanıyor...")
df_final['reviews_log'] = np.log1p(df_final['num_reviews_total'])
max_val = df_final['reviews_log'].max()
min_val = df_final['reviews_log'].min()
df_final['norm_reviews'] = (df_final['reviews_log'] - min_val) / (max_val - min_val)

print("✅ Tüm manuel düzeltmeler tamamlandı.")


# --- ADIM 14: NUM_REVIEWS SÜTUNUNU MODELE HAZIRLAMA ---

# 1. Logaritma Alma (Uçurumu Kapatma)
# np.log1p fonksiyonu log(1 + x) işlemini yapar. 
# (+1 eklememizin sebebi, 0 incelemesi olan oyunlarda log(0) hatası almamaktır)
df_final['reviews_log'] = np.log1p(df_final['num_reviews_total'])

# 2. 0-1 Arasına Sıkıştırma (Normalization)
# Modeldeki diğer tüm veriler 0 veya 1 olduğu için, bu veri de maksimum 1 olmalı.
max_val = df_final['reviews_log'].max()
min_val = df_final['reviews_log'].min()

# Formül: (Değer - Min) / (Max - Min)
df_final['norm_reviews'] = (df_final['reviews_log'] - min_val) / (max_val - min_val)

df_final.drop('reviews_log',axis=1,inplace=True)

# 3. Temizlik
# Artık ham sayıya (ve ara işlem log sütununa) modelde ihtiyacımız yok.
# SADECE norm_reviews kalacak.
# NOT: Orijinal 'num_reviews_total' sütununu SİLMİYORUZ, çünkü kullanıcıya
# "Bu oyunun 500k incelemesi var" diye göstermek için ona ihtiyacımız var.
# Ama modele sadece 'norm_reviews' girecek.

print("✅ İnceleme sayıları 0-1 arasına ölçeklendi.")
print(df_final[['final_name', 'num_reviews_total', 'norm_reviews']].sort_values('num_reviews_total', ascending=False).head())


# --- ADIM 15: FİNAL TEMİZLİK VE MANUEL SİLME ---

# Gereksiz sütunları temizle
cols_to_clean = ['merge_key', 'name', 'name_epic', 'price', 'price_epic', 
                 'genres_epic', 'release_year_epic', 'developer_epic', 
                 'windows_epic', 'mac_epic', 'linux_epic', 'text_search']
df_final = df_final.drop(columns=[c for c in cols_to_clean if c in df_final.columns])

# Eksik sayısal veriler (Model bozulmasın diye 0 ile doldur)
df_final['num_reviews_total'] = df_final['num_reviews_total'].fillna(0)
df_final['metacritic_score'] = df_final['metacritic_score'].fillna(0)
# pct_pos_total yukarda dolmuştu ama yine de kontrol
if 'pct_pos_total' in df_final.columns:
    df_final['pct_pos_total'] = df_final['pct_pos_total'].fillna(50)

# Platformları int yap
for col in ['windows', 'mac', 'linux']:
    df_final[col] = df_final[col].fillna(0).astype(int)

# Tekrarları sil
df_final = df_final.drop_duplicates(subset=['final_name'])

# Manuel İstenmeyenler Listesi
# ÖNEMLİ: Valorant burada silinirse manuel eklediğin de gider. 
# Eğer Epic'ten gelen hatalı "VALORANT" ise, büyük küçük harf duyarlı olduğu için sorun olmaz.
unwanted_names = [
    'OHDcore Mod Kit', 'iHeart: Radio, Music, Podcast', 'Discord', 
    'DCL The Game - Track Editor', 'Brave', 
    'Bus Simulator 21 - Modding Kit', 'Opera GX - The First Browser for Gamers', 
    'Itch.io',"It Takes Two Friend's Pass"
]
# İsim listesinde varsa at
df_final = df_final[~df_final['final_name'].isin(unwanted_names)]

print(f"İşlem Tamam! Final Veri Sayısı: {len(df_final)}")
df_final.to_csv("oyun_projesi_final_veri.csv", index=False)
print("✅ Kayıt Başarılı.")