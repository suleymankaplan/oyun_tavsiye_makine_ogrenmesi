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

if 'pct_pos_total' in df_final.columns:
    avg_score = df_final['pct_pos_total'].mean()
    df_final['pct_pos_total'] = df_final['pct_pos_total'].fillna(avg_score)
else:
    df_final['pct_pos_total'] = 70

df_final['windows'] = df_final['windows'].combine_first(df_final['windows_epic']).fillna(1)
df_final['mac'] = df_final['mac'].combine_first(df_final['mac_epic']).fillna(0)
df_final['linux'] = df_final['linux'].combine_first(df_final['linux_epic']).fillna(0)

df_final['on_steam'] = df_final['name'].notna().astype(int)
df_final['on_epic'] = df_final['name_epic'].notna().astype(int)


# --- ADIM 5: VR TEMİZLİĞİ ---
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


# --- ADIM 12: ÖNEMLİ GELİŞTİRİCİLERİ MODEL İÇİN İŞARETLEME ---

df_final['dev_pub_combined'] = (df_final['developers'].fillna('') + " " + df_final['publishers'].fillna('')).astype(str).str.lower()

target_devs = {
    'dev_rockstar': ['rockstar games', 'rockstar north'],
    'dev_ubisoft': ['ubisoft', 'ubisoft montreal'],
    'dev_valve': ['valve'], 
    'dev_bethesda': ['bethesda', 'bethesda softworks'],
    'dev_ea': ['electronic arts', 'ea sports', 'dice', 'bioware'],
    'dev_square_enix': ['square enix'],
    'dev_capcom': ['capcom'],
    'dev_fromsoftware': ['fromsoftware'],
    'dev_cdprojekt': ['cd projekt red'],
    'dev_sony': ['playstation', 'sony interactive', 'naughty dog', 'santa monica']
}

print("Önemli Geliştiriciler İşaretleniyor...")

for col_name, keywords in target_devs.items():
    df_final[col_name] = df_final['dev_pub_combined'].apply(lambda x: 1 if any(k in x for k in keywords) else 0)

df_final = df_final.drop(columns=['dev_pub_combined'])

# Kontrol
print(df_final[['final_name', 'dev_rockstar', 'dev_valve', 'dev_ubisoft']].head())

# Rockstar oyunlarına bakalım çalışmış mı?
print("\nRockstar Oyunları:")
print(df_final[df_final['dev_rockstar'] == 1]['final_name'].head())


# --- ADIM 13: EPIC GAMES TEMİZLİĞİ (ÇÖP VERİLERİ SİLME) ---

# 1. KORUNACAK VIP OYUNLAR (Whitelist)
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
    'Rocket League®', 
    'Genshin Impact', 
    'Fall Guys',
    'Control'
]

# 2. SİLME MANTIĞI
mask_trash_epic = (df_final['on_epic'] == 1) & \
                  (df_final['on_steam'] == 0) & \
                  (~df_final['final_name'].isin(vip_epic_games))

print(f"Silinecek Niteliksiz Epic Oyunu Sayısı: {mask_trash_epic.sum()}")
df_final = df_final[~mask_trash_epic]
print(f"Temizlik Sonrası Toplam Oyun: {len(df_final)}")


# --- ADIM: EPIC GAMES ÖZEL OYUNLARINI MANUEL DOLDURMA VE RESİM EKLEME ---
print("🛠️ Epic Games Özel Oyunları Manuel Olarak Dolduruluyor...")
epic_manual_fix = {
    "League of Legends": {
        "reviews": 15000000, 
        "traits": ["gen_rpg", "gen_strategy", "gen_3d", "cat_multiplayer", "cat_mmo", "cat_coop", "cat_pvp", "is_recent", "dev_riot"],
        "image": "https://upload.wikimedia.org/wikipedia/commons/d/d8/League_of_Legends_2019_vector.svg" # Logo/Art
    },
    "VALORANT": {
        "reviews": 10000000, 
        "traits": ["gen_action", "gen_3d", "cat_multiplayer", "cat_pvp", "cat_coop", "is_recent", "dev_riot","gen_fps"],
        "image": "https://upload.wikimedia.org/wikipedia/commons/f/fc/Valorant_logo_-_pink_color_version.svg"
    },
    "Fortnite": {
        "reviews": 12000000, 
        "traits": ["gen_action", "gen_survival", "gen_3d", "cat_controller", "cat_multiplayer", "cat_coop", "cat_pvp", "gen_open_world", "is_recent","gen_sandbox"],
        "image": "https://cdn2.unrealengine.com/fneco-2025-keyart-thumb-1920x1080-de84aedabf4d.jpg"
    },
    "Genshin Impact": {
        "reviews": 8000000, 
        "traits": ["gen_rpg", "gen_open_world", "gen_anime", "gen_adventure", "gen_3d", "cat_controller", "cat_singleplayer", "cat_coop", "is_recent"],
        "image": "https://image.api.playstation.com/vulcan/ap/rnd/202508/2602/cc615ad198b2727cea3aa9f63c68577d5aa322f4ec974905.jpg"
    },
    "Minecraft": {
        "reviews": 20000000,
        "traits": ["gen_adventure", "gen_simulation", "gen_open_world", "gen_survival", "gen_3d", "cat_controller", "cat_singleplayer", "cat_multiplayer", "cat_coop", "is_mid_era","gen_sandbox","gen_fps"],
        "image": "https://assets-prd.ignimgs.com/2021/12/14/minecraft-1639513933156.jpg"
    },
    "Tom Clancy's Splinter Cell": {
        "reviews": 20000, 
        "traits": ["gen_action", "gen_adventure", "gen_3d", "cat_controller", "cat_singleplayer", "is_retro", "dev_ubisoft"],
        "image": "https://cdn1.epicgames.com/54090fbc182e44d79b2872f24994c190/offer/SCEL_Store_Landscape_2560x1440-2560x1440-0966b116c19dc7029f8e19b918af7309.jpg"
    },
    "Splinter Cell Chaos Theory": {
        "reviews": 15000, 
        "traits": ["gen_action", "gen_adventure", "gen_3d", "cat_controller", "cat_singleplayer", "cat_coop", "is_retro", "dev_ubisoft"],
        "image": "https://store-images.s-microsoft.com/image/apps.21616.68747576901602782.585f0e62-5630-484c-8ce3-2c4ced4d9cef.e69c9a76-c585-4847-8663-01b1281f0ac0?q=90&w=480&h=270"
    },
    "Marvel's Guardians of the Galaxy": {
        "reviews": 30000, 
        "traits": ["gen_action", "gen_adventure", "gen_3d", "cat_controller", "cat_singleplayer", "gen_scifi", "gen_story", "is_recent", "dev_square_enix"],
        "image": "https://image.api.playstation.com/vulcan/ap/rnd/202106/0215/Pw9cWnyqkix3EoCOGqrN1cgN.png"
    },
    "LEGO® Batman™: The Videogame": {
        "reviews": 10000, 
        "traits": ["gen_action", "gen_adventure", "gen_3d", "cat_controller", "cat_singleplayer", "cat_coop", "cat_split_screen", "is_retro"],
        "image": "https://upload.wikimedia.org/wikipedia/en/e/e3/Lego_batman_cover.jpg"
    },
    "HITMAN 3": {
        "reviews": 40000, 
        "traits": ["gen_action", "gen_adventure", "gen_3d", "cat_controller", "cat_singleplayer", "gen_puzzle", "is_recent"],
        "image":"https://www.google.com/url?sa=t&source=web&rct=j&url=https%3A%2F%2Fwww.keycense.com%2Fhitman-3%3Fsrsltid%3DAfmBOoq6ZSSXb2Ap1FIx-XUcBGNcw9ckH75IWbT38Xrfa7K2QctIKzGo&ved=0CBUQjRxqFwoTCKi5t4LRy5EDFQAAAAAdAAAAABAj&opi=89978449"
    },
    "Dying Light 2 Stay Human": {
        "reviews": 120000, 
        "traits": ["gen_action", "gen_rpg", "gen_survival", "gen_open_world", "gen_horror", "gen_3d", "cat_controller", "cat_singleplayer", "cat_coop", "is_recent","gen_fps"],
        "image": "https://upload.wikimedia.org/wikipedia/en/6/6d/Dying_Light_2_cover_art.jpg"
    },
    "Crysis Remastered": {
        "reviews": 15000, 
        "traits": ["gen_action", "gen_scifi", "gen_3d", "cat_controller", "cat_singleplayer", "is_recent", "dev_ea","gen_fps","gen_story"],
        "image": "https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/1715130/capsule_616x353.jpg?t=1709651863"
    },
    "Crysis 2 Remastered": {
        "reviews": 12000, 
        "traits": ["gen_action", "gen_scifi", "gen_3d", "cat_controller", "cat_singleplayer", "is_recent", "dev_ea"],
        "image": "https://cdn1.epicgames.com/salesEvent/salesEvent/EGS_Crysis2Remastered_Crytek_S1_2560x1440-fd507ed73e95770a3fe990ee78e3012f"
    },
    "Control": {
        "reviews": 70000, 
        "traits": ["gen_action", "gen_adventure", "gen_3d", "cat_controller", "cat_singleplayer", "gen_scifi", "gen_story", "is_recent","gen_story"],
        "image": "https://upload.wikimedia.org/wikipedia/en/e/e5/Control_game_cover_art.jpg"
    },
    "NBA 2K21": {
        "reviews": 45000, 
        "traits": ["gen_sports_racing", "gen_simulation", "gen_3d", "cat_controller", "cat_singleplayer", "cat_multiplayer", "cat_pvp", "is_recent"],
        "image": "https://image.api.playstation.com/vulcan/ap/rnd/202011/0516/uymryFlvJymtWtSm8jnbuMxf.png"
    },
    "Legends of Runeterra": {
        "reviews": 50000, 
        "traits": ["gen_strategy", "gen_2d", "cat_multiplayer", "cat_pvp", "is_recent", "dev_riot"],
        "image": "https://cdn1.epicgames.com/offer/4fb89e9f47fe48258314c366649c398e/EGS_LegendsofRuneterra_RiotGames_S1_2560x1440-53b4135a798b686f67f2a95de625858f"
    },
    "Rocket League®": {
        "reviews": 500000,
        "traits": ["gen_sports_racing", "gen_action", "cat_multiplayer", "cat_pvp", "cat_coop", "cat_controller", "is_recent"],
        "image": "https://upload.wikimedia.org/wikipedia/commons/e/e0/Rocket_League_coverart.jpg"
    },
    "Fall Guys": {
        "reviews": 400000,
        "traits": ["gen_action", "gen_arcade", "cat_multiplayer", "cat_pvp", "cat_controller", "is_recent"],
        "image": "https://cdn1.epicgames.com/offer/50118b7f954e450f8823df1614b24e80/FGSS04_KeyArt_OfferImagePortrait_1200x1600_1200x1600-4bd46574e78464352e1f2c55714701f7"
    }
}

# Döngü ile verileri güncelleme (Sadece listedekileri yapar)
for game_name, data in epic_manual_fix.items():
    # 1. İsmi DataFrame'de bul (Tam eşleşme öncelikli)
    mask = df_final['final_name'] == game_name
    
    # Eğer birebir bulamazsa, küçük harfle ara
    if not mask.any():
        mask = df_final['final_name'].str.lower() == game_name.lower()
    
    if mask.any():
        # A. İnceleme Sayısı
        df_final.loc[mask, 'num_reviews_total'] = data.get('reviews', 0)
        
        # B. Özellikler
        for trait in data.get('traits', []):
            if trait in df_final.columns:
                df_final.loc[mask, trait] = 1
        
        # C. Resim (Varsa ekle)
        if 'image' in data:
            df_final.loc[mask, 'header_image'] = data['image']
        
        # D. Yıl Bilgisi (Era) Çakışmasını Önle
        if 'is_recent' in data.get('traits', []):
            df_final.loc[mask, ['is_retro', 'is_mid_era']] = 0
        elif 'is_retro' in data.get('traits', []):
            df_final.loc[mask, ['is_recent', 'is_mid_era']] = 0
            
        print(f"   ✅ {game_name} güncellendi.")
    else:
        print(f"   ⚠️ {game_name} veri setinde bulunamadı! (İsim eşleşmedi)")


# --- TÜR DÜZELTMELERİ (Rainbow Six & GTA) ---

print("\n🛠️ Tür Hataları Gideriliyor...")

# 1. Rainbow Six Siege - Simulation Kaldırma
mask_r6 = df_final['final_name'].str.contains("Rainbow Six", case=False, na=False) & \
          df_final['final_name'].str.contains("Siege", case=False, na=False)
if mask_r6.any():
    df_final.loc[mask_r6, 'gen_simulation'] = 0
    print("   ✅ Rainbow Six Siege'den 'Simulation' etiketi kaldırıldı.")

# 2. GTA Enhanced & Legacy & Counter-Strike 2 - Sports/Racing Kaldırma
mask_gta = df_final['final_name'].str.contains("Grand Theft Auto", case=False, na=False) & \
           df_final['final_name'].str.contains("Trilogy|Definitive|Enhanced", case=False, na=False)
mask_legacy = df_final['final_name'].str.contains("Legacy", case=False, na=False) # Hogwarts Legacy vb.
mask_cs2 = df_final['final_name'].str.contains("Counter-Strike 2", case=False, na=False) # CS2

mask_remove_sport = mask_gta | mask_legacy | mask_cs2

if mask_remove_sport.any():
    affected_games = df_final.loc[mask_remove_sport, 'final_name'].unique()
    df_final.loc[mask_remove_sport, 'gen_sports_racing'] = 0
    print(f"   ✅ Şu oyunlardan 'Sports/Racing' kaldırıldı: {', '.join(affected_games[:5])}...")

# --- ADIM 14: NUM_REVIEWS SÜTUNUNU MODELE HAZIRLAMA ---

print("\n🔄 Normalizasyon yeniden hesaplanıyor...")

# 1. Logaritma Alma
df_final['reviews_log'] = np.log1p(df_final['num_reviews_total'])

# 2. 0-1 Arasına Sıkıştırma (Normalization)
max_val = df_final['reviews_log'].max()
min_val = df_final['reviews_log'].min()

# Formül: (Değer - Min) / (Max - Min)
df_final['norm_reviews'] = (df_final['reviews_log'] - min_val) / (max_val - min_val)

df_final.drop('reviews_log',axis=1,inplace=True)

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
if 'pct_pos_total' in df_final.columns:
    df_final['pct_pos_total'] = df_final['pct_pos_total'].fillna(50)
for col in ['windows', 'mac', 'linux']:
    df_final[col] = df_final[col].fillna(0).astype(int)

# Tekrarları sil
df_final = df_final.drop_duplicates(subset=['final_name'])

# Manuel İstenmeyenler Listesi
unwanted_names = [
    'OHDcore Mod Kit', 'iHeart: Radio, Music, Podcast', 'Discord', 
    'DCL The Game - Track Editor', 'Brave', 
    'Bus Simulator 21 - Modding Kit', 'Opera GX - The First Browser for Gamers', 
    'Itch.io',"It Takes Two Friend's Pass","Angel Legion"
]
df_final = df_final[~df_final['final_name'].isin(unwanted_names)]

print(f"İşlem Tamam! Final Veri Sayısı: {len(df_final)}")
df_final.to_csv("oyun_projesi_final_veri.csv", index=False)
print("✅ Kayıt Başarılı.")