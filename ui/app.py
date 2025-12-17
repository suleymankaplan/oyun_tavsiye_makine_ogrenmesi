from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# --- DOSYA YOLLARI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(ROOT_DIR, 'oyun_projesi_clustered.csv')
KNN_MODEL_PATH = os.path.join(ROOT_DIR, 'knn_model.pkl')

# --- VERİ VE MODEL YÜKLEME ---
try:
    df = pd.read_csv(DATA_PATH)
    with open(KNN_MODEL_PATH, 'rb') as f:
        knn_model = pickle.load(f)
    print("✅ Veri ve Model başarıyla yüklendi.")
except FileNotFoundError as e:
    print(f"❌ Hata: Dosyalar bulunamadı! Lütfen yolları kontrol et.\n{e}")
    exit()

# --- ÖZELLİK SEÇİMİ ---
feature_cols = [col for col in df.columns if col.startswith(('gen_', 'cat_', 'is_', 'dev_'))]
if 'lang_turkish' in df.columns: feature_cols.append('lang_turkish')
if 'norm_reviews' in df.columns: feature_cols.append('norm_reviews')

# --- YARDIMCI FONKSİYON: 0/1 SÜTUNLARINDAN TÜR İSMİ ÇIKARMA ---
# 'genres' sütunu silindiği için, 'gen_action'=1 ise "Action" yazısını biz üretiyoruz.
def get_genres_from_row(row):
    genres = []
    for col in df.columns:
        if col.startswith('gen_') and row[col] == 1:
            # "gen_open_world" -> "Open World" çevirimi
            clean_name = col.replace('gen_', '').replace('_', ' ').title()
            
            # Özel düzeltmeler (Görsellik için)
            if clean_name == "Rpg": clean_name = "RPG"
            if clean_name == "Fps": clean_name = "FPS"
            if clean_name == "Tps": clean_name = "TPS"
            if clean_name == "Mmo": clean_name = "MMO"
            
            genres.append(clean_name)
    return genres

# --- ROUTES ---

@app.route('/')
def index():
    # İlk 3000 popüler oyunu listele
    game_list = df.sort_values('num_reviews_total', ascending=False)['final_name'].head(3000).tolist()
    return render_template('index.html', games=game_list)

@app.route('/recommend', methods=['POST'])
def recommend():
    game_name = request.form.get('game_name')
    
    # 1. Oyun kontrolü
    game_row = df[df['final_name'].str.lower() == game_name.lower()]
    
    if game_row.empty:
        # Hata durumunda listeyi tekrar gönderiyoruz ki dropdown çalışsın
        game_list = df.sort_values('num_reviews_total', ascending=False)['final_name'].head(3000).tolist()
        return render_template('index.html', error="Oyun bulunamadı! Lütfen listeden seçin.", games=game_list)

    # 2. Vektör alma
    game_index = game_row.index[0]
    query_vector = df.loc[game_index, feature_cols].values.reshape(1, -1)

    # 3. KNN ile komşu bulma (7 komşu: 1'i kendisi, 6'sı tavsiye)
    distances, indices = knn_model.kneighbors(query_vector, n_neighbors=7)
    
    # 4. Sonuçları hazırlama
    recommendations = []
    
    # indices[0][1:] -> İlk sonuç (0. index) oyunun kendisidir, onu atlıyoruz.
    for i in indices[0][1:]:
        rec_game = df.iloc[i]
        
        # Platformlar
        platforms = []
        if rec_game['windows'] == 1: platforms.append('Windows')
        if rec_game['mac'] == 1: platforms.append('Mac')
        if rec_game['linux'] == 1: platforms.append('Linux')
        
        # Mağazalar
        stores = []
        if rec_game['on_steam'] == 1: stores.append('Steam')
        if rec_game['on_epic'] == 1: stores.append('Epic')

        # DİNAMİK TÜR LİSTESİ (Hata buradaydı, düzeltildi)
        current_genres = get_genres_from_row(rec_game)

        rec_data = {
            'name': rec_game['final_name'],
            'image': rec_game['header_image'],
            'price': "Ücretsiz" if rec_game['final_price'] == 0 else f"${rec_game['final_price']}",
            'genres': current_genres[:3], # İlk 3 türü al
            'platforms': platforms,
            'stores': stores,
            'reviews': int(rec_game['num_reviews_total']),
            'release': int(rec_game['release_year'])
        }
        recommendations.append(rec_data)

    # Seçilen oyunun detayları
    # Burası da hata veriyordu, onu da düzelttik
    selected_row = game_row.iloc[0]
    selected_genres_list = get_genres_from_row(selected_row)
    
    selected_game = {
        'name': selected_row['final_name'],
        'image': selected_row['header_image'],
        'description': ", ".join(selected_genres_list[:4]) # İlk 4 türü virgülle birleştir
    }

    return render_template('result.html', recommendations=recommendations, selected=selected_game)

if __name__ == '__main__':
    app.run(debug=True, port=5000)