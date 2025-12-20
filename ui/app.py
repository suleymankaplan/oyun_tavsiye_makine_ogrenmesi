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
HIT_MODEL_PATH = os.path.join(ROOT_DIR, 'hit_model.pkl') # YENİ

# --- VERİ VE MODELLERİ YÜKLEME ---
try:
    df = pd.read_csv(DATA_PATH)
    
    with open(KNN_MODEL_PATH, 'rb') as f:
        knn_model = pickle.load(f)
        
    # YENİ: Hit Modelini Yükle
    with open(HIT_MODEL_PATH, 'rb') as f:
        hit_model = pickle.load(f)
        
    print("✅ Veri ve Modeller (KNN + Hit) başarıyla yüklendi.")
except FileNotFoundError as e:
    print(f"❌ Hata: Dosyalar bulunamadı! {e}")
    exit()

# --- ÖZELLİK SEÇİMİ (KNN İÇİN) ---
feature_cols = [col for col in df.columns if col.startswith(('gen_', 'cat_', 'is_', 'dev_'))]
if 'lang_turkish' in df.columns: feature_cols.append('lang_turkish')
if 'norm_reviews' in df.columns: feature_cols.append('norm_reviews')

# --- YENİ: ÖZELLİK SEÇİMİ (HIT MODELİ İÇİN) ---
# Hit modelini eğitirken review sayılarını çıkarmıştık, burada da çıkarmalıyız.
# Yoksa "Shape Mismatch" hatası alırız.
hit_features = [col for col in df.columns 
                if col not in ['final_name', 'header_image', 'cluster_label', 'pca_x', 'pca_y', 
                               'num_reviews_total', 'norm_reviews', 'is_hit'] 
                and df[col].dtype in [np.float64, np.int64]]

# --- YARDIMCI FONKSİYON ---
def get_genres_from_row(row):
    genres = []
    for col in df.columns:
        if col.startswith('gen_') and row[col] == 1:
            clean_name = col.replace('gen_', '').replace('_', ' ').title()
            if clean_name == "Rpg": clean_name = "RPG"
            if clean_name == "Fps": clean_name = "FPS"
            genres.append(clean_name)
    return genres

# --- ROUTES ---
@app.route('/')
def index():
    game_list = df.sort_values('num_reviews_total', ascending=False)['final_name'].head(3000).tolist()
    return render_template('index.html', games=game_list)

@app.route('/recommend', methods=['POST'])
def recommend():
    game_name = request.form.get('game_name')
    game_row = df[df['final_name'].str.lower() == game_name.lower()]
    
    if game_row.empty:
        game_list = df.sort_values('num_reviews_total', ascending=False)['final_name'].head(3000).tolist()
        return render_template('index.html', error="Oyun bulunamadı!", games=game_list)

    # KNN Tahmini
    game_index = game_row.index[0]
    query_vector = df.loc[game_index, feature_cols].values.reshape(1, -1)
    distances, indices = knn_model.kneighbors(query_vector, n_neighbors=7)
    
    recommendations = []
    for i in indices[0][1:]:
        rec_game = df.iloc[i]
        
        # --- YENİ: HIT TAHMİNİ ---
        # Oyunun özelliklerini hit modeline uygun hale getir
        hit_vector = rec_game[hit_features].values.reshape(1, -1)
        is_hit_prediction = hit_model.predict(hit_vector)[0] # 1 veya 0 döner
        
        platforms = []
        if rec_game['windows'] == 1: platforms.append('Windows')
        if rec_game['mac'] == 1: platforms.append('Mac')
        
        stores = []
        if rec_game['on_steam'] == 1: stores.append('Steam')
        if rec_game['on_epic'] == 1: stores.append('Epic')

        rec_data = {
            'name': rec_game['final_name'],
            'image': rec_game['header_image'],
            'price': "Ücretsiz" if rec_game['final_price'] == 0 else f"${rec_game['final_price']}",
            'genres': get_genres_from_row(rec_game)[:3],
            'platforms': platforms,
            'stores': stores,
            'reviews': int(rec_game['num_reviews_total']),
            'release': int(rec_game['release_year']),
            'is_hit': int(is_hit_prediction) # HTML'e gönderiyoruz
        }
        recommendations.append(rec_data)

    selected_row = game_row.iloc[0]
    selected_game = {
        'name': selected_row['final_name'],
        'image': selected_row['header_image'],
        'description': ", ".join(get_genres_from_row(selected_row)[:4])
    }

    return render_template('result.html', recommendations=recommendations, selected=selected_game)

if __name__ == '__main__':
    app.run(debug=True, port=5000)