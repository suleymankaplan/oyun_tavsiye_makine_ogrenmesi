# --- ADIM 13: EPIC GAMES TEMİZLİĞİ (ÇÖP VERİLERİ SİLME) ---

# # 1. KORUNACAK VIP OYUNLAR (Whitelist)
# # Buraya senin manuel eklediklerini ve silinmesini istemediğin büyük oyunları yaz.
# vip_epic_games = [
#     'League of Legends', 
#     'Valorant', 
#     'Minecraft', 
#     'Fortnite', 
#     'Rocket League', 
#     'Genshin Impact', 
#     'Alan Wake 2',
#     'Fall Guys'
# ]

# # 2. SİLME MANTIĞI
# # Koşul: (Sadece Epic'te olsun) VE (Steam'de olmasın) VE (İsmi VIP listede OLMASIN)
# mask_trash_epic = (df_final['on_epic'] == 1) & \
#                   (df_final['on_steam'] == 0) & \
#                   (~df_final['final_name'].isin(vip_epic_games))

# # Silmeden önce kaç tane gidecek görelim
# print(f"Silinecek Niteliksiz Epic Oyunu Sayısı: {mask_trash_epic.sum()}")

# # Temizlik
# df_final = df_final[~mask_trash_epic]

# print(f"Temizlik Sonrası Toplam Oyun: {len(df_final)}")

# # KONTROL: VIP oyunlar duruyor mu?
# print("\nVIP Oyun Kontrolü:")
# print(df_final[df_final['final_name'].isin(vip_epic_games)]['final_name'].unique())
