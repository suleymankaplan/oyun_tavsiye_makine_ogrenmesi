from preprocessing import df_final

# on_epic 1 OLSUN ve on_steam 0 OLSUN
epic_only_count = len(df_final[(df_final['on_epic'] == 1) & (df_final['on_steam'] == 0)])

print(f"Sadece Epic'te olan oyun sayısı: {epic_only_count}")