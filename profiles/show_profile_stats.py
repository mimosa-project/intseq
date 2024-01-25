import pstats

# プロファイリングデータを読み込み
stats = pstats.Stats('profiles/profile_data_after_correcting_generate_random_letter.prof')

# プロファイリングデータをtottimeで昇順にソート
stats.sort_stats('tottime')

# 統計情報を表示
stats.print_stats(-1)