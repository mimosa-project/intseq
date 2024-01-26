import pstats

# プロファイリングデータを読み込み
stats = pstats.Stats('profiles/profile_data.prof')

# 統計情報を表示
stats.print_stats()