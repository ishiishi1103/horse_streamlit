import pandas as pd

# CSVファイルの読み込み
df = pd.read_csv('encoded/encoded_data.csv')

# 上位100行の表示
print(df.head(100))

# 上位100行を新しいCSVファイルとして出力
df.head(100).to_csv('encoded/encoded_data_100.csv', index=False)