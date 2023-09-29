import streamlit as st
import lightgbm as lgb
import pandas as pd
import numpy as np

def predict(data):
    # 保存したモデルを読み込み
    booster = lgb.Booster(model_file='model/model.txt')
    X_test = data.drop(['着順','日付'], axis=1)
    y_pred = booster.predict(X_test)

    return y_pred

def main():
    st.title("競馬予想アプリ")

    uploaded_file = st.file_uploader("テストデータのCSVファイルをアップロードしてください", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, encoding='Shift-JIS')
        y_pred = predict(data)

        # 予測結果を表示
        st.header("予測結果")
        st.write(y_pred)

if __name__ == '__main__':
    main()
