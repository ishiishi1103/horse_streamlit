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

def processing(data):
    data['着順'] = data['着順'].map(lambda x: 1 if x<4 else 0)
  #  count_n_horse = data['race_id'].value_counts()
  #  data['出走頭数'] = data['race_id'].map(count_n_horse)
    data['乗り替わり']  = (data['騎手'] != data['騎手1']).astype(int)
    data['斤量差'] = data['斤量'] - data['斤量1']
    data['馬場差'] = data['馬場'] - data['馬場1']
    data['芝・ダート替わり'] = (data['芝・ダート'] != data['芝・ダート1']).astype(int)
    data['クラス差'] = data['クラス'] - data['クラス1']
    data = data.drop(['馬場1', '芝・ダート1', '斤量1','距離1', 'クラス1', 'race_id'], axis=1)
    data = data[data['人気'] == 1]

    data.shape

    return data

def main():
    st.title("競馬予想アプリ")

 # Input data collection
    st.sidebar.header('1番人気のデータを入力してください')

    # サイドバーでの入力
    騎手 = st.sidebar.text_input('騎手')
    馬番 = st.sidebar.number_input('馬番', value=1, step=1)
    三着以内 = st.sidebar.number_input('着順', value=1, step=1)
    体重 = st.sidebar.number_input('体重', value=50.0)
    体重変化 = st.sidebar.number_input('体重変化', value=0.0)
    性 = st.sidebar.selectbox('性', ['牡', '牝', 'セ'])
    齢 = st.sidebar.number_input('齢', value=1, step=1)
    斤量 = st.sidebar.number_input('斤量', value=50.0)
    人気 = st.sidebar.number_input('人気', value=1, step=1)
    日付 = st.sidebar.date_input('日付')
    クラス = st.sidebar.number_input('クラス', value=1, step=1)
    芝ダート = st.sidebar.selectbox('芝・ダート', ['芝', 'ダート'])
    距離 = st.sidebar.number_input('距離', value=1000, step=50)
    回り = st.sidebar.selectbox('回り', ['右', '左'])
    馬場 = st.sidebar.number_input('馬場', value=1, step=1)
    場id = st.sidebar.text_input('場id')
    場名 = st.sidebar.text_input('場名')
    距離差 = st.sidebar.number_input('距離差', value=0, step=50)
    日付差 = st.sidebar.number_input('日付差', value=0, step=1)
    オッズ1 = st.sidebar.number_input('オッズ1', value=1.0)
    着順1 = st.sidebar.number_input('着順1', value=1, step=1)
    出走頭数 = st.sidebar.number_input('出走頭数', value=1, step=1)
    乗り替わり = st.sidebar.number_input('乗り替わり', value=0, step=1)
    斤量差 = st.sidebar.number_input('斤量差', value=0.0)
    馬場差 = st.sidebar.number_input('馬場差', value=0)
    芝ダート替わり = st.sidebar.number_input('芝・ダート替わり', value=0, step=1)
    クラス差 = st.sidebar.number_input('クラス差', value=0, step=1)

    if st.button("予測する"):
        input_data = pd.DataFrame({
            '騎手': [騎手], '馬番': [馬番], '着順': [三着以内], '体重': [体重], 
            '体重変化': [体重変化], '性': [性], '齢': [齢], '斤量': [斤量], 
            '人気': [人気], '日付': [日付], 'クラス': [クラス], '芝・ダート': [芝ダート], 
            '距離': [距離], '回り': [回り], '馬場': [馬場], '場id': [場id], '場名': [場名],
            '距離差': [距離差], '日付差': [日付差], 'オッズ1': [オッズ1], '着順1': [着順1], 
            '出走頭数': [出走頭数], '乗り替わり': [乗り替わり], '斤量差': [斤量差], '馬場差': [馬場差],
            '芝・ダート替わり': [芝ダート替わり], 'クラス差': [クラス差]
        })

        # モデルで予測
        y_pred = predict(input_data)

        # 予測結果を表示
        st.header("予測結果")
        st.write(y_pred)


if __name__ == '__main__':
    main()
