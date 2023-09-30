import streamlit as st
import lightgbm as lgb
import pandas as pd
import numpy as np

def predict(data):
    # 保存したモデルを読み込み
    booster = lgb.Booster(model_file='model/model_rev3.txt')
   # X_test = data.drop(['着順','日付'], axis=1)
    y_pred = booster.predict(data)

    return y_pred

def processing(data):
  #  data['着順'] = data['着順'].map(lambda x: 1 if x<4 else 0)
  #  count_n_horse = data['race_id'].value_counts()
  #  data['出走頭数'] = data['race_id'].map(count_n_horse)
  #  data['乗り替わり']  = (data['騎手'] != data['騎手1']).astype(int)
  #  data['斤量差'] = data['斤量'] - data['斤量1']
  #  data['馬場差'] = data['馬場'] - data['馬場1']
  #  data['芝・ダート替わり'] = (data['芝・ダート'] != data['芝・ダート1']).astype(int)
  #  data['クラス差'] = data['クラス'] - data['クラス1']
  #  data = data.drop(['馬場1', '芝・ダート1', '斤量1','距離1', 'クラス1', 'race_id'], axis=1)
  #  data = data[data['人気'] == 1]

    data.shape

    return data

def processing_inputdata(data):
    data['性'] = data['性'].replace({'牡': 0, '牝': 1, 'セ': 2})
    data['芝・ダート'] = data['芝・ダート'].replace({'芝': 0, 'ダ': 1, '障': 2})
    data['回り'] = data['回り'].replace({'右': 0, '左': 1})
    data['クラス'] = data['クラス'].replace({'障害':0, 'G1': 10, 'G2': 9, 'G3': 8, '(L)': 7, 'オープン': 7,'OP': 7, '3勝': 6, '1600': 6, '2勝': 5, '1000': 5, '1勝': 4, '500': 4, '新馬': 3, '未勝利': 1})
    data['馬場'] = data['馬場'].replace({'良': 0, '稍重': 1, '重': 2, '不良': 3})
    data['場id'] = data['場id'].replace({'札幌': 0, '函館': 1, '福島': 2, '新潟': 3, '東京': 4, '中山': 5, '中京': 6, '京都': 7, '阪神': 8, '小倉': 9})

    return data



def main():
    st.title("競馬予想アプリ")
    st.header('入力データのルール')
    st.write('通過順：前走通過順の平均値を入力すること。')
    st.write('馬場差：前走の馬場状態との差を数値で入力すること。馬場は以下の通り数値化して差をとる。\n　良: 0, 稍重: 1, 重: 2, 不良: 3')

 # Input data collection
    st.sidebar.header('1番人気のデータを入力してください')

    # サイドバーでの入力
    馬番 = st.sidebar.number_input('馬番', value=1, step=1)
    体重 = st.sidebar.number_input('体重', value=500)
    体重変化 = st.sidebar.number_input('体重変化', value=0)
    性 = st.sidebar.selectbox('性', ['牡', '牝', 'セ'])
    齢 = st.sidebar.number_input('齢', value=1, step=1)
    斤量 = st.sidebar.number_input('斤量', value=50.0)
    クラス = st.sidebar.selectbox('クラス', ['障害',  'G1', 'G2', 'G3', '(L)', 'OP', '3勝', '2勝','1勝', '新馬', '未勝利'])
    芝ダート = st.sidebar.selectbox('芝・ダート', ['芝', 'ダート', '障害'])
    距離 = st.sidebar.number_input('距離', value=1000, step=100)
    回り = st.sidebar.selectbox('回り', ['右', '左'])
    馬場 = st.sidebar.selectbox('馬場', ['良', '稍重', '重', '不良'])
    場id = st.sidebar.selectbox('場id', ['札幌', '函館', '福島', '新潟', '東京', '中山', '中京', '京都', '阪神', '小倉'])
    距離差 = st.sidebar.number_input('距離差', value=0, step=50)
    日付差 = st.sidebar.number_input('日付差', value=0, step=1)
    オッズ1 = st.sidebar.number_input('オッズ1', value=1.0)
    着順1 = st.sidebar.number_input('着順1', value=1, step=1)
    前走通過順_平均= st.sidebar.number_input('通過順', value=1.0)
    出走頭数 = st.sidebar.number_input('出走頭数', value=1, step=1)
    斤量差 = st.sidebar.number_input('斤量差', value=0.0)
    馬場差 = st.sidebar.number_input('馬場差', value=0)

    if st.button("予測する"):
        input_data = pd.DataFrame({
            '馬番': [馬番], '体重': [体重], 
            '体重変化': [体重変化], '性': [性], '齢': [齢], '斤量': [斤量], 
            'クラス': [クラス], '芝・ダート': [芝ダート], 
            '距離': [距離], '回り': [回り], '馬場': [馬場], '場id': [場id],
            '距離差': [距離差], '日付差': [日付差], 'オッズ1': [オッズ1], '着順1': [着順1], '通過順': [前走通過順_平均],
            '出走頭数': [出走頭数],  '斤量差': [斤量差], '馬場差': [馬場差],
        })

        input_data = processing_inputdata(input_data)

        # モデルで予測
        y_pred = predict(input_data)

        # 予測結果を表示
        st.header("予測結果")
        st.write(y_pred)


if __name__ == '__main__':
    main()
