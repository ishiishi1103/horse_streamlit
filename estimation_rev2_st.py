import streamlit as st
import lightgbm as lgb
import pandas as pd
import numpy as np

def predict(data):
    # 保存したモデルを読み込み
    booster = lgb.Booster(model_file='model_horse/model.txt')
    X_test = data.drop(['着順','日付'], axis=1)
    y_pred = booster.predict(X_test)

    return y_pred

def processing(data):
    data['着順'] = data['着順'].map(lambda x: 1 if x<4 else 0)
    count_n_horse = data['race_id'].value_counts()
    data['出走頭数'] = data['race_id'].map(count_n_horse)
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

    uploaded_file = st.file_uploader("テストデータのCSVファイルをアップロードしてください", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, encoding='Shift-JIS')
        data = processing(data)
        y_test = data['着順']
        total_cases = len(y_test)  # テストデータの総数
        y_pred = predict(data)
        
        # thresholdをスライダーで指定
        threshold = st.slider('Thresholdを設定してください', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        df = pd.DataFrame({'予測': y_pred, '予測（二値）': y_pred >= threshold ,'着順': data['着順']})

        TP = (y_test == 1) & (y_pred >= threshold)  # True positives
        FP = (y_test == 0) & (y_pred >= threshold)  # False positives
        TN = (y_test == 0) & (y_pred < threshold)  # True negatives
        FN = (y_test == 1) & (y_pred < threshold)  # False negatives

        TP_count = sum(TP)
        FP_count = sum(FP)
        TN_count = sum(TN)
        FN_count = sum(FN)

        accuracy_TP = TP_count / total_cases * 100
        misclassification_rate_FP = FP_count / total_cases * 100
        accuracy_TN = TN_count / total_cases * 100
        misclassification_rate_FN = FN_count / total_cases * 100

        precision = TP_count / (TP_count + FP_count)


        # 予測結果を表示
        st.header("予測結果")
        st.write(df)
        st.write("Total cases:", total_cases)
        st.write("True positives:", TP_count, "(", "{:.2f}".format(accuracy_TP), "%)")
        st.write("False positives:", FP_count, "(", "{:.2f}".format(misclassification_rate_FP), "%)")
        st.write("True negatives:", TN_count, "(", "{:.2f}".format(accuracy_TN), "%)")
        st.write("False negatives:", FN_count, "(", "{:.2f}".format(misclassification_rate_FN), "%)")

        st.write("precision:", precision * 100, "%")


if __name__ == '__main__':
    main()
