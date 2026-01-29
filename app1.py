import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("用户购买概率预测")

pv = st.number_input("PV 次数", 0)
cart = st.number_input("加购次数", 0)
fav = st.number_input("收藏次数", 0)

if st.button("预测"):
    X = pd.DataFrame([[pv, cart, fav]],
                     columns=['pv_count', 'cart_count', 'fav_count'])
    prob = model.predict_proba(X)[0][1]
    st.write(f"购买概率：{prob:.2%}")
st.title("用户购买概率预测（批量模拟）")

uploaded_file = st.file_uploader("上传用户行为数据 CSV 文件", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("原始数据预览：")
    st.dataframe(df.head())

    st.write("检测到的列名：")
    st.write(list(df.columns))
    st.subheader("请选择特征列（需与模型含义一致）")

    pv_col = st.selectbox("选择 PV 次数列", df.columns)
    cart_col = st.selectbox("选择 加购次数列", df.columns)
    fav_col = st.selectbox("选择 收藏次数列", df.columns)
    if st.button("开始预测"):
        X = df[[pv_col, cart_col, fav_col]]

        probs = model.predict_proba(X)[:, 1]

        df["购买概率"] = probs

        st.success("预测完成 ✅")
        st.dataframe(df)

        st.download_button(
            "下载预测结果 CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name="prediction_result.csv",
            mime="text/csv"
        )