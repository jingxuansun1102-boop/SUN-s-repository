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
        if prob > 0.5:
        st.write(f"该用户购买概率 {prob:.2%}，建议推送优惠券")
    else:
        st.write(f"购买概率：{prob:.2%}")

