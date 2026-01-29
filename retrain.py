import pandas as pd
import joblib

# 1. 加载模型
model = joblib.load("model.pkl")

# 2. 读取用户行为数据
df = pd.read_csv("dataset.csv")

# 3. 选择模型需要的特征
X = df[["pv", "cart", "fav"]]

# 4. 预测购买概率（取为 1 的概率）
df["purchase_prob"] = model.predict_proba(X)[:, 1]

# 5. 保存预测结果
df.to_csv("predict_result.csv", index=False, encoding="utf-8-sig")

print("批量预测完成，结果已保存为 predict_result.csv")
