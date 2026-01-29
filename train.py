import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score
import lightgbm as lgb
import joblib

# 1. 读取数据
df = pd.read_csv("dataset.csv")

# 2. 特征和标签
features = ['pv_count', 'cart_count', 'fav_count']
X = df[features]
y = df['is_buy']

# 3. 数据集切分（随机 + 分层）
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. 逻辑回归
lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)
lr.fit(X_train, y_train)

# 5. 评估逻辑回归
y_prob_lr = lr.predict_proba(X_test)[:, 1]
y_pred_lr = lr.predict(X_test)

auc_lr = roc_auc_score(y_test, y_prob_lr)
recall_lr = recall_score(y_test, y_pred_lr)

print("LR AUC:", auc_lr)
print("LR Recall:", recall_lr)

# 6. LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    class_weight='balanced'
)
lgb_model.fit(X_train, y_train)

# 7. 评估 LightGBM
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
y_pred_lgb = lgb_model.predict(X_test)

auc_lgb = roc_auc_score(y_test, y_prob_lgb)
recall_lgb = recall_score(y_test, y_pred_lgb)

print("LGB AUC:", auc_lgb)
print("LGB Recall:", recall_lgb)

# 8. 保存模型
joblib.dump(lgb_model, "model.pkl")

