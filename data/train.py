import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import joblib

print("正在读取数据...")
X = pd.read_csv('data/X_all.csv')
y = pd.read_csv('data/y_all.csv')

print(f"X 维度: {X.shape}")
print(f"y 维度: {y.shape}")

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 1. Random Forest
print("\n训练 Random Forest...")
rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
rf_scores = cross_val_score(rf, X, y, cv=kf, scoring='r2')
print(f"RF  -> 平均 R²: {rf_scores.mean():.4f}")

# 2. XGBoost
print("\n训练 XGBoost...")
xgb_model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, random_state=42))
xgb_scores = cross_val_score(xgb_model, X, y, cv=kf, scoring='r2')
print(f"XGB -> 平均 R²: {xgb_scores.mean():.4f}")

# 3. SVR
print("\n训练 SVR...")
svr = MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.1))
svr_scores = cross_val_score(svr, X, y, cv=kf, scoring='r2')
print(f"SVR -> 平均 R²: {svr_scores.mean():.4f}")

# 选出最好的模型
best_model = None
best_name = ""
best_score = -np.inf

if rf_scores.mean() > best_score:
    best_score = rf_scores.mean()
    best_model = rf
    best_name = "Random Forest"
if xgb_scores.mean() > best_score:
    best_score = xgb_scores.mean()
    best_model = xgb_model
    best_name = "XGBoost"
if svr_scores.mean() > best_score:
    best_score = svr_scores.mean()
    best_model = svr
    best_name = "SVR"

print(f"\n最优模型：{best_name} (R²={best_score:.4f})")
print("正在用全部数据训练最优模型...")
best_model.fit(X, y)
joblib.dump(best_model, 'models/best_model.pkl')
print("最优模型已保存到 models/best_model.pkl")