import pandas as pd
import joblib

print("正在加载模型...")
model = joblib.load('models/best_model.pkl')

print("请输入新配方的参数：")
mn = input("Mn2+ (输入 + 或 -): ")
ra = input("RA_conc (输入 + 或 -): ")
ph = float(input("Preparation_pH (输入 7.4 或 5.5): "))
poly_a = input("Polymer_A (输入 OHA 或 OHA-DAB): ")
poly_b = input("Polymer_B (输入 CMC): ")

# 1. 把 + 和 - 转换成 1 和 0
mn_val = 1 if mn == '+' else 0
ra_val = 1 if ra == '+' else 0
poly_a_val = 1 if poly_a == 'OHA-DAB' else 0
poly_b_val = 1 if poly_b == 'CMC' else 0

# 2. 构造输入数据
# 注意：代码里用的是你X_all.csv的列名
new_data = pd.DataFrame([[mn_val, ra_val, ph, poly_a_val, poly_b_val]], 
                        columns=['Mn2+', 'RA_conc', 'Preparation_pH', 'Polymer_A_OHA-DAB', 'Polymer_B_CMC'])

print("\n正在预测...")
prediction = model.predict(new_data)

print(f"\n预测结果（Polymer_A_OHA 的值）：{prediction[0][0]:.4f}")
print("模型预测完成！")