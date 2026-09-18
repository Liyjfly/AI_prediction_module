import pandas as pd
import numpy as np
import os

print("正在读取数据...")
df = pd.read_csv('data/formulation_dataset.csv')

# 1. 把 + 和 - 转换成 1 和 0
df['Mn2+'] = df['Mn2+'].replace({'+': 1, '-': 0})
df['RA_conc'] = df['RA_conc'].replace({'+': 1, '-': 0})

# 2. 文字列转换为数字
df = pd.get_dummies(df, columns=['Polymer_A', 'Polymer_B'])

# 3. ⚠️ 直接手动指定：输入列和输出列
input_cols = ['Mn2+', 'RA_conc', 'Preparation_pH', 'Polymer_A_OHA-DAB', 'Polymer_B_CMC']

# 输出列 = 所有列 - 输入列 - Candidate_ID
# 不检查名字，直接减，只要在 df 里，就自动加入
output_cols = [c for c in df.columns if c not in input_cols and c != 'Candidate_ID']

print(f"输入列（{len(input_cols)}个）：", input_cols)
print(f"输出列（{len(output_cols)}个）：", output_cols)

# 4. 处理缺失值和异常值
for c in input_cols + output_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    df[c] = df[c].fillna(df[c].median())
    median = df[c].median()
    df[c] = df[c].apply(lambda x: median if abs(x) > 1e10 else x)

# 5. 分离输入和输出
X = df[input_cols]
y = df[output_cols]

print(f"输入特征维度: {X.shape}")
print(f"输出指标维度: {y.shape}")

# 6. 保存
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
X.to_csv('data/X_all.csv', index=False)
y.to_csv('data/y_all.csv', index=False)

print("预处理完成！数据已保存到 data/X_all.csv 和 data/y_all.csv")