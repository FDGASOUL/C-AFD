# import pandas as pd
# from sklearn.feature_selection import mutual_info_classif
#
# # 读取CSV文件
# df = pd.read_csv('data/earthquake.csv')
#
# # 假设所有列都是分类变量，首先需要将数据转换为适合计算互信息的形式
# # 你可以根据需要选择合适的列
# # 例如，如果你要计算每两列之间的互信息
# columns = df.columns  # 获取所有列名
#
# # 计算所有列与目标列之间的互信息
# # 假设我们选择 'target' 列作为目标变量，其他列为特征
# X = df.drop('Burglary', axis=1)  # 删除目标列，其他列作为特征
# y = df['Burglary']  # 目标列
#
# # 计算特征与目标列的互信息
# mi = mutual_info_classif(X, y)
#
# # 显示结果
# mi_dict = {column: mi[i] for i, column in enumerate(X.columns)}
# print("各特征与目标变量的互信息：")
# for feature, mi_value in mi_dict.items():
#     print(f"{feature}: {mi_value}")


# import pandas as pd
# from sklearn.feature_selection import mutual_info_classif
# from sklearn.preprocessing import LabelEncoder
# import numpy as np
#
# # 读取CSV文件
# df = pd.read_csv('data/earthquake.csv')
#
# # 使用LabelEncoder将字符串类型的类别数据转换为数值
# label_encoder = LabelEncoder()
#
# # 对每一列应用LabelEncoder（假设每列都是分类变量）
# for column in df.columns:
#     if df[column].dtype == 'object':  # 检查是否是字符串类型
#         df[column] = label_encoder.fit_transform(df[column])
#
# # 计算每对列之间的互信息
# mi_matrix = np.zeros((len(df.columns), len(df.columns)))
#
# for i, col1 in enumerate(df.columns):
#     for j, col2 in enumerate(df.columns):
#         if i != j:
#             # 计算每一对列之间的互信息
#             mi_matrix[i, j] = mutual_info_classif(df[col1].values.reshape(-1, 1), df[col2])[0]
#
# # 打印互信息矩阵
# mi_df = pd.DataFrame(mi_matrix, columns=df.columns, index=df.columns)
# print("列与列之间的互信息矩阵：")
# print(mi_df)


import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 读取CSV文件
df = pd.read_csv('data/cancer.csv')

# 使用LabelEncoder将字符串类型的类别数据转换为数值
label_encoder = LabelEncoder()

# 对每一列应用LabelEncoder（假设每列都是分类变量）
for column in df.columns:
    if df[column].dtype == 'object':  # 检查是否是字符串类型
        df[column] = label_encoder.fit_transform(df[column])

# 计算每对列之间的归一化互信息
mi_matrix = np.zeros((len(df.columns), len(df.columns)))

for i, col1 in enumerate(df.columns):
    for j, col2 in enumerate(df.columns):
        if i != j:
            # 计算每一对列之间的归一化互信息
            mi_matrix[i, j] = normalized_mutual_info_score(df[col1], df[col2])

# 打印归一化互信息矩阵
mi_df = pd.DataFrame(mi_matrix, columns=df.columns, index=df.columns)
print("列与列之间的归一化互信息矩阵：")
print(mi_df)





