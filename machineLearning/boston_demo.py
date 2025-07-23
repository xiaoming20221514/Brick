import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import Ridge  # 导入岭回归

from sklearn.linear_model import Lasso
from sklearn.datasets import load_wine

data = load_wine()
x,y = data.data, data.target
print(x)
print(y)
#
#
# # 1. 读取数据
# def load_data(file_path):
#     """读取 CSV 文件并返回 DataFrame"""
#     return pd.read_csv(file_path)
#
# # 2. 预处理（删除缺失值）
# def preprocess_data(data):
#     """检查并删除缺失值"""
#     if data.isnull().sum().any():
#         print("数据中存在缺失值，正在删除缺失值...")
#         data = data.dropna()  # 删除缺失值
#         print(f"删除后数据行数: {len(data)}")
#     else:
#         print("数据中没有缺失值。")
#     return data
#
# # 3. 提取自变量和因变量
# def extract_features_target(data, feature_col, target_col):
#     """提取自变量和因变量"""
#     X = data[feature_col].values.reshape(-1, 1)  # 自变量
#     y = data[target_col].values  # 因变量
#     # 检查是否存在 NaN
#     if np.isnan(X).any() or np.isnan(y).any():
#         raise ValueError("自变量或因变量中仍然存在 NaN 值，请检查数据。")
#     return X, y
#
# # 4. 绘制散点图
# def plot_scatter(X, y, xlabel, ylabel, title):
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei 是黑体
#     plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
#     """绘制散点图"""
#     plt.figure(figsize=(8, 6))
#     plt.scatter(X, y, alpha=0.5)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.show()
#
# # 5. 使用 numpy 重构自变量（最高 3 次方 + 常数项）
# def transform_features_numpy(X):
#     """使用 numpy 重构自变量"""
#     X_poly = np.column_stack((X**3, X**2, X, np.ones_like(X)))  # 3次方 + 2次方 + 1次方 + 常数项
#     return X_poly
#
# # 5. 替代方案：使用 sklearn 的 PolynomialFeatures
# def transform_features_sklearn(X):
#     """使用 sklearn 的 PolynomialFeatures 重构自变量"""
#     poly = PolynomialFeatures(degree=3, include_bias=True)
#     X_poly = poly.fit_transform(X)
#     return X_poly
#
# # 6. 拆分数据
# def split_data(X, y, test_size=0.2, random_state=42):
#     """拆分数据集为训练集和测试集"""
#     return train_test_split(X, y, test_size=test_size, random_state=random_state)
#
# # # 7. 训练模型
# # def train_model(X_train, y_train):
# #     """训练线性回归模型"""
# #     model = LinearRegression()
# #     model.fit(X_train, y_train)
# #     return model
#
# # 7. 训练模型（修改为岭回归）
# def train_model(X_train, y_train, alpha=1.0):
#     """训练岭回归模型"""
#     model = Ridge(alpha=alpha)  # 设置正则化参数 alpha
#     model.fit(X_train, y_train)
#     return model
#
# # 7. 训练模型（修改为岭回归）
# def train_model_Lasso(X_train, y_train, alpha=1.0):
#     """训练岭回归模型"""
#     model = Lasso(alpha=alpha)  # 设置正则化参数 alpha
#     model.fit(X_train, y_train)
#     return model
#
# # 8. 评估模型
# def evaluate_model(model, X_test, y_test):
#     """评估模型并输出结果"""
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     print(f"均方误差 (MSE): {mse:.2f}")
#     print(f"决定系数 (R²): {r2:.2f}")
#
# # 9. 绘制折线图
# def plot_regression_line(X, y, model, xlabel, ylabel, title):
#     """绘制回归线"""
#     plt.figure(figsize=(8, 6))
#     plt.scatter(X, y, alpha=0.5, label='实际值')
#     X_sorted = np.sort(X, axis=0)
#     y_pred = model.predict(transform_features_numpy(X_sorted))
#     plt.plot(X_sorted, y_pred, color='red', label='预测值')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.legend()
#     plt.show()
#
# # 主流程
# def main():
#     # 1. 读取数据
#     data = load_data('boston_housing.csv')
#
#     # 2. 预处理
#     data = preprocess_data(data)
#
#     # 3. 提取自变量和因变量
#     X, y = extract_features_target(data, 'LSTAT', 'MEDV')
#
#     # 4. 绘制散点图
#     plot_scatter(X, y, 'LSTAT', 'MEDV', 'LSTAT vs MEDV 散点图')
#
#     # 5. 重构自变量（选择方案）
#     # 方案 1: 使用 numpy
#     X_poly = transform_features_numpy(X)
#
#     # 方案 2: 使用 sklearn
#     # X_poly = transform_features_sklearn(X)
#
#     # 6. 拆分数据
#     X_train, X_test, y_train, y_test = split_data(X_poly, y)
#
#     # 7. 训练模型
#     # model = train_model(X_train, y_train)
#     alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
#     for alpha in alphas:
#         # model = train_model(X_train, y_train, alpha=alpha)
#         model = train_model_Lasso(X_train, y_train, alpha=alpha)
#         evaluate_model(model, X_test, y_test)
#
#     # 8. 评估模型
#     evaluate_model(model, X_test, y_test)
#
#     # 9. 绘制折线图
#     plot_regression_line(X, y, model, 'LSTAT', 'MEDV', '多项式回归预测')
#
# # 运行主流程
# if __name__ == "__main__":
#     main()