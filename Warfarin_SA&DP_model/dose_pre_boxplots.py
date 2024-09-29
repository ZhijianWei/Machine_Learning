import matplotlib.pyplot as plt

# 假设你有三个模型的评估指标值，每个模型的指标值存储在列表中
# 例如：model_metrics = [RFR_r2, RFR_rmse, RFR_mae, SVR_r2, SVR_rmse, SVR_mae, ...]
# 这里我们创建一些示例数据
RFR_r2 = [0.55193052,0.70383386,0.6499401,0.60842416,0.65083358]
RFR_rmse = [0.51983059,0.43305314, 0.47568603, 0.49024219, 0.39623821]
RFR_mae = [0.42010514, 0.34669416, 0.33019164, 0.38695052, 0.34260846]

SVR_r2 = [0.55066265, 0.47183905, 0.62328313, 0.71124337, 0.56313855]
SVR_rmse = [0.52056553, 0.57830457, 0.49346545, 0.42098704, 0.44321248]
SVR_mae = [0.39893759, 0.3899056,  0.34803161, 0.33690821, 0.36100698]

XGB_r2 = [0.54726156, 0.64540918, 0.58370096, 0.57894098, 0.61533709]
XGB_rmse = [0.52253193, 0.47384601, 0.5187426, 0.50836334, 0.41589172]
XGB_mae = [0.39034833, 0.364799,  0.38380177, 0.42053918, 0.32601724]

SM_r2 = [0.57563268, 0.7289919,  0.64518995, 0.6397432,  0.65173867]
SM_rmse = [0.50589468, 0.414252,   0.47890257, 0.47022838, 0.39572432]
SM_mae = [0.40538116, 0.30624072, 0.32527658, 0.37916536, 0.33590718]

# 将所有模型的指标值合并到一个大列表中
# 为了箱型图的清晰展示，我们将R2、RMSE和MAE分别存储在不同的列表中
# all_r2 = RFR_r2 + SVR_r2 + XGB_r2 + SM_r2
# all_rmse = RFR_rmse + SVR_rmse+ XGB_rmse + SM_rmse
# all_mae = RFR_mae + SVR_mae+XGB_mae + SM_mae


# 创建r2箱型图
plt.figure(figsize=(10, 6))  # 设置图表大小
plt.boxplot([RFR_r2, SVR_r2, XGB_r2,SM_r2], vert=True, labels=['RFR', 'SVR', 'XGBoost','Stacking'])
plt.title('R^2 Comparison')
plt.xlabel('')
plt.ylabel('')
plt.show()

# 创建RMSE箱型图
plt.figure(figsize=(10, 6))
plt.boxplot([RFR_rmse, SVR_rmse, XGB_rmse, SM_rmse], vert=True, labels=['RFR', 'SVR', 'XGBoost', 'Stacking'])
plt.title('RMSE Comparison')
plt.xlabel('')
plt.ylabel('')
plt.show()

# 创建MAE箱型图
plt.figure(figsize=(10, 6))
plt.boxplot([RFR_mae, SVR_mae, XGB_mae, SM_mae], vert=True, labels=['RFR', 'SVR', 'XGBoost', 'Stacking'])
plt.title('MAE Comparison')
plt.xlabel('')
plt.ylabel('')
plt.show()

# 改成函数
# def plot_boxplot(metric_name, models_data):
#     """
#     创建并显示箱型图，比较不同模型的给定评估指标。
#
#     参数:
#     metric_name (str): 评估指标的名称，用于设置图表标题。
#     models_data (list of lists): 包含每个模型的评估指标数据的列表。
#     """
#     # 创建箱型图
#     plt.figure(figsize=(10, 6))
#     plt.boxplot(models_data, vert=True, labels=models_names)  # models_names 应该是包含模型名称的列表
#
#     # 设置标题和轴标签
#     plt.title(f'{metric_name} Comparison')
#     plt.xlabel('Model')
#     plt.ylabel(metric_name)
#
#     # 显示图表
#     plt.show()
#
# # 假设你已经有了每个模型的评估指标数据
# RFR_r2 = [0.6, 0.65, 0.62, 0.64, 0.63]
# SVR_r2 = [0.55, 0.58, 0.57, 0.59, 0.56]
# XGB_r2 = [0.7, 0.69, 0.72, 0.71, 0.73]
# Stacking_r2 = [0.75, 0.73, 0.74, 0.76, 0.72]
#
# # 调用函数来创建R^2箱型图
# plot_boxplot('R^2', [RFR_r2, SVR_r2, XGB_r2, Stacking_r2])
#
# # 假设你还有RMSE和MAE的数据
# RFR_rmse = [1.1, 1.08, 1.12, 1.05, 1.11]
# SVR_rmse = [1.4, 1.38, 1.42, 1.35, 1.43]
# XGB_rmse = [0.9, 0.89, 0.92, 0.88, 0.93]
# Stacking_rmse = [0.8, 0.82, 0.83, 0.85, 0.81]
#
# RFR_mae = [0.8, 0.78, 0.79, 0.81, 0.77]
# SVR_mae = [1.0, 0.98, 1.02, 0.99, 1.03]
# XGB_mae = [0.7, 0.69, 0.72, 0.68, 0.73]
# Stacking_mae = [0.6, 0.62, 0.63, 0.65, 0.64]
#
# # 调用函数来创建RMSE箱型图
# plot_boxplot('RMSE', [RFR_rmse, SVR_rmse, XGB_rmse, Stacking_rmse])
#
# # 调用函数来创建MAE箱型图
# plot_boxplot('MAE', [RFR_mae, SVR_mae, XGB_mae, Stacking_mae])