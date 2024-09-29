import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

file_path = r'C:\Users\XavierWynne\Desktop\月月毕业论文\病人敏感性分析\1204all.xlsx'
data = pd.read_excel(file_path)

# 独热编码
categorical_cols = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_cols)

selected_features = ['Uridine', 'VKORC1', 'BSA', 'ALT', 'Age', 'TBil', 'INRday0', 'Y=Dose-mean']
data_selected = data[selected_features]

# 划分训练集和测试集
train_data = data_encoded.iloc[1:179]
test_data = data_encoded.iloc[179:]
X_train = train_data.drop(columns=['Y=Dose-mean'])
y_train = train_data['Y=Dose-mean']
X_test = test_data.drop(columns=['Y=Dose-mean'])
y_test = test_data['Y=Dose-mean']

# 网格搜索
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# 用最优参数训练
best_model = RandomForestRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# 模型评价
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 计算准确度
percentage_error = ((y_pred - y_test) / y_test) * 100
accurate_predictions = np.sum(np.abs(percentage_error) <= 25)
total_predictions = len(y_pred)
accuracy = (accurate_predictions / total_predictions) * 100
mae = mean_absolute_error(y_test, y_pred)

accuracy = np.sum(np.abs((y_pred - y_test) / y_test) <= 0.25) / len(y_pred) * 100
print('准确预测值 （±25%):', accuracy)

# 计算高估和低估
overestimated = np.sum((y_pred - y_test) / y_test > 0.25)
underestimated = np.sum((y_pred - y_test) / y_test < -0.25)
overestimation_rate = (overestimated / len(y_pred)) * 100
underestimation_rate = (underestimated / len(y_pred)) * 100

print('高估预测率:', overestimation_rate)
print('低估预测率:', underestimation_rate)
print('MSE:', mse)
print('R^2 Score:', r2)
# print('Accuracy:', accuracy)
print('MAE:', mae)

# print('Best Parameters:', best_params)
# print('Mean Squared Error:', mse)
# print('R^2 Score:', r2)
# print('Accuracy:', accuracy)
# print('MAE:', mae)
#
# # 计算理想预测率，这里我们定义理想预测为误差在20%以内
# error_percentage = ((y_pred - y_test) / y_test) * 100
# ideal_predictions = np.sum(np.abs(error_percentage) <= 25)
# total_predictions = len(y_pred)
# ideal_prediction_rate = (ideal_predictions / total_predictions) * 100
# print('理想预测率:', ideal_prediction_rate)
#
# # 计算高估预测率和低估预测率
# overestimated = np.sum(y_pred > y_test)
# underestimated = np.sum(y_pred < y_test)
# overestimation_rate = (overestimated / len(y_pred)) * 100
# underestimation_rate = (underestimated / len(y_pred)) * 100
# print('高估预测率:', overestimation_rate)
# print('低估预测率:', underestimation_rate)


# 执行交叉验证
cv_results = cross_validate(estimator=best_model, X=X_train, y=y_train, cv=5, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'accuracy'))

# 提取每个折上的评估指标值
r2_scores = cv_results['test_r2']
mse_scores = -cv_results['test_neg_mean_squared_error']  # 注意这里取负值是因为cross_validate返回的是负的MSE
mae_scores = -cv_results['test_neg_mean_absolute_error']  # 注意这里取负值是因为cross_validate返回的是负的MAE
accuracy_scores = cv_results['test_accuracy']

# 打印每个折上的评估指标值
print("5-Fold Cross-Validation Results:")
print("R2 Scores:", r2_scores)
print("RMSE Scores (MSE):", np.sqrt(mse_scores))  # 将MSE转换为RMSE
print("MSE Scores:", mse_scores)
print("MAE Scores:", mae_scores)
print("Accuracy Scores:", accuracy_scores)