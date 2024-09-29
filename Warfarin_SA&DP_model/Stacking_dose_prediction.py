import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

file_path = r'C:\Users\XavierWynne\Desktop\月月毕业论文\病人敏感性分析\1204all.xlsx'
data = pd.read_excel(file_path)

# Identifying categorical columns for one-hot encoding
categorical_cols = data.select_dtypes(include=['object']).columns

# Applying one-hot encoding to categorical columns
encoder = OneHotEncoder()
data_encoded = pd.get_dummies(data, columns=categorical_cols)

# Selecting specific features
selected_features = ['Uridine', 'VKORC1', 'BSA', 'ALT', 'Age', 'TBil', 'INRday0', 'Y=Dose-mean']

# Filtering the dataset to include only the selected features
data_selected = data[selected_features]

# Splitting the dataset into training and testing sets based on the specified rows
train_data = data_encoded.iloc[1:179]
test_data = data_encoded.iloc[179:]
X_train = train_data.drop(columns=['Y=Dose-mean'])
y_train = train_data['Y=Dose-mean']
X_test = test_data.drop(columns=['Y=Dose-mean'])
y_test = test_data['Y=Dose-mean']

# Defining base models for stacking
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('gbr', GradientBoostingRegressor(n_estimators=100))
]

# Defining the meta-model
meta_model = Ridge()

# Setting up the parameter grid for GridSearchCV0·
param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'gbr__n_estimators': [100, 200, 300],
    'final_estimator__alpha': [0.1, 1, 10]
}

# Creating and fitting the GridSearchCV for StackingRegressor
grid_search = GridSearchCV(estimator=StackingRegressor(estimators=base_models, final_estimator=meta_model), param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_

# Training the stacking model with the best parameters
best_model = StackingRegressor(estimators=base_models, final_estimator=Ridge(alpha=best_params['final_estimator__alpha']))
best_model.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculating accuracy
percentage_error = ((y_pred - y_test) / y_test) * 100
accurate_predictions = np.sum(np.abs(percentage_error) <= 25)
total_predictions = len(y_pred)
accuracy = (accurate_predictions / total_predictions) * 100
mae = mean_absolute_error(y_test, y_pred)
accuracy = np.sum(np.abs((y_pred - y_test) / y_test) <= 0.25) / len(y_pred) * 100
print('准确预测值 （±25%):', accuracy)

# Calculating overestimation and underestimation rates
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


# 执行五折交叉验证
scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
cv_results = cross_validate(best_model, X_train, y_train, cv=5, scoring=scoring)

# 提取每个折上的评估指标值
r2_scores = cv_results['test_r2']
mse_scores = -cv_results['test_neg_mean_squared_error']  # 注意这里取负值是因为cross_validate返回的是负的MSE
mae_scores = -cv_results['test_neg_mean_absolute_error']  # 注意这里取负值是因为cross_validate返回的是负的MAE

# 计算RMSE
rmse_scores = np.sqrt(mse_scores)

# 打印每个折上的评估指标值
print("5-Fold Cross-Validation Results:")
print("R2 Scores:", r2_scores)
print("RMSE Scores:", rmse_scores)
print("MAE Scores:", mae_scores)

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

#在训练集上测试性能
# # Predicting on the training set using the best model
# y_pred_train = best_model.predict(X_train)
#
# # Calculating model performance on the training set
# mse_train = mean_squared_error(y_train, y_pred_train)
# r2_train = r2_score(y_train, y_pred_train)
#
# # Calculating accuracy on the training set
# percentage_error_train = ((y_pred_train - y_train) / y_train) * 100
# accurate_predictions_train = np.sum(np.abs(percentage_error_train) <= 20)
# total_predictions_train = len(y_pred_train)
# accuracy_train = (accurate_predictions_train / total_predictions_train) * 100
#
# print('Mean Squared Error on Training Set:', mse_train)
# print('R^2 Score on Training Set:', r2_train)
# print('Accuracy on Training Set:', accuracy_train)