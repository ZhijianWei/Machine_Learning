import pandas as pd
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # Set font to SimHei
plt.rcParams['axes.unicode_minus'] = False  # Enable display of the negative sign

# Load the dataset
file_path = r"C:\Users\XavierWynne\Desktop\月月毕业论文\病人敏感性分析\1204all.xlsx"
data = pd.read_excel(file_path)

# Selecting specific features
X = data[['Uridine', 'VKORC1', 'BSA', 'Amiodarone', 'Strok', 'Age', 'Diabetes']]
y = data['Y=Dose-mean']

start_time = time.time()

# Setting up the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10, 30, 50],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}

# Creating and fitting the GridSearchCV for RandomForestRegressor
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X, y)

# Best parameters
best_params = grid_search.best_params_

# Training the model with the best parameters
best_model = RandomForestRegressor(**best_params, random_state=42)
best_model.fit(X, y)

# Predicting and evaluating the model
y_pred = best_model.predict(X)

# Creating a SHAP Explainer object
explainer = shap.Explainer(best_model)

# Calculating SHAP values
shap_values = explainer.shap_values(X)

# Ending the timer
end_time = time.time()

# Outputting execution time
execution_time = end_time - start_time
print('Model training execution time: {:.2f} seconds'.format(execution_time))

# Plotting the SHAP summary plot
shap.summary_plot(shap_values, X, show=False)
plt.title('SHAP Summary Plot for RandomForest Model')
plt.show()

# Plotting the feature importance bar chart
shap.summary_plot(shap_values, X, plot_type='bar', show=False)
plt.title('Feature Importance Bar Chart for RandomForest Model')
plt.show()