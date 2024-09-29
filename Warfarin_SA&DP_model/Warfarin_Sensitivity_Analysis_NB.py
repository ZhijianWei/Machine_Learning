import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score

data = pd.read_excel(r"C:\Users\XavierWynne\Desktop\月月毕业论文\病人敏感性分析\Total-RLogistic.xlsx")

# 分离特征和标签
X = data.iloc[:, 1:-1]  # 所有自变量，除了最后一列
y = data.iloc[:, 0]    # 目标变量（第一列）

# 将分类变量和连续变量分开处理
categorical_columns = X.columns[:-1]  # 除了最后一列的所有列
continuous_columns = [X.columns[-1]]  # 最后一列

# 初始化LabelEncoder和OneHotEncoder
le = LabelEncoder()
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns),
        ('cont', StandardScaler(), continuous_columns)
    ])

# 对分类变量进行One-Hot编码
X_encoded = preprocessor.fit_transform(X)

# 将编码后的数据转换为DataFrame
X_encoded_df = pd.DataFrame(X_encoded, columns=preprocessor.get_feature_names_out())

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y, test_size=0.2, random_state=42)
#
# # 创建朴素贝叶斯分类器实例
# gnb = GaussianNB()
#
# # 训练模型
# gnb.fit(X_train, y_train)
#
# # 进行预测
# y_pred = gnb.predict(X_test)

# # 评估模型
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# 如果你想要对整个数据集进行预测
# y_full_pred = gnb.predict(X_encoded_df)

# # 预测概率
# y_pred_prob = gnb.predict_proba(X_test)[:, 1]
#
# # 生成ROC曲线值
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# roc_auc = auc(fpr, tpr)
#
# # 绘制ROC曲线
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Specificity')
# plt.ylabel('Sensitivity')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()




# 定义性能指标函数
def performance_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tn + tp) / (tn + fp + fn + tp) if (tn + fp + tn + tp) > 0 else 0
    npv = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    return specificity, sensitivity, accuracy, npv, ppv, precision, f1

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y, test_size=0.2, random_state=42)

# 创建朴素贝叶斯分类器实例
gnb = GaussianNB()

# 训练模型
gnb.fit(X_train, y_train)

y_train_pred = gnb.predict(X_train)

# 使用混淆矩阵来计算TP, FP, TN, FN
conf_matrix = confusion_matrix(y_train, y_train_pred)

# 从混淆矩阵中提取TP, FP, TN, FN
tn, fp, fn, tp = conf_matrix.ravel()

# 计算所需的性能指标
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
npv = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0


print("Training Set Metrics:")
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)
print("Accuracy:", accuracy)
print("NPV:", npv)
print("PPV:", ppv)

# 计算并打印精确度和F1分数
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
print("Precision:", precision)

# 计算测试集的性能指标
y_test_pred = gnb.predict(X_test)
y_test_prob = gnb.predict_proba(X_test)[:, 1]
test_metrics = performance_metrics(y_test, y_test_pred, y_test_prob)
print("Test Set Metrics:")
print("Specificity:", test_metrics[0])
print("Sensitivity:", test_metrics[1])
print("Accuracy:", test_metrics[2])
print("NPV:", test_metrics[3])
print("PPV:", test_metrics[4])
print("Precision:", test_metrics[5])
print("F1 Score:", test_metrics[6])

# 五折交叉验证
cv = 5
gnb = GaussianNB()

# 计算五折交叉验证的性能指标
gnb.fit(X_train, y_train)
y_train_pred = cross_val_predict(gnb, X_train, y_train, cv=cv)
y_train_prob = gnb.predict_proba(X_train)[:, 1]
train_metrics_5fold = performance_metrics(y_train, y_train_pred, y_train_prob)
print(f"5-Fold CV Training Set Metrics:")
print("Specificity:", train_metrics_5fold[0])
print("Sensitivity:", train_metrics_5fold[1])
print("Accuracy:", train_metrics_5fold[2])
print("NPV:", train_metrics_5fold[3])
print("PPV:", train_metrics_5fold[4])
print("Precision:", train_metrics_5fold[5])
print("F1 Score:", train_metrics_5fold[6])

# 十折交叉验证
cv = 10
gnb = GaussianNB()

# 计算十折交叉验证的性能指标
gnb.fit(X_train, y_train)
y_train_pred = cross_val_predict(gnb, X_train, y_train, cv=cv)
y_train_prob = gnb.predict_proba(X_train)[:, 1]
train_metrics_10fold = performance_metrics(y_train, y_train_pred, y_train_prob)
print(f"10-Fold CV Training Set Metrics:")
print("Specificity:", train_metrics_10fold[0])
print("Sensitivity:", train_metrics_10fold[1])
print("Accuracy:", train_metrics_10fold[2])
print("NPV:", train_metrics_10fold[3])
print("PPV:", train_metrics_10fold[4])
print("Precision:", train_metrics_10fold[5])
print("F1 Score:", train_metrics_10fold[6])

# # 获取模型的预测概率（训练集和测试集）
# y_train_probs = gnb.predict_proba(X_train)[:, 1]
# y_test_probs = gnb.predict_proba(X_test)[:, 1]
#
# # 计算训练集的TPR和FPR
# fpr_train, tpr_train, _ = roc_curve(y_train, y_train_probs)
# # 计算测试集的TPR和FPR
# fpr_test, tpr_test, _ = roc_curve(y_test, y_test_probs)
# # 计算训练集和测试集的AUC
# roc_auc_train = auc(fpr_train, tpr_train)
# roc_auc_test = auc(fpr_test, tpr_test)
#
# # 绘制ROC曲线
# plt.figure()
# plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label='Training ROC (area = %0.2f)' % roc_auc_train)
# plt.plot(fpr_test, tpr_test, color='navy', lw=2, label='Test ROC (area = %0.2f)' % roc_auc_test)
# plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()
