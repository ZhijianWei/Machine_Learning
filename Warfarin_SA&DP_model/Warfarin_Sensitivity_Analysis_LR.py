
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 读取Excel文件
filepath = r"C:\Users\XavierWynne\Desktop\月月毕业论文\病人敏感性分析\normalized_data.xlsx"
data = pd.read_excel(filepath)

# 分离特征和目标变量
X = data.iloc[:, 1:]  # 所有行，除了第一列的所有列
y = data.iloc[:, 0]   # 所有行，第一列

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建逻辑回归模型并拟合
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# 预测概率
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# 生成ROC曲线值
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Specificity')
plt.ylabel('Sensitivity')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
