from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# 读取鸢尾花数据
data = load_iris()
# # 对数据进行分割，训练集和测试集比例为9:1
X_train, X_test, y_train, y_test =     train_test_split(data.data,
             data.target,
             test_size=0.1,
             stratify=data.target, # 确保训练集和测试集标签分布一致
             random_state=18)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 训练模型
clf = LogisticRegression()
clf.fit(X_train, y_train)
print(clf)
# 预测
y_test_pred = clf.predict(X_test)
# 模型预测效果评估
print(classification_report(y_test, y_test_pred,
                                 target_names=data.target_names))
print(y_test)
print(y_test_pred)