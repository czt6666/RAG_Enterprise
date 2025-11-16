from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.1, random_state=18)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_test_predict = clf.predict(X_test)
print(classification_report(y_test, y_test_predict, target_names=data.target_names))
print(y_test_predict)
print(y_test)
