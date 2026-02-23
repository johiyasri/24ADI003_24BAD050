print("24BAD050-JOHIYA SRI S")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\Sibiya Shri\Documents\python\1\train_u6lujuX_CVtuZ9i (1).csv")

data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
data['ApplicantIncome'] = data['ApplicantIncome'].fillna(data['ApplicantIncome'].median())
data['Education'] = data['Education'].fillna(data['Education'].mode()[0])
data['Property_Area'] = data['Property_Area'].fillna(data['Property_Area'].mode()[0])
data['Loan_Status'] = data['Loan_Status'].fillna(data['Loan_Status'].mode()[0])

features = ['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Education', 'Property_Area']
X = data[features]
y = data['Loan_Status']

X = pd.get_dummies(X, drop_first=True)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Decision Tree")
plt.show()

plt.figure(figsize=(15,8))
plot_tree(dt, feature_names=X.columns, class_names=encoder.classes_, filled=True)
plt.show()

importances = pd.Series(dt.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure()
importances.plot(kind='bar')
plt.title("Feature Importance")
plt.show()

depth_values = range(1, 21)
train_scores = []
test_scores = []

for depth in depth_values:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.figure()
plt.plot(depth_values, train_scores)
plt.plot(depth_values, test_scores)
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Shallow vs Deep Tree Performance")
plt.legend(["Train Accuracy", "Test Accuracy"])
plt.show()
