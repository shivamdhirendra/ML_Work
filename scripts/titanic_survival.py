import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

ds_names = sns.get_dataset_names()

df = sns.load_dataset("titanic")

df.info()

df.drop(
    ["deck", "embark_town", "adult_male", "alive", "who", "class"], axis=1, inplace=True
)
df["age"] = df["age"].fillna(df["age"].mean(), inplace=True)
df.dropna(subset=["embarked"], inplace=True)

# label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df.head()
df["sex"] = le.fit_transform(df["sex"])
df["embarked"] = le.fit_transform(df["embarked"])  # C=0,Q=1,S=2

# convert dataset into integer
df = df.astype(int)

x = df.drop("survived", axis=1)
y = df["survived"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train, y_train)

y_prediction = model.predict(x_test)

# evaluating the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy_score(y_test, y_prediction)
# 0.8033707865168539

confusion_matrix(y_test, y_prediction)
# array([[90, 19],
#        [16, 53]])

print(classification_report(y_test, y_prediction))
#               precision    recall  f1-score   support
#
#            0       0.85      0.83      0.84       109
#            1       0.74      0.77      0.75        69
#
#     accuracy                           0.80       178
#    macro avg       0.79      0.80      0.79       178
# weighted avg       0.81      0.80      0.80       178


# KNN
# feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train_scaled, y_train)

y_pred_knn = knn_model.predict(x_test_scaled)

accuracy_score(y_test, y_pred_knn)
# 0.7808988764044944

confusion_matrix(y_test, y_pred_knn)
# array([[90, 19],
#        [20, 49]])

print(classification_report(y_test, y_pred_knn))
#               precision    recall  f1-score   support
#
#            0       0.82      0.83      0.82       109
#            1       0.72      0.71      0.72        69
#
#     accuracy                           0.78       178
#    macro avg       0.77      0.77      0.77       178
# weighted avg       0.78      0.78      0.78       178


# Naive Bayes
# prediction without using standard scaling
x_train
from sklearn.naive_bayes import GaussianNB

model_NB = GaussianNB()

model_NB.fit(x_train, y_train)
y_pred_NB = model_NB.predict(x_test)

accuracy_score(y_test, y_pred_NB)
# 0.7752808988764045

confusion_matrix(y_test, y_pred_NB)
# array([[84, 25],
#       [15, 54]])

print(classification_report(y_test, y_pred_NB))
#               precision    recall  f1-score   support
#
#            0       0.85      0.77      0.81       109
#            1       0.68      0.78      0.73        69
#
#     accuracy                           0.78       178
#    macro avg       0.77      0.78      0.77       178
# weighted avg       0.78      0.78      0.78       178


# Naive Bayes
# prediction with standard scaling
model_NB1 = GaussianNB()

model_NB1.fit(x_train_scaled, y_train)
y_pred_NB1 = model_NB1.predict(x_test_scaled)

accuracy_score(y_test, y_pred_NB1)
# 0.7528089887640449

confusion_matrix(y_test, y_pred_NB1)
# array([[89, 20],
#        [24, 45]])

print(classification_report(y_test, y_pred_NB1))
#               precision    recall  f1-score   support
#
#            0       0.79      0.82      0.80       109
#            1       0.69      0.65      0.67        69
#
#     accuracy                           0.75       178
#    macro avg       0.74      0.73      0.74       178
# weighted avg       0.75      0.75      0.75       178


# Decision Tree
from sklearn.tree import DecisionTreeClassifier

model_DT = DecisionTreeClassifier(random_state=42)

model_DT.fit(x_train_scaled, y_train)
y_pred_DT = model_DT.predict(x_test_scaled)

accuracy_score(y_test, y_pred_DT)
# 0.7696629213483146

confusion_matrix(y_test, y_pred_DT)
# array([[88, 21],
#        [20, 49]])

print(classification_report(y_test, y_pred_DT))
#               precision    recall  f1-score   support
#
#            0       0.81      0.81      0.81       109
#            1       0.70      0.71      0.71        69
#
#     accuracy                           0.77       178
#    macro avg       0.76      0.76      0.76       178
# weighted avg       0.77      0.77      0.77       178


# SVM
from sklearn.svm import SVC

model_svm = SVC(kernel="rbf")

model_svm.fit(x_train_scaled, y_train)
y_pred_svm = model_svm.predict(x_test_scaled)

accuracy_score(y_test, y_pred_svm)
# 0.8258426966292135

confusion_matrix(y_test, y_pred_svm)
# array([[96, 13],
#        [18, 51]])

print(classification_report(y_test, y_pred_svm))
#               precision    recall  f1-score   support
#
#            0       0.84      0.88      0.86       109
#            1       0.80      0.74      0.77        69
#
#     accuracy                           0.83       178
#    macro avg       0.82      0.81      0.81       178
# weighted avg       0.82      0.83      0.82       178


# Cross Validation
df
x = df.drop("survived", axis=1)
y = df["survived"]

from sklearn.model_selection import cross_val_score

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

scores = cross_val_score(model_svm, x_scaled, y, cv=5, scoring="accuracy")
# array([0.83146067, 0.82022472, 0.81460674, 0.80898876, 0.86440678])
print(scores.mean())
# 0.8279375357074844

scores = cross_val_score(knn_model, x_scaled, y, cv=5, scoring="accuracy")
print(scores.mean())
# 0.7986542245921411

scores = cross_val_score(model_DT, x_scaled, y, cv=5, scoring="accuracy")
print(scores.mean())
# 0.7975623690725577

scores = cross_val_score(model_NB, x_scaled, y, cv=5, scoring="accuracy")
print(scores.mean())
# 0.7840474830191074
