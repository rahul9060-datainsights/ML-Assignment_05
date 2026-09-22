import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    'Study_Hours': [1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                    6, 7, 7, 8, 8, 9, 9, 10, 10, 11],
    'Attendance': [50, 55, 60, 62, 65, 68, 70, 72, 75, 78,
                   80, 82, 84, 86, 88, 90, 92, 94, 95, 97],
    'Pass': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance']]
y = df['Pass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Results:")
print(y_test.values)

print("\nPredicted Results:")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=['Fail', 'Pass']
))

new_student = pd.DataFrame({
    'Study_Hours': [6],
    'Attendance': [80]
})

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("\nNew Student Prediction: Pass")
else:
    print("\nNew Student Prediction: Fail")