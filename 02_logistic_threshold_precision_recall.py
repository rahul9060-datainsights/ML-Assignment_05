import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

data = {
    'Study_Hours': [
        1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
        6, 7, 7, 8, 8, 9, 9, 10, 10, 11,
        3, 4, 5, 6, 7, 8, 9, 5, 6, 8
    ],
    'Attendance': [
        45, 50, 55, 60, 65, 70, 75, 80, 85, 90,
        95, 50, 55, 60, 65, 70, 75, 80, 85, 90,
        55, 65, 75, 85, 60, 70, 80, 90, 50, 95
    ],
    'Pass': [
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 1, 1, 0, 1, 1, 0, 0, 1
    ]
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance']]
y = df['Pass']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]

thresholds = [0.3, 0.5, 0.7]

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print("\nThreshold:", threshold)
    print("Predicted Results:", y_pred)
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))