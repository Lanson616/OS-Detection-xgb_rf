import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load features and labels from the text file
feature_vectors_loaded = []
with open('28_feature_array.txt', 'r') as file:
    for line in file:
        feature_vector = list(map(float, line.strip().split(',')))
        feature_vectors_loaded.append(feature_vector)

with open('50250_os_system_label.txt', 'r') as file:
    selected_os_loaded = [line.strip() for line in file]

X = np.array(feature_vectors_loaded)
y = np.array(selected_os_loaded)

print(X.shape)
print(y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train the Random Forest model
XGB = xgb.XGBClassifier()
XGB.fit(X_train, y_train_encoded)

train_preds = XGB.predict(X_train)
test_preds = XGB.predict(X_test)

# Append predictions as new features
X_train_extended = np.hstack((X_train, train_preds.reshape(-1, 1).astype(float)))
X_test_extended = np.hstack((X_test, test_preds.reshape(-1, 1).astype(float)))

# Train the SVM model with the extended features
rf = RandomForestClassifier()
rf.fit(X_train_extended, y_train_encoded)

y_pred_rf_encoded = rf.predict(X_test_extended)
y_pred_rf = label_encoder.inverse_transform(y_pred_rf_encoded)

print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")

# Classification report
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_)
print(report)

"""
# Compute confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
"""