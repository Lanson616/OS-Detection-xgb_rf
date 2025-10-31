import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

from sklearn.ensemble import RandomForestClassifier
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf}")

labels = ['duration', 'packets per second', 
          'bytes per second', 'mean',
          'standard deviation', '25', 
          'median', '75', 'minimum', 
          'maximum', 'packets', 'bytes',
          'mean iat', 'std_dev iat',
          'min iat', 'max iat', 
          'iat variance','entropy', 
          'mean_ttl', 'std_dev_ttl',
          'max_ttl', 'min_ttl',
          'ttl_variance', 'mean_window',
          'std_dev_window', 'max_window',
          'min_window', 'window_variance'
          ]

target_names = ["windows_windows-8.1", "mac_mac-os-x", 
                "ubuntu_ubuntu-16.4-64b", "ubuntu_ubuntu-16.4-32b", 
                "windows_windows-10-pro", "windows_windows-10", 
                "ubuntu_ubuntu-14.4-64b", "ubuntu_ubuntu-server", 
                "none_kali-linux", "windows_windows-7-pro", 
                "ubuntu_web-server", "ubuntu_ubuntu-14.4-32b", 
                "windows_windows-vista"]

# Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf}")

importances = rf.feature_importances_
indices = importances.argsort()[::-1]

print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{labels[indices[f]]}: {importances[indices[f]]}")

# Classification Report
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred_rf, target_names=target_names)
print(report)

# Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

unique_labels = np.unique(np.concatenate((y_test, y_pred_rf)))

cm = confusion_matrix(y_test, y_pred_rf, labels=unique_labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names[:len(unique_labels)], yticklabels=target_names[:len(unique_labels)])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(labels)[indices])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importances in Random Forest Model')
plt.show()

"""
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [20, 30, 40],
    'min_samples_split': [2 ,5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_dist, 
    n_iter=50, 
    scoring='accuracy', 
    cv=3, 
    random_state=42, 
    n_jobs=-1, 
    verbose=2
)

random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")

best_rf = random_search.best_estimator_
best_rf.fit(X_train, y_train)

y_pred_best_rf = best_rf.predict(X_test)

acc_best_rf = accuracy_score(y_test, y_pred_best_rf)
print(f'Best Random Forest Accuracy: {acc_best_rf}')
"""

#        random forest           xgboost             xgb_rf             
# 12 - 0.7017910447761194  0.6837810945273631  0.7017910447761194 (+Packets & Size stats)
# 17 - 0.7376119402985075  0.7264676616915423  0.7406965174129353 (+IAT stats)
# 18 - 0.7376119402985075  0.7280597014925373  0.7423880597014926 (+Entropy)
# 28 - 0.7996019900497513  0.7911442786069651  0.8019900497512438 (+ttl stats & Windows stats)