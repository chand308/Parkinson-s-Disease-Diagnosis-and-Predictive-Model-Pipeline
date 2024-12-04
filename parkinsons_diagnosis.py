
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
import joblib

# Load the dataset
data = pd.read_csv('Parkinsons_data.csv')

# Drop non-predictive columns
data = data.drop(columns=['name'])

# Separate features and target
X = data.drop(columns=['status'])
y = data['status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train and evaluate XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
print('XGBoost Classification Report:')
print(classification_report(y_test, xgb_preds))
xgb_accuracy = accuracy_score(y_test, xgb_preds)
print(f'XGBoost Accuracy: {xgb_accuracy}')

# Train and evaluate SVM model
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
print('SVM Classification Report:')
print(classification_report(y_test, svm_preds))
svm_accuracy = accuracy_score(y_test, svm_preds)
print(f'SVM Accuracy: {svm_accuracy}')

# Save the better performing model
if xgb_accuracy >= svm_accuracy:
    joblib.dump(xgb, 'parkinsons_xgboost_model.pkl')
    print('XGBoost model saved as parkinsons_xgboost_model.pkl')
else:
    joblib.dump(svm, 'parkinsons_svm_model.pkl')
    print('SVM model saved as parkinsons_svm_model.pkl')
