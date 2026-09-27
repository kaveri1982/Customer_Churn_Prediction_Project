import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print('Shape:', df.shape)
print(df.head())
print(df.info())
print('\nMissing values:\n', df.isnull().sum())

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges']).copy()
df = df.drop('customerID', axis=1)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

plt.figure(figsize=(6,4)); sns.countplot(x='Churn', data=df); plt.title('Customer Churn Distribution'); plt.show()
plt.figure(figsize=(8,5)); sns.histplot(data=df, x='tenure', hue='Churn', bins=30, kde=True, element='step'); plt.title('Tenure Distribution by Churn'); plt.show()
plt.figure(figsize=(8,5)); sns.boxplot(x='Churn', y='MonthlyCharges', data=df); plt.title('Monthly Charges vs Churn'); plt.show()

X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000, random_state=42); log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled); log_prob = log_model.predict_proba(X_test_scaled)[:,1]
print('\nLOGISTIC REGRESSION'); print('Accuracy:', accuracy_score(y_test, log_pred)); print('ROC-AUC:', roc_auc_score(y_test, log_prob)); print(classification_report(y_test, log_pred))

rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'); rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test); rf_prob = rf_model.predict_proba(X_test)[:,1]
print('\nRANDOM FOREST'); print('Accuracy:', accuracy_score(y_test, rf_pred)); print('ROC-AUC:', roc_auc_score(y_test, rf_prob)); print(classification_report(y_test, rf_pred))

plt.figure(figsize=(5,4)); sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues'); plt.title('Random Forest Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.show()
log_fpr, log_tpr, _ = roc_curve(y_test, log_prob); rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
plt.figure(figsize=(7,5)); plt.plot(log_fpr, log_tpr, label='Logistic Regression'); plt.plot(rf_fpr, rf_tpr, label='Random Forest'); plt.plot([0,1],[0,1],'--',label='Random Guess'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve'); plt.legend(); plt.show()
importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
plt.figure(figsize=(8,6)); importance.sort_values().plot(kind='barh'); plt.title('Top 15 Features - Random Forest'); plt.xlabel('Importance'); plt.show()
sample = X_test.iloc[[0]]; p = rf_model.predict(sample)[0]; prob = rf_model.predict_proba(sample)[0,1]; print('\nExample Customer Prediction:', 'Churn' if p==1 else 'No Churn'); print('Churn Probability:', round(prob*100,2), '%')
print(pd.DataFrame({'Model':['Logistic Regression','Random Forest'],'Accuracy':[accuracy_score(y_test,log_pred),accuracy_score(y_test,rf_pred)],'ROC_AUC':[roc_auc_score(y_test,log_prob),roc_auc_score(y_test,rf_prob)]}))
