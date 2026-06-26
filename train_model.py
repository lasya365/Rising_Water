import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

print("Starting Rising Waters ML Pipeline...")

# 1. Load Data
dataset = pd.read_excel('flood dataset.xlsx')

# 2. Handle Missing Values
dataset.fillna(dataset.median(numeric_only=True), inplace=True)

# 3. Handle Outliers (IQR Method)
for col in dataset.select_dtypes(include=[np.number]).columns:
    if col != 'flood':
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dataset[col] = np.where(dataset[col] > upper_bound, upper_bound, 
                       np.where(dataset[col] < lower_bound, lower_bound, dataset[col]))

# 4. Train/Test Split
X = dataset[['Cloud Cover', 'ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep']]
y = dataset['flood']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# 5. Feature Scaling
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# 6. Train XGBoost Model
print("Training XGBoost Model...")
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train_scaled, y_train)
xgb_pred = xgb.predict(X_test_scaled)

# 7. Evaluate & Save
print(f"Accuracy: {accuracy_score(y_test, xgb_pred) * 100:.2f}%")
joblib.dump(xgb, 'floods.save')
joblib.dump(sc, 'transform.save')
print("Pipeline Complete! Models saved.")