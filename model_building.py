import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

print("Starting Epic 4: Model Building (Finalized Version)...\n")

# --- 1. Data Prep ---
dataset = pd.read_excel('flood dataset.xlsx')
dataset.fillna(dataset.median(numeric_only=True), inplace=True)

# Outlier Handling
for col in dataset.select_dtypes(include=[np.number]).columns:
    if col != 'flood':
        Q1, Q3 = dataset[col].quantile(0.25), dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        dataset[col] = np.where(dataset[col] > Q3 + 1.5 * IQR, Q3 + 1.5 * IQR, 
                       np.where(dataset[col] < Q1 - 1.5 * IQR, Q1 - 1.5 * IQR, dataset[col]))

# --- 2. Label Encoding & Mapping Check ---
le = LabelEncoder()
dataset['flood'] = le.fit_transform(dataset['flood'])
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("\n========== LABEL MAPPING CHECK ==========")
print("Label Mapping:", mapping)
# Keep this mapping in mind for your app.py logic!

# --- 3. Split & Scale ---
X = dataset[['Cloud Cover', 'ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep']].values
y = dataset['flood'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# --- 4. Training (XGBoost/GradientBoosting) ---
# Using the model that performed best
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# --- 5. Evaluation ---
y_pred = model.predict(X_test)
print("\n========== FINAL EVALUATION ==========")
print("Accuracy:", accuracy_score(y_test, y_pred))

# --- 6. Save ---
joblib.dump(model, 'floods.save')
joblib.dump(sc, 'transform.save')
print("\n[SUCCESS] Saved 'floods.save' and 'transform.save'!")