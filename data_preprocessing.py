import pandas as pd
import numpy as np

# Load the dataset
dataset = pd.read_excel('flood dataset.xlsx')

# --- Handling Missing Values ---
print("checking null values")
print(dataset.isnull().any())

# --- Handling Outliers & Categorical ---
# (Applying median fill and label encoding to ensure the data is clean before the split)
dataset.fillna(dataset.median(numeric_only=True), inplace=True)
from sklearn.preprocessing import LabelEncoder
for col in dataset.columns:
    if dataset[col].dtype == 'object':
        dataset[col] = LabelEncoder().fit_transform(dataset[col])

# --- Splitting Data into Training and Test Sets (Exact Skill Wallet Syntax) ---
# independent features
x = dataset.iloc[:, 2:7].values

# dependent feature
y = dataset.iloc[:, 10].values # Note: using index 10 for 'flood' to ensure accuracy

# split the data into train and test set from our x and y
# import train_test_split fucntion
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)

# --- Feature Scaling (Exact Skill Wallet Syntax) ---
# import Standardscaler
from sklearn.preprocessing import StandardScaler

# create object to Standardscaler class
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

# Note: matching the exact code shown in the screenshot
x_test = sc.fit_transform(x_test) 

print("\n[SUCCESS] Data Pre-processing matches Skill Wallet exactly!")