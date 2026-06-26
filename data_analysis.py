# ==========================================
# EPIC 2: Visualizing and Analysing the Data
# ==========================================

# ------------------------------------------
# 1. Importing the Libraries
# ------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")

# ------------------------------------------
# 2. Reading the Dataset
# ------------------------------------------
# Make sure 'flood dataset.xlsx' is in the same folder as this script.
dataset = pd.read_excel('flood dataset.xlsx')
print("Dataset loaded successfully!")

# ------------------------------------------
# 3. Descriptive Analysis
# ------------------------------------------
# It is usually best to run descriptive statistics immediately after loading 
# to get a clear overview of the data structure.
print("\n--- Descriptive Analysis ---")

print("\n1. First 5 observations (head):")
print(dataset.head())

print("\n2. Dataset Structure (info):")
dataset.info()

print("\n3. Statistical Summary (describe):")
print(dataset.describe())

# ------------------------------------------
# 4. Univariate Analysis
# ------------------------------------------
# Examining a single variable to understand its distribution.
print("\n--- Univariate Analysis ---")
print("Generating Annual Rainfall Distribution Plot...")

plt.figure(figsize=(8, 5))
# Using histplot as it is the modern replacement for the deprecated distplot
sns.histplot(dataset['ANNUAL'], kde=True, color='blue')
plt.title('Distribution of Annual Rainfall')
plt.xlabel('Annual Rainfall (mm)')
plt.ylabel('Density / Count')
plt.show()  
# IMPORTANT: The script will pause here. You must close the graph window to continue!

# ------------------------------------------
# 5. Multivariate Analysis
# ------------------------------------------
# Examining relationships between multiple variables simultaneously.
print("\n--- Multivariate Analysis ---")
print("Generating Feature Correlation Heatmap...")

plt.figure(figsize=(12, 8))
# numeric_only=True ensures the heatmap only tries to correlate numerical columns
correlation_matrix = dataset.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='summer', linewidths=1, linecolor='k', square=True)
plt.title('Correlation Heatmap of Weather Features')
plt.show()