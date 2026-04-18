import pandas as pd

# ✅ Load dataset (fixed path issue)
data = pd.read_csv(r"C:\Users\prince kumar\Downloads\studentperformance.csv")

# ==============================
# 🔍 BASIC DATA UNDERSTANDING
# ==============================

# 1. First 5 rows
print("\n===== First 5 Rows =====")
print(data.head())

# 2. Last 5 rows
print("\n===== Last 5 Rows =====")
print(data.tail())

# 3. Dataset shape
print("\n===== Shape of Dataset =====")
print("Rows:", data.shape[0], "Columns:", data.shape[1])

# 4. Column names
print("\n===== Column Names =====")
print(data.columns)

# 5. Data types & info
print("\n===== Dataset Info =====")
print(data.info())

# 6. Statistical summary (numerical)
print("\n===== Statistical Summary =====")
print(data.describe())

# 7. Missing values
print("\n===== Missing Values =====")
print(data.isnull().sum())

# 8. Unique values (categorical understanding)
print("\n===== Unique Values Count =====")
for col in data.columns:
    print(f"{col}:", data[col].nunique())

# 9. Value counts (example for categorical columns)
print("\n===== Value Counts (Sample Columns) =====")
if "gender" in data.columns:
    print("\nGender Distribution:\n", data["gender"].value_counts())

if "race/ethnicity" in data.columns:
    print("\nRace/Ethnicity:\n", data["race/ethnicity"].value_counts())

# 10. Correlation (numerical features)
print("\n===== Correlation Matrix =====")
print(data.corr(numeric_only=True))
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution plot
data.hist(figsize=(10,8))
plt.show()

# Correlation heatmap
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()