import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ✅ Load dataset
data = pd.read_csv(r"C:\Users\prince kumar\Downloads\studentperformance.csv")

# ✅ Fill missing values
data = data.fillna(data.mode().iloc[0])

# ✅ Encode categorical features
le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# ✅ Check output
print(data.head())