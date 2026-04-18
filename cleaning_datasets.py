import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\prince kumar\Downloads\studentperformance.csv")

# Target define
target = "Exam_Score"

# Handle missing values
data = data.fillna(data.mode().iloc[0])