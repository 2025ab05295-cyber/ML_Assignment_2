from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

df = pd.concat([X, y], axis=1)

print(df.head())
print("\nShape:", df.shape)
print("\nClass distribution:")
print(y.value_counts())
df.to_csv("breast_cancer.csv", index=False)
print("\nCSV saved!")