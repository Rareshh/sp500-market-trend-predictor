import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


data_train = pd.read_csv("../data/train_data.csv")
data_test = pd.read_csv("../data/test_data.csv")

# Missing values analysis
print("Missing values count:")
print(data_train.isna().sum())

# Descriptive statistics
print("\nDescriptive statistics:")
print(data_train.describe())

# Distribution of numerical features
data_train[['Close', 'Open', 'Volume', 'RSI_14']].hist(bins=20, figsize=(10, 8), edgecolor='black')
plt.suptitle('Distribution of Numerical Features')
plt.savefig("../images/feature_histograms.png")
plt.close()

# Target class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=data_train, x='Target', palette='viridis', hue='Target', legend=False)
plt.title('Target Class Distribution')
plt.savefig("../images/target_distribution.png")
plt.close()

# Outlier detection (Volume)
plt.figure(figsize=(6, 4))
sns.boxplot(data=data_train, y='Volume', color='lightblue')
plt.title('Volume Outlier Detection')
plt.savefig("../images/volume_boxplot.png")
plt.close()

# Feature correlation heatmap
plt.figure(figsize=(10, 8))
numeric_cols = data_train[['Open', 'Close', 'Volume', 'Fear_Index', 'SMA_5', 'SMA_20', 'RSI_14']]
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.savefig("../images/correlation_heatmap.png")
plt.close()

# Relationship between RSI-14 and Target
plt.figure(figsize=(6, 5))
sns.violinplot(data=data_train, x='Target', y='RSI_14', hue='Target', palette='Set2', legend=False)
plt.title('RSI_14 vs Target Distribution')
plt.savefig("../images/rsi_violin_plot.png")
plt.close()


train1 = data_train.drop(columns=['Target', 'Date'])
train2 = data_train['Target'].astype(int)
test1 = data_test.drop(columns=['Target', 'Date'])
test2 = data_test['Target'].astype(int)

# Categorical encoding for Day_of_Week
encoder = LabelEncoder()
train1['Day_of_Week'] = encoder.fit_transform(train1['Day_of_Week'].astype(str))
test1['Day_of_Week'] = encoder.transform(test1['Day_of_Week'].astype(str))

# Feature standardization
scaler = StandardScaler()
train1_scaled = scaler.fit_transform(train1)
test1_scaled = scaler.transform(test1)

# Fit Logistic Regression baseline model
model = LogisticRegression(max_iter=1000)
model.fit(train1_scaled, train2)

y_pred = model.predict(test1_scaled)

print(f"\nAccuracy: {accuracy_score(test2, y_pred):.4f}")
print(f"Precision: {precision_score(test2, y_pred):.4f}")
print(f"Recall: {recall_score(test2, y_pred):.4f}")
print(f"F1-Score: {f1_score(test2, y_pred):.4f}")

# Confusion matrix
plt.figure(figsize=(5, 4))
cm = confusion_matrix(test2, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("../images/confusion_matrix_baseline_lr.png")
plt.close()
