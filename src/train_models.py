import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc


train_data = pd.read_csv("../data/train_data.csv")
test_data = pd.read_csv("../data/test_data.csv")

# Define continuous numerical columns
cols_numerice = ['Open', 'Close', 'Volume', 'Fear_Index', 'SMA_5', 'SMA_20', 'RSI_14']

# Impute missing values with column median
for col in cols_numerice:
    if col in train_data.columns:
        train_data[col] = train_data[col].fillna(train_data[col].median())
        test_data[col] = test_data[col].fillna(test_data[col].median())


# Prepare feature matrix excluding Target and Date
train1 = train_data.drop(columns=['Target', 'Date'], errors='ignore')
# Target vector as integer
train2 = train_data['Target'].astype(int)

# Test feature matrix and target vector
test1 = test_data.drop(columns=['Target', 'Date'], errors='ignore')
test2 = test_data['Target'].astype(int)

# One-hot encode categorical Day_of_Week feature
if 'Day_of_Week' in train1.columns:
    train1 = pd.get_dummies(train1, columns=['Day_of_Week'])
    test1 = pd.get_dummies(test1, columns=['Day_of_Week'])

# Reindex test columns to match training feature set
test1 = test1.reindex(columns=train1.columns, fill_value=0)

# Store feature names for importance and UI mapping
nume_coloane = train1.columns

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and scale training features
train1_scaled = scaler.fit_transform(train1)
# Transform test features
test1_scaled = scaler.transform(test1)

# Persist scaler for Gradio deployment
joblib.dump(scaler, "../models/scaler.pkl")


# Reconstruct DataFrame from scaled array
data_train_scaled = pd.DataFrame(train1_scaled, columns=nume_coloane)

# Display scaled summary statistics
print("Scaled data summary statistics:")
print(data_train_scaled.describe().round(2))

# Boxplot of scaled features
plt.figure(figsize=(10, 6))
cols_grafic = ['Open', 'Close', 'Volume', 'Fear_Index', 'SMA_5', 'SMA_20']
sns.boxplot(data=data_train_scaled[cols_grafic], palette="Set2")
plt.title('Feature Distribution After Standardization')
plt.savefig("../images/scaled_features_boxplot.png")
plt.close()

# Histograms of standardized features
data_train_scaled[['Close', 'Volume', 'RSI_14', 'SMA_20']].hist(bins=20, figsize=(10, 8), edgecolor='black')
plt.suptitle('Distribution of Numerical Features After Standardization')
plt.savefig("../images/scaled_feature_histograms.png")
plt.close()

# Correlation heatmap of continuous features
plt.figure(figsize=(10, 8))
cols_continue = ['Open', 'Close', 'Volume', 'Fear_Index', 'SMA_5', 'SMA_20', 'RSI_14']
sns.heatmap(data_train_scaled[cols_continue].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap After Standardization')
plt.savefig("../images/scaled_correlation_heatmap.png")
plt.close()

# Violin plot of scaled RSI-14 vs Target
data_train_scaled_viz = data_train_scaled.copy()
data_train_scaled_viz['Target'] = train2.values

plt.figure(figsize=(6, 5))
sns.violinplot(data=data_train_scaled_viz, x='Target', y='RSI_14', hue='Target', palette='Set2', legend=False)
plt.title('Scaled RSI-14 vs Target Distribution')
plt.savefig("../images/scaled_rsi_violin_plot.png")
plt.close()


# Initialize classification models
model_lr = LogisticRegression(max_iter=1000, random_state=100)
model_rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=100)
model_knn = KNeighborsClassifier(n_neighbors=50)

modele = [("LogisticRegression", model_lr), ("RandomForest", model_rf), ("KNN", model_knn)]

rezultate = []
roc_data = {}

for nume, model in modele:
    # Train model on scaled dataset
    model.fit(train1_scaled, train2)
    
    # Save trained model weights
    joblib.dump(model, f"../models/{nume}.pkl")
    # Also save backwards-compatible name for LogisticRegression
    if nume == "LogisticRegression":
        joblib.dump(model, "../models/RegresieLogistica.pkl")
    
    # Generate class predictions and probabilities
    y_pred = model.predict(test1_scaled)
    y_prob = model.predict_proba(test1_scaled)[:, 1]
    
    # Compute performance evaluation metrics
    acc = accuracy_score(test2, y_pred)
    prec = precision_score(test2, y_pred)
    rec = recall_score(test2, y_pred)
    f1 = f1_score(test2, y_pred)
    
    rezultate.append({"Model": nume, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1})
    
    # Compute ROC Curve and AUC
    fpr, tpr, _ = roc_curve(test2, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_data[nume] = (fpr, tpr, roc_auc)
    
    # Confusion Matrix
    cm = confusion_matrix(test2, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title(f"Confusion Matrix ({nume})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # Save with clean descriptive name
    name_map = {
        "RandomForest": "random_forest",
        "LogisticRegression": "logistic_regression",
        "KNN": "knn"
    }
    plt.savefig(f"../images/confusion_matrix_{name_map[nume]}.png")
    plt.close()

# Plot comparative ROC curves
plt.figure(figsize=(8, 6))
for nume, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f'{nume} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparative ROC Curves')
plt.legend(loc="lower right")
plt.savefig("../images/roc_curves_comparison.png")
plt.close()

# Display comparative evaluation metrics table
data_rezultate = pd.DataFrame(rezultate)
print("\nModel Evaluation Comparison:")
print(data_rezultate.to_string(index=False))
