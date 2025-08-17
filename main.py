import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt


# Perceptron and training tracking
def perceptron_train(X, y, T):
    theta = np.zeros(X.shape[1]);
    theta_zero = 0
    accuracy_history = []

    for _ in range(T):
        for i in range(X.shape[0]):
            if y[i] * (np.dot(theta, X[i]) + theta_zero) <= 0:
                theta = theta + y[i] * X[i]
                theta_zero = theta_zero + y[i]
        # Track training accuracy per epoch
        y_pred_train = np.where(np.dot(X, theta) + theta_zero >= 0, 1, -1)
        accuracy_history.append((y_pred_train == y).mean())

    return theta, theta_zero, accuracy_history


def perceptron_predict(X, theta, theta_zero):
    return np.where(np.dot(X, theta) + theta_zero >= 0, 1, -1)


# 1) Load data
df = pd.read_csv("data/transactions.csv")

# Feature engineering
df['balance_change'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['dest_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']
df['amount_to_oldbalance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)  # avoid division by zero
df['amount_to_newbalance_ratio'] = df['amount'] / (df['newbalanceOrig'] + 1)

# 2) Prepare features and target
X_df = df.drop("isFraud", axis=1)
y = np.where(df["isFraud"].values == 1, 1, -1)

# 3) Preprocessing
numeric_features = ["amount", "oldbalanceOrg", "newbalanceOrig",
                    "oldbalanceDest", "newbalanceDest",
                    "balance_change", "dest_balance_change",
                    "amount_to_oldbalance_ratio", "amount_to_newbalance_ratio"]

categorical_features = ["type"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

X = preprocessor.fit_transform(X_df)

# 4) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=(y == 1)
)

# 5) Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", np.bincount(y_train + 1))
print("After SMOTE:", np.bincount(y_train_res + 1))

# 6) Train perceptron (track accuracy)
T = 40
theta, theta_zero, acc_history = perceptron_train(X_train_res, y_train_res, T)

# Plot training accuracy over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, T + 1), acc_history, marker='o')
plt.title("Perceptron Training Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.grid(True)
plt.show()

# 7) Predict + Evaluate
y_pred = perceptron_predict(X_test, theta, theta_zero)
print(classification_report(y_test, y_pred, target_names=["Legit (-1)", "Fraud (+1)"]))


# 8) Inference helper
def flag_transaction(tx_dict, preprocessor, theta, theta_zero):
    tx_df = pd.DataFrame([tx_dict])
    X_tx = preprocessor.transform(tx_df).toarray()
    return "FRAUD" if perceptron_predict(X_tx, theta, theta_zero)[0] == 1 else "LEGIT"
