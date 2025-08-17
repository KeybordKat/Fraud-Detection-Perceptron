import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def perceptron_train(X, y, T):
    theta = np.zeros(X.shape[1]); theta_zero = 0
    for _ in range(T):
        for i in range(X.shape[0]):
            if y[i] * (np.dot(theta, X[i]) + theta_zero) <= 0:
                theta = theta + y[i] * X[i]
                theta_zero = theta_zero + y[i]
    return theta, theta_zero

def perceptron_predict(X, theta, theta_zero):
    return np.where(np.dot(X, theta) + theta_zero >= 0, 1, -1)

# 1) Load
df = pd.read_csv("data/transactions.csv")
X_df = df.drop("isFraud", axis=1)
y = np.where(df["isFraud"].values == 1, 1, -1)

# 2) Preprocess
numeric_features = ["amount", "oldbalanceOrg", "newbalanceOrig",
                    "oldbalanceDest", "newbalanceDest"]
categorical_features = ["type"]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])
X = preprocessor.fit_transform(X_df)

# 3) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=(y==1)
)

# 4) Train
T = 40
theta, theta_zero = perceptron_train(X_train, y_train, T)

# 5) Evaluate
y_pred = perceptron_predict(X_test, theta, theta_zero)
print(classification_report(y_test, y_pred, target_names=["Legit (-1)", "Fraud (+1)"]))

# 6) Inference helper
def flag_transaction(tx_dict, preprocessor, theta, theta_zero):
    tx_df = pd.DataFrame([tx_dict])
    X_tx = preprocessor.transform(tx_df).toarray()
    return "FRAUD" if perceptron_predict(X_tx, theta, theta_zero)[0] == 1 else "LEGIT"
