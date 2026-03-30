import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Sample dataset (replace with real dataset if needed)
data = {
    "age": [63,37,41,56,57,57,56,44,52,57],
    "sex": [1,1,0,1,0,1,0,1,1,0],
    "cp": [3,2,1,1,0,0,1,1,2,0],
    "trestbps": [145,130,130,120,120,140,140,120,172,150],
    "chol": [233,250,204,236,354,192,294,263,199,168],
    "fbs": [1,0,0,0,0,0,0,0,1,0],
    "restecg": [0,1,0,1,1,1,0,1,1,1],
    "thalach": [150,187,172,178,163,148,153,173,162,174],
    "exang": [0,0,0,0,1,0,0,0,0,0],
    "oldpeak": [2.3,3.5,1.4,0.8,0.6,0.4,1.3,0.0,0.5,1.6],
    "target": [1,0,1,0,1,0,1,0,1,0]
}

df = pd.DataFrame(data)

X = df.drop("target", axis=1)
y = df["target"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_scaled, y)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print(" Model & Scaler saved!")