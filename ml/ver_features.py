import joblib

features = joblib.load("ml/features_btts.pkl")

print(type(features))
print(features)