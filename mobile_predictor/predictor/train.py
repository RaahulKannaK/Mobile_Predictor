import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
data = pd.read_csv("D:/raahulkanna/New_Desktop/intern/mobile_predictor/mobiles.csv")

# Features (added Camera)
X = data[['Price','RAM','Storage','Battery','Camera']]
y = data['Category']

# Scaling (IMPORTANT for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_scaled, y)

# Save model
pickle.dump(model, open('D:/raahulkanna/New_Desktop/intern/mobile_predictor/predictor/static/pkl/model.pkl', 'wb'))

# Save scaler (NEW)
pickle.dump(scaler, open('D:/raahulkanna/New_Desktop/intern/mobile_predictor/predictor/static/pkl/scaler.pkl', 'wb'))

print("Model & Scaler trained and saved successfully!")