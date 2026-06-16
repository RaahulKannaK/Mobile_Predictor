import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle

data = pd.read_csv("mobiles.csv")

X = data[['Price','RAM','Storage','Battery']]
y = data['Category']

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained & saved!")