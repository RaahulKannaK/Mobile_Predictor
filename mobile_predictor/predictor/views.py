import os
import pickle
import pandas as pd
from django.shortcuts import render

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# paths (kept same as yours ✅)
model_path = os.path.join(BASE_DIR, 'predictor/static/pkl/model.pkl')
scaler_path = os.path.join(BASE_DIR, 'predictor/static/pkl/scaler.pkl')

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

def home(request):
    result = None

    if request.method == "POST":
        price = int(request.POST['price'])
        ram = int(request.POST['ram'])
        storage = int(request.POST['storage'])
        battery = int(request.POST['battery'])
        camera = int(request.POST['camera'])

        # ✅ FIX: use DataFrame with column names
        input_data = pd.DataFrame([[price, ram, storage, battery, camera]],
                                 columns=['Price', 'RAM', 'Storage', 'Battery', 'Camera'])

        # scale + predict
        data_scaled = scaler.transform(input_data)
        result = model.predict(data_scaled)[0]

    return render(request, 'home.html', {'result': result})