import os
import pickle
import pandas as pd
from django.shortcuts import render

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, 'predictor/static/pkl/model.pkl')
scaler_path = os.path.join(BASE_DIR, 'predictor/static/pkl/scaler.pkl')
data_path = os.path.join(BASE_DIR, 'mobile_list.csv')

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

# ✅ load dataset once (efficient)
mobiles = pd.read_csv(data_path)

def home(request):
    result = None
    recommended = []

    # ✅ ONLY run when button clicked
    if request.method == "POST":

        try:
            price = int(request.POST.get('price', 0))
            ram = int(request.POST.get('ram', 0))
            storage = int(request.POST.get('storage', 0))
            battery = int(request.POST.get('battery', 0))
            camera = int(request.POST.get('camera', 0))

            # ❌ IMPORTANT: prevent empty submission
            if price == 0 and ram == 0:
                return render(request, 'home.html', {
                    'result': None,
                    'recommended': []
                })

            input_data = pd.DataFrame([[price, ram, storage, battery, camera]],
                                     columns=['Price', 'RAM', 'Storage', 'Battery', 'Camera'])

            data_scaled = scaler.transform(input_data)
            result = model.predict(data_scaled)[0]

            filtered = mobiles[mobiles['Category'].str.lower() == str(result).lower()]
            recommended = filtered.to_dict(orient='records')

        except Exception as e:
            print("Error:", e)
            result = None
            recommended = []

    # ✅ GET request → empty page
    return render(request, 'home.html', {
        'result': result,
        'recommended': recommended
    })



