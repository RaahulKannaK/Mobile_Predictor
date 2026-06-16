import pickle
import numpy as np
from django.shortcuts import render

# load trained model
model = pickle.load(open('model.pkl', 'rb'))

def home(request):
    result = None

    if request.method == "POST":
        price = int(request.POST['price'])
        ram = int(request.POST['ram'])
        storage = int(request.POST['storage'])
        battery = int(request.POST['battery'])

        data = np.array([[price, ram, storage, battery]])
        prediction = model.predict(data)

        result = prediction[0]

    return render(request, 'home.html', {'result': result})