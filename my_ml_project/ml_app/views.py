from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from PIL import Image
import numpy as np
import io
from .utils import make_prediction,save_model

# Create your views here.
def predict_image_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image = Image.open(image_file)
        
        print('Image: ', image)
        # image = image.resize((224, 224))  # Resize based on your model input size
        image_array = np.array(image) / 255.0  # Normalize pixel values if needed
        # print('Image array: ', image_array)
        # Convert image to the format required by your model
        save_model()
        prediction = make_prediction(image_array)
        
        splid_data = prediction.split('_')
        context={}
        if len(splid_data) >1:
            context = {
                'name':splid_data[0],
                'dept':splid_data[1],
                'batch':splid_data[2],
                'id':splid_data[3]
            }
            return JsonResponse({'context':context})
        
        print('Prediction is : ',prediction)
        print(context)
        
        return JsonResponse({'prediction': prediction})
    return render(request, 'ml_app/index.html')