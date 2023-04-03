import os
import base64
import cv2
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from .models import Image
from .forms import ImageForm

@csrf_exempt
def index(request):
    form = ImageForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()
        return JsonResponse({'success': True})
    
    context = {'form':form}
    return render(request, 'cropping/capture.html', context)

@csrf_exempt
def save_image(request):
    if request.method == 'POST':
        # Decode the base64-encoded image data
        dataUrl = request.POST.get('image')
        try:
            _, imgstr = dataUrl.split(';base64,')
        except ValueError:
            return JsonResponse({'success': False, 'message': 'Invalid image data'})
        image_data = base64.b64decode(imgstr)
        # Decode the image data to an OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Crop the image to the desired size
        x = int(request.POST.get('x'))
        y = int(request.POST.get('y'))
        w = int(request.POST.get('w'))
        h = int(request.POST.get('h'))
        img_cropped = img[y:y+h, x:x+w]
        # Create a filename for the image
        dir_path = os.path.join('capture-image')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        files = os.listdir(dir_path)
        files_count = len(files)
        filename = os.path.join(dir_path, f'image_{files_count}.png')
        # Save the image to the server
        cv2.imwrite(filename, img_cropped)
        return JsonResponse({'success': True})
    else:
        return JsonResponse({'success': False, 'message': 'Invalid request method'})
