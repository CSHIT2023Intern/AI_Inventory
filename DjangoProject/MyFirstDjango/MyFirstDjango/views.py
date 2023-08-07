from django.shortcuts import render
from PIL import Image
import sys
import os
import sys
sys.path.insert(0, 'C:/Users/user/Desktop/中山醫醫資/大三下/實習/112_intern/AI_Inventory/DjangoProject/MyFirstDjango')
print(sys.path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from .models.experimental import attempt_load
#from .detect import detectapi
from .detect import detectapi
import cv2
import numpy as np
from io import BytesIO
import base64
import re
from collections import Counter
from collections import defaultdict

a=detectapi(weights='static/07211657_best.pt')

def index(request):
    context={}
    context["name"] = "aaa"
    return render(request, "index.html", context)

def carema(request):
    context={}
    context["name"] = "bbb"
    return render(request, "carema.html", context)

def FinalAns(request):
    context = {}
    if request.method == 'GET':
        return render(request, "FinalAns.html")
    elif request.method == "POST":

        if request.POST.get('base64_file') != "":
            files = request.POST.get('base64_file')
            base64_data = re.sub('^data:image/.+;base64,', '', files)
            byte_data = base64.b64decode(base64_data)
            image_data = BytesIO(byte_data)
            img = Image.open(image_data)

        else:
            files = request.FILES['file']
            img = Image.open(files.file)

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        # Create an instance of the detectapi class
        #a = detectapi(weights='path/to/weights.pth', img_size=640)  # Replace with actual weights path

        # Call the detect method to get the result and number of detected objects
        result, num = a.detect([img])

        # Get the detected class name for each detected object and ignore the confidence values
        class_counts = {}

        for _, result_txt in result:
            for class_index, _,_ in result_txt:
                class_name = a.names[class_index]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        finalImg = Image.fromarray(result[0][0])

        base64_img = image_to_base64(finalImg)
        context["img"] = base64_img
        context["name"] = "FN"
        context["num"] = num
        context["class_counts"] = class_counts

        return render(request, "FinalAns.html", context)

def image_to_base64(img):
    img=img.resize((400,400))
    img=cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
    img=Image.fromarray(img)
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    output_buffer.seek(0)
    data_uri = base64.b64encode(output_buffer.read()).decode('utf-8')
    return data_uri


