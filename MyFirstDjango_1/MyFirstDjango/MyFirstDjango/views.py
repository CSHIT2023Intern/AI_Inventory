from django.shortcuts import render

from PIL import Image
from .detect import detectapi, save_train_img, save_mark_img
from django.conf import settings
import sys
sys.path.insert(0, '/Users/User/Desktop/MyFirstDjango')
import cv2
import numpy as np
from io import BytesIO
import base64
import re
import os
a=detectapi(weights='static/20220817.pt')
set_img=''
def index(request):
    context = {}
    context["name"] = "aaa"
    if request.POST.get('label') != None:
        files = request.POST.get('im0')
        byte_data = base64.b64decode(files)
        image_data = BytesIO(byte_data)
        img = Image.open(image_data)
        if request.POST.get('label') != "":
            label=request.POST.get('label')
            save_train_img(img,label)
        else:
            save_mark_img(img)
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

        if request.POST.get('base64_file')!="":
            files = request.POST.get('base64_file')
            base64_data = re.sub('^data:image/.+;base64,', '', files)
            byte_data = base64.b64decode(base64_data)
            image_data = BytesIO(byte_data)
            img = Image.open(image_data)
        else:
            files = request.FILES['file']
            img = Image.open(files.file)

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        result,num,labels,label,im0=a.detect([img])
        finalImg = Image.fromarray(result[0][0])
        im0=Image.fromarray(im0[0])
        base64_img=image_to_base64(finalImg)
        base64_im0 = image_to_base64(im0)
        context["img"] = base64_img
        context["im0"] = base64_im0
        context["name"] = "FN"
        context["num"] = num
        context["labels"] = labels
        context["label"] = label

        return render(request, "FinalAns.html", context)


def image_to_base64(img):
    img=img.resize((640,640))
    img=cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
    img=Image.fromarray(img)
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    output_buffer.seek(0)
    data_uri = base64.b64encode(output_buffer.read()).decode('utf-8')
    return data_uri



