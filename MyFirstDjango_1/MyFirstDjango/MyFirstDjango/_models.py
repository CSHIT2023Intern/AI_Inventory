from django.db import models


class Img(models.Model):
    image = models.ImageField(upload_to="files/train/images")
    label = models.FileField(upload_to="files/train/labels")
