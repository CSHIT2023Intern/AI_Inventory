# Generated by Django 2.2.5 on 2022-08-17 06:19

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Img',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='files/train/images')),
                ('label', models.FileField(upload_to='files/train/labels')),
            ],
        ),
    ]
