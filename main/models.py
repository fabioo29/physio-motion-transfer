from django.db import models

# Create your models here.
class ImagesUp(models.Model):
    name = models.CharField(max_length=100, blank=True)
    image = models.ImageField()

    class Meta:
        verbose_name_plural = "Images"

    def __str__(self):
        return self.name

class VideoUp(models.Model):
    name = models.CharField(max_length=100, blank=True)
    video = models.FileField()

    class Meta:
        verbose_name_plural = "Videos"

    def __str__(self):
        return self.name