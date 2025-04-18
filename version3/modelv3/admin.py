from django.contrib import admin
from .models import PCAResult, AnovaResult, RandomForestResult

# Register your models here.
admin.site.register(PCAResult)
admin.site.register(AnovaResult)
admin.site.register(RandomForestResult)