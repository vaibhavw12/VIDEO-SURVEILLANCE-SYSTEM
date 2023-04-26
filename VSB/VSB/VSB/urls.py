
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    # path("",include("VSB_mini.urls")),
    path("",include("webapp.urls")),
    path("admin/", admin.site.urls),
]
