from django.core.files.storage import default_storage
from django.shortcuts import render, redirect

# Create your views here.
from django.http import HttpResponse

def welcome(request):
    return render(request , 'index.html')
#
# def user(request):
#     u = request.GET['uname']
#     print(u)
#     return render(request , 'user.html',{'name' : u})

from django.shortcuts import render
from django.contrib import messages
#from .models import Media

# def upload_video(request):
#     if request.method == 'POST':
#         video = request.FILES['video']
#        # media = Media.objects.create(file=video)
#        # messages.success(request, f'Successfully uploaded {video.name}.')
#         print(video)
#         print("ok")
#     return render(request, 'upload.html',{'file' : video})
def upload_video(request):
    if request.method == 'POST':
        video_file = request.FILES['video']
        video_path = default_storage.save('videos/' + video_file.name, video_file)
        # You can do further processing with the video file here

        return render(request, 'upload.html',{'file' : video_file})