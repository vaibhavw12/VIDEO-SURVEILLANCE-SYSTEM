from collections import deque
import os
import cv2
import math
# import pafy
import random
import numpy as np
import datetime as dt
import tensorflow as tf
# from moviepy.editor import *
from collections import deque
import matplotlib.pyplot as plt
# %matplotlib inline
from django.core.files.storage import default_storage
from django.shortcuts import render
import tensorflow as tf
from moviepy.video.io.VideoFileClip import VideoFileClip
# Create your views here.
from keras.models import load_model

# Load the saved model
model = load_model('savedmodels\Model___Date_Time_2023_02_15__01_10_15___Loss_0.008450016379356384___Accuracy_1.0.h5')


def welcome(request):
    return render(request, 'index.html')


def upload(request):
    if request.method == 'POST':
        video_file = request.FILES['video']
        input_number_str = request.GET.get('input_number')
        frame_skip = 10
        # if input_number_str != "":
        #     frame_skip = int(input_number_str)
        print(frame_skip)
        video_path = default_storage.save('videos/' + video_file.name, video_file)
        # You can do further processing with the video file here
        v = str(video_file)
        print(v)
        window_size = 25

        # Constructing The Output YouTube Video Path
        output_file_path = 'output_video/output_'+v+'.mp4'

        video_file_path = 'media/videos/'+v
        #video_input = "C:/Users/vaibhav/final/learnopencv/y2mate.com - Misbehave to female banker by customer in bank of baroda_360p"
        predict_on_live_video(video_file_path, output_file_path, window_size, frame_skip)
        setting_metadata(output_file_path)
       # print(VideoFileClip(output_file_path).ipython_display(width=700))
        #show_output()
        print("finally done")
        return render(request, 'upload.html', {'file': video_file})


# def predict_on_live_video(video_file_path, output_file_path, window_size, frame_skip):
#     image_height, image_width = 64, 64
#     frame_count = 0
#
#     # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
#     predicted_labels_probabilities_deque = deque(maxlen=window_size)
#
#     # Reading the Video File using the VideoCapture Object
#     video_reader = cv2.VideoCapture(video_file_path)
#
#     # Getting the width and height of the video
#     original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
#     original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # Writing the Overlayed Video Files Using the VideoWriter Object
#     video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24,
#                                    (original_video_width, original_video_height))
#     print("i")
#     while True:
#         print("bet")
#         # Reading The Frame
#         # status, frame = video_reader.read()
#
#         # Reading The Frame
#         status, frame = video_reader.read()
#
#         # skip frames
#         if frame_count % frame_skip != 0:
#             frame_count += 1
#             continue
#
#         if not status:
#             break
#
#         # rest of the processing code...
#
#         # Writing The Frame
#         video_writer.write(frame)
#         frame_count += 1
#
#         # if not status:
#         #    break
#
#         # Resize the Frame to fixed Dimensions
#         resized_frame = cv2.resize(frame, (image_height, image_width))
#
#         # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
#         normalized_frame = resized_frame / 255
#
#         # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
#         predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis=0))[0]
#
#         # Appending predicted label probabilities to the deque object
#         predicted_labels_probabilities_deque.append(predicted_labels_probabilities)
#
#         # Assuring that the Deque is completely filled before starting the averaging process
#         if len(predicted_labels_probabilities_deque) == window_size:
#             # Converting Predicted Labels Probabilities Deque into Numpy array
#             predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)
#
#             # Calculating Average of Predicted Labels Probabilities Column Wise
#             predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis=0)
#
#             # Converting the predicted probabilities into labels by returning the index of the maximum value.
#             predicted_label = np.argmax(predicted_labels_probabilities_averaged)
#
#             # Accessing The Class Name using predicted label.
#             classes_list = ["Abnormal", "Normal"]
#             predicted_class_name = classes_list[predicted_label]
#
#             # Overlaying Class Name Text Ontop of the Frame
#             cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         print("ii")
#         # Writing The Frame
#         video_writer.write(frame)
#
#         # cv2.imshow('Predicted Frames', frame)
#
#         # key_pressed = cv2.waitKey(10)
#
#         # if key_pressed == ord('q'):
#         #     break
#
#     # cv2.destroyAllWindows()
#
#     # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them.
#     video_reader.release()
#     video_writer.release()

def predict_on_live_video(video_file_path, output_file_path, window_size, frame_skip):
    image_height, image_width = 64, 64
    frame_count = 0

    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen=window_size)

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24,
                                   (original_video_width, original_video_height))
    print("Processing video...")

    while True:
        # Reading The Frame
        status, frame = video_reader.read()

        # skip frames
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        if not status:
            print("break")
            break

        # rest of the processing code...

        # Writing The Frame
        video_writer.write(frame)
        frame_count += 1

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis=0))[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:
            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

            # Calculating Average of Predicted Labels Probabilities Column Wise
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis=0)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            # Accessing The Class Name using predicted label.
            classes_list = ["Abnormal", "Normal"]
            predicted_class_name = classes_list[predicted_label]

            # Overlaying Class Name Text Ontop of the Frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Writing The Frame
        video_writer.write(frame)
        print("Frame processed: ", frame_count)

    print("Video processing completed.")

    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them.
    video_reader.release()
    video_writer.release()

def setting_metadata(input_file ):
    # Load the input video file
    video = VideoFileClip(input_file)

    # Define the output video file settings
    fps = 24
    codec = "libx264"
    bitrate = "2000k"

    # Define the metadata for the output video file
    metadata = {"title": "My Video", "artist": "Me"}

    # Set the metadata for the output file
    video.metadata = metadata

    # Write the output video file with the new settings and metadata
    output_directory = "static\VSB"
    output_file = output_directory + "/output.mp4"
    video.write_videofile(output_file, fps=fps, codec=codec, bitrate=bitrate)
    print("done")




def show_output():
    # Open the video file
    cap = cv2.VideoCapture('output_video/{window}.mp4')
    print("happening")
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error opening video file")

    # Loop through the frames of the video
    while cap.isOpened():
        # Read a frame from the video file
        ret, frame = cap.read()

        # If the frame was read successfully, display it
        if ret:
            cv2.imshow('Frame', frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video file and close all windows
    cap.release()
    cv2.destroyAllWindows()
