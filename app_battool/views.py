# from django.shortcuts import render, HttpResponse, redirect
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt  # Add this import
# from .models import BatSpecies, UserData
# from PIL import Image
# import json
# import zipfile
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import json
# import base64
# import io
# import uuid
# from scipy.io import wavfile
# import librosa.display
# import seaborn as sns
# import matplotlib
# matplotlib.use('Agg')

# sns.set_style('whitegrid')

# def home(request):
#     # adding line to test
#     species_names = list(BatSpecies.objects.values_list('name', flat=True))
#     return render(request,"test.html",{"image":0,
#                                        'species_names': species_names})

# def input_view(request):
#     if request.method == 'POST':
#         print('Raw Data: "%s"' % request.body)
#         print(type(request.body))
#         raw_data = request.body
#         try:
#             json_data = json.loads(raw_data)
#             print('Parsed JSON:', json_data)
#             print(type(json_data))

#             # Create a unique token for the user
#             token = uuid.uuid4()

#             # Store user data in the database
#             UserData.objects.create(token=token, user_data=json_data)

#             return HttpResponse("Data saved successfully.")
#         except json.JSONDecodeError as e:
#             print('Error decoding JSON:', str(e))

#     return HttpResponse("Invalid request method")

# def submit_and_download(request):
#     # Check if user_data is present in the session
#     token = request.GET.get('token')

#     if not token:
#         return HttpResponse("Token not provided.")

#     # Retrieve user data from the database
#     user_data_objects = UserData.objects.filter(token=token)
#     if not user_data_objects.exists():
#         return HttpResponse("No data found for the provided token.")

#     # Generate a unique filename for the user based on their token
#     zip_filename = f"user_data_{token}.zip"
#     zip_filepath = os.path.join("static", zip_filename)

#     # Create a zip file in memory
#     with zipfile.ZipFile(zip_filepath, "w") as zip_file:
#         # Add user data to the zip file
#         for index, user_data_object in enumerate(user_data_objects):
#             # Modify this based on your actual data structure
#             zip_file.writestr(f"annotation_{index}.json", json.dumps(user_data_object.user_data))

#     # Provide a link or redirect to the generated zip file
#     return redirect(zip_filename)

# def denoise(spec_noisy, mask=None):
#     if mask is None:
#         me = np.mean(spec_noisy, 1)
#         spec_denoise = spec_noisy - me[:, np.newaxis]
#     else:
#         mask_inv = np.invert(mask)
#         spec_denoise = spec_noisy.copy()
#         if np.sum(mask) > 0:
#             me = np.mean(spec_denoise[:, mask], 1)
#             spec_denoise[:, mask] = spec_denoise[:, mask] - me[:, np.newaxis]
#         if np.sum(mask_inv) > 0:
#             me_inv = np.mean(spec_denoise[:, mask_inv], 1)
#             spec_denoise[:, mask_inv] = spec_denoise[:, mask_inv] - me_inv[:, np.newaxis]
#     spec_denoise.clip(min=0, out=spec_denoise)
#     return spec_denoise

# def gen_spectrogram(audio_samples, sampling_rate):
#     fft_win_length = 0.02322
#     fft_overlap = 0.75
#     max_freq = 270
#     min_freq = 10
#     nfft = int(sampling_rate*fft_win_length)
#     noverlap = int(fft_overlap*nfft)
#     step = nfft - noverlap
#     shape = (nfft, (audio_samples.shape[-1]-noverlap)//step)
#     strides = (audio_samples.strides[0], step*audio_samples.strides[0])
#     x_wins = np.lib.stride_tricks.as_strided(audio_samples, shape=shape, strides=strides)
#     x_wins_han = np.hanning(x_wins.shape[0])[..., np.newaxis] * x_wins
#     complex_spec = np.fft.rfft(x_wins_han, axis=0)
#     mag_spec = (np.conjugate(complex_spec) * complex_spec).real
#     spec = mag_spec[1:, :]
#     spec = np.flipud(spec)
#     spec = spec[-max_freq:-min_freq, :]
#     req_height = max_freq-min_freq
#     if spec.shape[0] < req_height:
#         zero_pad = np.zeros((req_height-spec.shape[0], spec.shape[1]))
#         spec = np.vstack((zero_pad, spec))
#     log_scaling = 2.0 * (1.0 / sampling_rate) * (1.0/(np.abs(np.hanning(int(fft_win_length*sampling_rate)))**2).sum())
#     spec = np.log(1.0 + log_scaling*spec)
#     return spec


# def process_audio(request):
#     if request.method == 'POST':
#         # Get the file name from the POST request
#         file_name = request.FILES['file_name']
#         print(file_name)
#         species_names = list(BatSpecies.objects.values_list('name', flat=True))

#         if str(file_name).endswith('.wav'):
#             # Load the audio file and perform necessary processing
#             sampling_rate, audio_samples = wavfile.read(file_name)
#             spectrogram = gen_spectrogram(audio_samples, sampling_rate)
#             mask = spectrogram.mean(0) > 0.5
#             fig = plt.figure(figsize=(12, 4))
#             print(spectrogram.shape[1])# Number of time frames
#             librosa.display.specshow(np.flipud(spectrogram), sr=sampling_rate)
#             # Save the spectrogram image in the static folder
#             save_path = os.path.join('./app_battool/static/spectrogram_beforedenoise.png')
#             plt.tight_layout()
#             # Save the plot without extra white spaces
#             plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#             plt.close(fig)

#             spectrogram = denoise(spectrogram, mask)
#             # Generate the spectrogram image
#             fig = plt.figure(figsize=(12, 4))
#             librosa.display.specshow(np.flipud(spectrogram), sr=sampling_rate)

#             # Save the spectrogram image in the static folder
#             save_path = os.path.join('./app_battool/static/spectrogram.png')
#             plt.tight_layout()
#             # Save the plot without extra white spaces
#             plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#             plt.close(fig)
            
#             image_path="./app_battool/static/spectrogram.png"

#             with open(image_path, "rb") as image_file:
#                 image_data = image_file.read()
#             image = Image.open(io.BytesIO(image_data))
#             width, height = image.size
#             image_data_base64 = base64.b64encode(image_data).decode('utf-8')
#             ctx= image_data_base64
#             image_path="./app_battool/static/spectrogram_beforedenoise.png"

#             with open(image_path, "rb") as image_file:
#                 image_data = image_file.read()
#             image = Image.open(io.BytesIO(image_data))
#             width, height = image.size
#             image_data_base64 = base64.b64encode(image_data).decode('utf-8')
#             ctx1= image_data_base64
            
#             return render(request,"test.html",{"image":ctx,
#                                                "denoise_image":ctx1,
#                                                "height":height,
#                                                "width":width,
#                                                "annotated_data": None,
#                                                'species_names': species_names})
        
#         elif str(file_name).endswith('.zip'):
#             with zipfile.ZipFile(file_name, 'r') as zipf:
#                 try:
#                     extract_path = './app_battool/static/uploaded_data'
#                     extract_files = {'json_file': "annotated_data.json", 'image_file': "spectrogram.png"}

#                     zipf.extractall(extract_path, extract_files.values())
#                 except:
#                     # write code to handle corrupt or incorrect zip upload
#                     print("Incorrect or ZIP corrupted")
                
#                 try:
#                     image_path = extract_path + '/' + extract_files['image_file']
                    
#                     # Open the image file
#                     with open(image_path, "rb") as image_file:
#                         # Read the image data
#                         image_data = image_file.read()

#                     # Get the image dimensions using Pillow
#                     image = Image.open(io.BytesIO(image_data))
#                     width, height = image.size

#                     # Now, you have the image dimensions in the 'width' and 'height' variables
#                     print("Image Width:", width)
#                     print("Image Height:", height)

#                     # If you still need to encode the image for some reason
#                     image_data_base64 = base64.b64encode(image_data).decode('utf-8')
#                     ctx = image_data_base64
#                     # return render(request,"home.html",{"image":ctx})
                
#                 except:
#                     # write code to handle reading img           
#                     print("Incorrect or image corrupted")

#                 try:
#                     json_file = open(extract_path+'/'+extract_files['json_file'])
#                     annotated_data = json.load(json_file)
#                     print(annotated_data)
#                     return render(request,"test.html",{"image":ctx,
#                                                        "annotated_data": annotated_data,
#                                                        "height":height,
#                                                        "width":width,
#                                                        'species_names': species_names})
                
#                 except:
#                     # write code to handle reading json file           
#                     print("Incorrect or json corrupted")

#         else:
#             # write code to handle invalid file type (accepted - .wav and .zip)
#             print("Invalid file type uploaded")
#             return render(request,"test.html")
        
#     return render(request,"test.html")



# @csrf_exempt  # Add this decorator to allow CSRF-exempt requests (for simplicity; consider other security measures)
# def add_species(request):
#     if request.method == 'POST':
#         new_species = request.POST.get('new_species')
#         new_species=new_species.title()
        
#         """
#         Remove print statement during production
        
#         """
#         print(new_species)

#         # Check if the species already exists in the database
#         if not BatSpecies.objects.filter(name=new_species).exists():
#             # Add the new species to the database
#             BatSpecies.objects.create(name=new_species)
#             return JsonResponse({'status': 'success', 'message': 'Species added successfully'})
#         else:
#             return JsonResponse({'status': 'error', 'message': 'Species already exists'})

#     return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

from django.shortcuts import render, HttpResponse, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt  # Add this import
from .models import BatSpecies, UserData
from PIL import Image
import json
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import base64
import io
import uuid
from scipy.io import wavfile
import librosa.display
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

sns.set_style('whitegrid')

def home(request):
    # adding line to test
    species_names = list(BatSpecies.objects.values_list('name', flat=True))
    return render(request,"test.html",{"image":0,
                                       'species_names': species_names})

def input_view(request):
    if request.method == 'POST':
        print('Raw Data: "%s"' % request.body)
        print(type(request.body))
        raw_data = request.body
        try:
            json_data = json.loads(raw_data)
            print('Parsed JSON:', json_data)
            print(type(json_data))

            # Create a unique token for the user
            token = uuid.uuid4()

            # Store user data in the database
            UserData.objects.create(token=token, user_data=json_data)

            return HttpResponse("Data saved successfully.")
        except json.JSONDecodeError as e:
            print('Error decoding JSON:', str(e))

    return HttpResponse("Invalid request method")

def submit_and_download(request):
    # Check if user_data is present in the session
    token = request.GET.get('token')

    if not token:
        return HttpResponse("Token not provided.")

    # Retrieve user data from the database
    user_data_objects = UserData.objects.filter(token=token)
    if not user_data_objects.exists():
        return HttpResponse("No data found for the provided token.")

    # Generate a unique filename for the user based on their token
    zip_filename = f"user_data_{token}.zip"
    zip_filepath = os.path.join("static", zip_filename)

    # Create a zip file in memory
    with zipfile.ZipFile(zip_filepath, "w") as zip_file:
        # Add user data to the zip file
        for index, user_data_object in enumerate(user_data_objects):
            # Modify this based on your actual data structure
            zip_file.writestr(f"annotation_{index}.json", json.dumps(user_data_object.user_data))

    # Provide a link or redirect to the generated zip file
    return redirect(zip_filename)

def denoise(spec_noisy, mask=None):
    if mask is None:
        me = np.mean(spec_noisy, 1)
        spec_denoise = spec_noisy - me[:, np.newaxis]
    else:
        mask_inv = np.invert(mask)
        spec_denoise = spec_noisy.copy()
        if np.sum(mask) > 0:
            me = np.mean(spec_denoise[:, mask], 1)
            spec_denoise[:, mask] = spec_denoise[:, mask] - me[:, np.newaxis]
        if np.sum(mask_inv) > 0:
            me_inv = np.mean(spec_denoise[:, mask_inv], 1)
            spec_denoise[:, mask_inv] = spec_denoise[:, mask_inv] - me_inv[:, np.newaxis]
    spec_denoise.clip(min=0, out=spec_denoise)
    return spec_denoise

def gen_spectrogram(audio_samples, sampling_rate):
    fft_win_length = 0.02322
    fft_overlap = 0.75
    max_freq = 270
    min_freq = 10
    nfft = int(sampling_rate*fft_win_length)
    noverlap = int(fft_overlap*nfft)
    step = nfft - noverlap
    shape = (nfft, (audio_samples.shape[-1]-noverlap)//step)
    strides = (audio_samples.strides[0], step*audio_samples.strides[0])
    x_wins = np.lib.stride_tricks.as_strided(audio_samples, shape=shape, strides=strides)
    x_wins_han = np.hanning(x_wins.shape[0])[..., np.newaxis] * x_wins
    complex_spec = np.fft.rfft(x_wins_han, axis=0)
    mag_spec = (np.conjugate(complex_spec) * complex_spec).real
    spec = mag_spec[1:, :]
    spec = np.flipud(spec)
    spec = spec[-max_freq:-min_freq, :]
    req_height = max_freq-min_freq
    if spec.shape[0] < req_height:
        zero_pad = np.zeros((req_height-spec.shape[0], spec.shape[1]))
        spec = np.vstack((zero_pad, spec))
    log_scaling = 2.0 * (1.0 / sampling_rate) * (1.0/(np.abs(np.hanning(int(fft_win_length*sampling_rate)))**2).sum())
    spec = np.log(1.0 + log_scaling*spec)
    return spec


def process_audio(request):
    if request.method == 'POST':
        # Get the file name from the POST request
        file_name = request.FILES['file_name']
        print(file_name)
        species_names = list(BatSpecies.objects.values_list('name', flat=True))

        if str(file_name).endswith('.wav'):
            # Load the audio file and perform necessary processing
            sampling_rate, audio_samples = wavfile.read(file_name)
            spectrogram = gen_spectrogram(audio_samples, sampling_rate)
            mask = spectrogram.mean(0) > 0.5

            # Save the spectrogram before denoising
            fig = plt.figure(figsize=(12, 4))
            librosa.display.specshow(np.flipud(spectrogram), sr=sampling_rate)
            save_path = os.path.join('./app_battool/static/spectrogram_beforedenoise.png')
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # Denoise the spectrogram
            spectrogram = denoise(spectrogram, mask)

            # Save the denoised spectrogram
            fig = plt.figure(figsize=(12, 4))
            time_pixel_values = np.arange(0, spectrogram.shape[1], step=40)
            frequency_pixel_values = np.arange(0, spectrogram.shape[0], step=20)
            librosa.display.specshow(np.flipud(spectrogram), sr=sampling_rate)
            plt.xticks(time_pixel_values, labels=time_pixel_values)
            plt.yticks(frequency_pixel_values, labels=frequency_pixel_values)
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.title("After Denoising")
            save_path = os.path.join('./app_battool/static/spectrogram.png')
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # Convert the images to base64
            image_path = "./app_battool/static/spectrogram.png"
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            image_data_base64 = base64.b64encode(image_data).decode('utf-8')
            ctx = image_data_base64

            image_path = "./app_battool/static/spectrogram_beforedenoise.png"
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            image_data_base64 = base64.b64encode(image_data).decode('utf-8')
            ctx1 = image_data_base64

            return render(request, "test.html", {"image": ctx,
                                                 "denoise_image": ctx1,
                                                 "height": height,
                                                 "width": width,
                                                 "annotated_data": None,
                                                 'species_names': species_names})

        elif str(file_name).endswith('.zip'):
            with zipfile.ZipFile(file_name, 'r') as zipf:
                try:
                    extract_path = './app_battool/static/uploaded_data'
                    extract_files = {'json_file': "annotated_data.json", 'image_file': "spectrogram.png"}
                    zipf.extractall(extract_path, extract_files.values())
                except:
                    print("Incorrect or ZIP corrupted")

                try:
                    image_path = extract_path + '/' + extract_files['image_file']
                    with open(image_path, "rb") as image_file:
                        image_data = image_file.read()
                    image = Image.open(io.BytesIO(image_data))
                    width, height = image.size
                    print("Image Width:", width)
                    print("Image Height:", height)
                    image_data_base64 = base64.b64encode(image_data).decode('utf-8')
                    ctx = image_data_base64
                except:
                    print("Incorrect or image corrupted")

                try:
                    json_file = open(extract_path + '/' + extract_files['json_file'])
                    annotated_data = json.load(json_file)
                    print(annotated_data)
                    return render(request, "test.html", {"image": ctx,
                                                         "annotated_data": annotated_data,
                                                         "height": height,
                                                         "width": width,
                                                         'species_names': species_names})
                except:
                    print("Incorrect or json corrupted")
        else:
            print("Invalid file type uploaded")
            return render(request, "test.html")

    return render(request, "test.html")


@csrf_exempt  # Add this decorator to allow CSRF-exempt requests (for simplicity; consider other security measures)
def add_species(request):
    if request.method == 'POST':
        new_species = request.POST.get('new_species')
        new_species = new_species.title()

        print(new_species)

        if not BatSpecies.objects.filter(name=new_species).exists():
            BatSpecies.objects.create(name=new_species)
            return JsonResponse({'status': 'success', 'message': 'Species added successfully'})
        else:
            return JsonResponse({'status': 'error', 'message': 'Species already exists'})

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})
