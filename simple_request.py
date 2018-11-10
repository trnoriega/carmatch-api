"""Script to make a simple request to the flask server."""

import requests

KERAS_REST_API_URL = 'http://localhost:5000/predict'
IMAGE_PATH = 'Dodge-Ram_Pickup_3500-2009.jpg'

# Load the input image and construct the payload for the request:
image = open(IMAGE_PATH, 'rb').read()
payload = {'image': image}

# Submit the request:
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# Ensure the request was successful:
if r['success']:
    # Loop over the predictions and display them
    for (i, result) in enumerate(r['predictions']):
        print(f'{i + 1}. {result["label"]}: {result["probability"]:.4f}')
else:
    print('Request failed')
