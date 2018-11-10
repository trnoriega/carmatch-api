"""
Based on:
https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
and,
https://github.com/tensorflow/tensorflow/issues/14356#issuecomment-385962623

To run:
> python run_keras_server.py

To test:
> curl -X POST -F image=@Dodge-Ram_Pickup_3500-2009.jpg http://localhost:5000/predict
or
> python simple_request
"""

from collections import Counter
import io
import json
import os
from typing import Dict, List, Tuple

import flask
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import tensorflow as tf

DATA_DIR = os.path.join(os.getcwd(), 'data')
WEIGHT_PATH = os.path.join(DATA_DIR,
                           'InceptionV3_21_FT3_30_40.h5')
JSON_PATH = os.path.join(DATA_DIR,
                         'InceptionV3_21_FT3_30_40.json')
LOOKUP_PATH = os.path.join(DATA_DIR,
                           'lookup.json')

PERCENT_CUTOFF = 95
NUM_RESULTS = 5
IMG_WIDTH, IMG_HEIGHT = 299, 299

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
graph = None
lookup = None


def load_model():
    """Loads Keras model, label lookup_dict and TensorFlow graph."""

    global model
    with open(JSON_PATH, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(WEIGHT_PATH)

    global lookup
    lookup = _load_lookup()

    global graph
    graph = tf.get_default_graph()


def _load_lookup() -> Dict[int, str]:
    """Returns prediction lookup dictionary."""
    with open(LOOKUP_PATH, 'r') as f:
        dictionary = json.load(f)
    dictionary = {int(key): value for key, value in dictionary.items()}
    return dictionary


@app.route("/predict", methods=["POST"])
def predict() -> flask.Response:
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # Pre-process the image and prepare it for classification:
            image = _prepare_image(image)

            # Classify the input image and then initialize the list
            # of predictions to return to the client:
            with graph.as_default():
                predictions = _load_predictions(image)

            data["predictions"] = []

            # Loop over the results and add them to the list of
            # returned predictions:
            for label, probability in predictions:
                r = {"label": label, "probability": float(probability)}
                data["predictions"].append(r)

            # Indicate that the request was a success:
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


def _prepare_image(image: Image) -> np.ndarray:
    """
    Converts image found in image_path to a numpy array that
    can be used by Keras model to make a prediction
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize the input image and pre-process it:
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.
    return image


def _load_predictions(image: np.ndarray) -> List[Tuple[str, float]]:
    """Loads predictions made by model

    :param image: Image to predict.
    :return: Ranked list of labels and their probabilities.
    """
    prediction = model.predict(image)[0]

    # Indices sorted from most to least likely predictions:
    sorted_indices = np.argsort(prediction)[::-1]

    # Sort labels, probabilities and bodies associated with the prediction:
    labels = []
    probs = []
    bodies = []
    for i in sorted_indices:
        label = lookup.get(i, {}).get('label', 'unknown')
        labels.append(label)
        probs.append(prediction[i])
        body = lookup.get(i, {}).get('body_style', 'NA')
        bodies.append(body)

    # Find the most common bodies within the probability PERCENT_CUTOFF:
    i = 0
    total_prob = 0
    top_bodies = []
    while total_prob < PERCENT_CUTOFF/100:
        top_bodies.append(bodies[i])
        total_prob += probs[i]
        i += 1
    body_count = Counter(top_bodies)
    most_common_bodies = body_count.most_common(2)
    most_common_bodies = [tup[0] for tup in most_common_bodies]

    # Select only those labels whose bodies match the most_common_bodies:
    consensus_labels = []
    consensus_probabilities = []
    for j, label in enumerate(labels):
        if bodies[j] in most_common_bodies:
            consensus_labels.append(label)
            consensus_probabilities.append(probs[j])

    consensus = list(zip(consensus_labels, consensus_probabilities))

    # Truncate results based on NUM_RESULTS:
    truncated_consensus = consensus[:NUM_RESULTS]

    return truncated_consensus


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run()
