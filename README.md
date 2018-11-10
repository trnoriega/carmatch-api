# A simple Flask API server for the CarMatch prediction model

Based on:
- https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
- https://github.com/tensorflow/tensorflow/issues/14356#issuecomment-385962623

To run:
`python run_keras_server.py`

To test:

`curl -X POST -F image=@Dodge-Ram_Pickup_3500-2009.jpg http://localhost:5000/predict`

or

`python simple_request`