from rest_framework.response import Response
from rest_framework.decorators import api_view
import tensorflow as tf
import numpy as np
import os
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt

# Create your views here.
img_size = 24
channel = 1
unique = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def process_image(img_path):
    img = tf.constant(img_path)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.decode_jpeg(img, channels=channel)
    img = tf.image.resize(img, size=[img_size, img_size])
    return img

def load_model(path):
    return tf.compat.v2.keras.models.load_model(path)

model = load_model("./models/facial-expression-v1/saved_model_3")

def predict(img_arr):
    img = process_image(img_arr)
    # plt.imsave("./test.jpeg", img)

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction)
    label = unique[np.argmax(prediction)]
    print(f"Prediction - {label} score - {np.max(score[0])}")
    return label, np.max(score[0]), np.argmax(prediction)

def predict_v2(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction)
    label = unique[prediction[0].argmax()]
    print("Prediction: {} {:.2f}% accuracy".format(label, 100 * np.max(score)))
    return label, np.max(score[0]), np.argmax(prediction)


@api_view(["POST", "GET"])
def facial_expression_analysis(request, *args, **kwargs):
    response = request.data
    if response == None:
        return Response({"msg": ""})
    else:
        data_str = response["image"]
        point = data_str.find(",")
        base64_str = data_str[point:]  # remove unused part like this: "data:image/jpeg;base64,"

        image = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(image))

        if img.mode != "RGB":
            img = img.convert("RGB")

        image_np = np.array(img)
#         plt.imsave("test.jpeg", image_np)
    
        try:
#             label, score, val = predict(image_np)
            label, score, val = predict_v2("test.jpeg")
            return Response({"label": f"{label}", "score": f"{score}", "val": f"{val}"})
        except Exception as e:
            return Response({"msg": f"{e}",})
