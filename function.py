import tensorflow as tf
import cv2
from io import BytesIO
from google.cloud import storage
import keras_cv
import numpy as np
import datetime


def load_model(model_path):
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={
                                           "YOLOV8Detector": keras_cv.models.YOLOV8Detector,
                                           "YOLOV8Backbone": keras_cv.models.YOLOV8Backbone
                                       },
                                       compile=False)
    return model


def read_image_from_bucket(bucket_name, blob_name):
    project_name = "chickmed"

    """Reads an image from a GCP Storage bucket."""
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download content as bytes
    content = blob.download_as_bytes()

    # Convert the content to a numpy array
    np_array = np.frombuffer(content, np.uint8)

    # Decode the image using OpenCV
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return img


def get_prediction(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.image.resize(image, (256, 256))
    image = tf.expand_dims(image, axis=0)
    predictions = model.predict(image)
    return predictions


def upload_image_to_bucket(bucket_name, blob_name, image):
    """Uploads an image from an OpenCV image object to a GCP Storage bucket."""
    storage_client = storage.Client(project="chickmed")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Convert the OpenCV image to bytes
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = BytesIO(img_encoded.tobytes())

    # Upload the image to GCP Storage
    blob.upload_from_file(img_bytes, content_type='image/jpeg')

    blob.make_public()

    url = blob.public_url

    return url


def draw_prediction(image, model):
    # compute class mapping
    class_ids = ["salmo", 'cocci', 'healthy', 'ncd',]
    class_mapping = dict(zip(range(len(class_ids)), class_ids))
    color_list = [(231, 76, 60), (52, 152, 219), (39, 231, 96), (243, 156, 18)]
    # get original image
    original_image = image

    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    width = original_image.shape[1]
    height = original_image.shape[0]

    # get scaled image
    scale_width = width / 256
    scale_height = height / 256

    # get predictions
    predictions = get_prediction(image, model)

    # get boxes, confindences and classes
    num_detections = int(predictions['num_detections'][0])
    classes = predictions['classes'][0]
    confindences = predictions['confidence'][0]
    boxes = predictions['boxes'][0]
    results = []

    for i in range(num_detections):
        class_id = int(classes[i])
        x, y, w, h = boxes[i]
        x1 = int(x * scale_width)
        y1 = int(y * scale_height)
        w = int(w * scale_width)
        h = int(h * scale_height)
        x2 = x1 + w
        y2 = y1 + h
        # check if box is out of bounds
        if x1 < 0:
            x1 = int(0 + width*0.05)
        if y1 < 0:
            y1 = int(0 + height*0.05)
        if x2 > width:
            x2 = int(width - width*0.05)
        if y2 > height:
            y2 = int(height - height*0.05)

        cv2.rectangle(original_image, (x1, y1),
                      (x2, y2), color_list[class_id], 1)

        linewidth = min(int((x2-x1)*0.2), int((y2-y1)*0.2))
        cv2.line(original_image, (x1, y1),
                 (x1+linewidth, y1), color_list[class_id], 4)
        cv2.line(original_image, (x1, y1),
                 (x1, y1+linewidth), color_list[class_id], 4)
        cv2.line(original_image, (x2, y1),
                 (x2-linewidth, y1), color_list[class_id], 4)
        cv2.line(original_image, (x2, y1),
                 (x2, y1+linewidth), color_list[class_id], 4)

        cv2.line(original_image, (x1, y2),
                 (x1+linewidth, y2), color_list[class_id], 4)
        cv2.line(original_image, (x1, y2),
                 (x1, y2-linewidth), color_list[class_id], 4)
        cv2.line(original_image, (x2, y2),
                 (x2-linewidth, y2), color_list[class_id], 4)
        cv2.line(original_image, (x2, y2),
                 (x2, y2-linewidth), color_list[class_id], 4)
        results.append({
            'class': class_id,
            'confidence': str(confindences[i]),
            'boxes': [str(x1), str(y1), str(x2), str(y2)]
        })

    # save image
    # cv2.imwrite('prediction.jpg',original_image)
    return results, original_image

# result



def store_to_db(results, id):
    database = 'chickmed'
    user = 'root'
    password = ''
    unix_socket = '/cloudsql/chickmed:asia-southeast2:chickmed'

    cnx = mysql.connector.connect(user=user, password=password,
                                  unix_socket=unix_socket,
                                  database=database)

    cursor = cnx.cursor()
    query_models = "INSERT INTO report_models (user_id,date, raw_image, result_image) VALUES (%s,%s, %s, %s)"
    value = (results['user_id'], results['date'], results['raw_image'],
             results['processed_image'])
    cursor.execute(query_models, value)

    last_id = cursor.lastrowid
    query_report_disease_models = "INSERT INTO report_disease_models (report_model_id, disease_model_id, confidence, bounding_box) VALUES (%s, %s, %s, %s)"
    for data in results['data']:
        bounding_box_merge = ' '.join(data['boxes'])
        value = (last_id, data['class'],
                 data['confidence'], bounding_box_merge)
        # Move to the next result set
        cursor.execute(query_report_disease_models, value)
    # get the last inserted id
    cnx.commit()

