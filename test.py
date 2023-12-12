# from tqdm.auto import tqdm
# import xml.etree.ElementTree as ET
# import tensorflow as tf
# from tensorflow import keras
# from itertools import chain
# import cv2
# from function import *
# from google.cloud import storage
# import io
# from io import BytesIO
# from PIL import Image
# import datetime
# import numpy as np
# # import keras_cv
# import mysql.connector
# from mysql.connector import Error
import requests
import json

results = {
    "data": [
        {
            "boxes": [
                "76",
                "71",
                "609",
                "979"
            ],
            "class": 1,
            "confidence": "0.89685863"
        }
    ],
    "date": "20231211_101507",
    "message": "OK",
    "processed_image": "https://storage.googleapis.com/chickmed/processed_images/process_image_20231211_101507.jpg",
    "raw_image": "https://storage.googleapis.com/chickmed/raw_images/original_image_20231211_101507.jpg",
    "status": 200,
    "user_id": "1"
}

url = "http://127.0.0.1:8000/api/reports/store"
requests.post(url, json=json.dumps(results))
print(json.dumps(results))

# database = 'chickmed'
# user = 'root'
# password = 'ch1ckm3d_db'
# unix_socket = '/cloudsql/chickmed:asia-southeast2:chickmed'

# cnx = mysql.connector.connect(user=user, password=password,
#                                 unix_socket=unix_socket,
#                                 database=database)

# cursor = cnx.cursor()
# query_models = "INSERT INTO report_models (user_id,date, raw_image, result_image) VALUES (%s,%s, %s, %s)"
# value = (results['user_id'], results['date'], results['raw_image'],
#             results['processed_image'])
# cursor.execute(query_models, value)

# last_id = cursor.lastrowid
# query_report_disease_models = "INSERT INTO report_disease_models (report_model_id, disease_model_id, confidence, bounding_box) VALUES (%s, %s, %s, %s)"
# for data in results['data']:
#     bounding_box_merge = ' '.join(data['boxes'])
#     value = (last_id, data['class'],
#                 data['confidence'], bounding_box_merge)
#     # Move to the next result set
#     cursor.execute(query_report_disease_models, value)
# # get the last inserted id
# cnx.commit()

# host = 'localhost'
# database = 'chickmed'
# user = 'root'
# password = ''

# cnx = mysql.connector.connect(user=user, password=password,
#                                 host=host,
#                                 database=database)

# cursor = cnx.cursor()


 
# query_models = "BEGIN; INSERT INTO report_models (date, raw_image, result_image) VALUES (%s, %s, %s); INSERT INTO report_disease_models (report_model_id, disease_model_id, confidence, bounding_box) VALUES (LAST_INSERT_ID(), %s, %s, %s); COMMIT;"
# bounding_box_merge = ' '.join(results['data'][0]['boxes'])
# value = (results['date'], results['raw_image'], results['processed_image'], results['data']
#             [0]['class'], results['data'][0]['confidence'], bounding_box_merge)
# cursor.execute(query_models, value)



# # Define variables
# bucket_name = "chickmedbuckets"
# destination_blob_filename = "chickmed/prediction3.jpg"
# project_name = "chickmed"

# # Configure bucket and blob
# client = storage.Client(project=project_name)
# bucket = client.bucket(bucket_name)

# # Load the image
# im = Image.open("cocci.1.jpg")

# # Convert image to bytes
# bs = io.BytesIO()
# im.save(bs, "JPEG")
# bs.seek(0)

# blob = bucket.blob(destination_blob_filename)
# blob.upload_from_file(bs, content_type="image/jpeg")

# # GCP Storage settings
# gcp_bucket_name = 'chickmedbuckets'
# gcp_blob_name = 'chickmed/prediction3.jpg'

# Read the image from GCP Storage
# image = read_image_from_bucket(gcp_bucket_name, gcp_blob_name)

# model_2 = tf.keras.models.load_model("Best_Model_YOLOv8.h5",
# custom_objects={
# "YOLOV8Detector": keras_cv.models.YOLOV8Detector,
# "YOLOV8Backbone": keras_cv.models.YOLOV8Backbone
#         },
# compile = False)

# prediction = draw_prediction(image,model_2)

# print(type(prediction))

# def upload_image_to_bucket(bucket_name, blob_name, image):
#         """Uploads an image from an OpenCV image object to a GCP Storage bucket."""
#         storage_client = storage.Client()
#         bucket = storage_client.bucket(bucket_name)
#         blob = bucket.blob(blob_name)

#         # Convert the OpenCV image to bytes
#         _, img_encoded = cv2.imencode('.jpg', image)
#         img_bytes = BytesIO(img_encoded.tobytes())

#         # Upload the image to GCP Storage
#         blob.upload_from_file(img_bytes, content_type='image/jpeg')

# # GCP Storage settings
# gcp_bucket_name = 'chickmedbuckets'
# gcp_blob_name = 'chickmed/prediction3.jpg'

        
# width, height = 640, 480
# image = np.zeros((height, width, 3), dtype=np.uint8)  

# # Upload the image to GCP Storage
# upload_image_to_bucket(gcp_bucket_name, gcp_blob_name, image)

    

# print(type(message))