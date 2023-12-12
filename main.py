import json
from flask import Flask, request, jsonify
import tensorflow as tf
import keras_cv
from function import *
import datetime
import io
from PIL import Image
import requests

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    original_file_name = 'original_image' + '_' + time_now + '.jpg'

    bucket_name = "chickmed"
    raw_image = "raw_images/" + original_file_name
    project_name = "chickmed"

    # Configure bucket and blob
    client = storage.Client(project=project_name)
    bucket = client.bucket(bucket_name)

    # Load the image from the POST request
    im = request.files['image']
    user_id = request.form.get('user_id')

    im = Image.open(im)

    # Convert image to bytes
    bs = io.BytesIO()
    im.save(bs, "JPEG")
    bs.seek(0)

    blob = bucket.blob(raw_image)
    blob.upload_from_file(bs, content_type="image/jpeg")
    blob.make_public()
    raw_image_url = blob.public_url


    # GCP Storage settings
    gcp_bucket_name = 'chickmed'

    # Read the image from GCP Storage
    image = read_image_from_bucket(gcp_bucket_name, raw_image)

    # Load model from bucket
    blob_model = bucket.blob('models/model.h5')
    blob_model.download_to_filename('tmp/model.h5')

    model = load_model('tmp/model.h5')

    processed_file_name = 'processed_images/process_image' + '_' + time_now + '.jpg'

    results, image_processed = draw_prediction(image, model)
    # Upload the image to GCP Storage
    image_processed_url = upload_image_to_bucket(
        gcp_bucket_name, processed_file_name, image_processed)

    message = {
        'user_id': user_id,
        'status': 200,
        'message': 'OK',
        'data': results,
        'processed_image': image_processed_url,
        'raw_image': raw_image_url,
        'date': time_now
    }

    url = "http://127.0.0.1:8000/api/reports/store"
    response = requests.post(url, json=json.dumps(message))

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # store_to_db(message, image_processed_url)
        return jsonify("success")
    else :
        return jsonify("failed")


if __name__ == '__main__':
    app.run()
