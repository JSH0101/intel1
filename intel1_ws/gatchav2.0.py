import time
import serial
import requests
import numpy
from io import BytesIO
from pprint import pprint
from datetime import datetime
import os
import cv2

ser = serial.Serial("/dev/ttyACM0", 9600)

# API endpoint
api_url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/4054f229-a65a-4aa9-9515-c9c049fed546/inference"

# Ensure the img directory exists
img_folder = "img"
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

def get_img():
    """Get Image From USB Camera

    Returns:
        numpy.array: Image numpy array
    """

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Camera Error")
        exit(-1)

    ret, img = cam.read()
    cam.release()

    return img


def crop_img(img, size_dict):
    x = size_dict["x"]
    y = size_dict["y"]
    w = size_dict["width"]
    h = size_dict["height"]
    img = img[y : y + h, x : x + w]
    return img


def inference_reqeust(img: numpy.array, api_rul: str):
    """_summary_

    Args:
        img (numpy.array): Image numpy array
        api_rul (str): API URL. Inference Endpoint
    """
    _, img_encoded = cv2.imencode(".jpg", img)

    # Prepare the image for sending
    img_bytes = BytesIO(img_encoded.tobytes())

    # Send the image to the API
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

    print(files)

    try:
        response = requests.post(api_url, files=files)
        if response.status_code == 200:
            pprint(response.json())
            return response.json()
            print("Image sent successfully")
        else:
            print(f"Failed to send image. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")


while 1:
    data = ser.read()
    print(data)
    if data == b"0":
        img = get_img()
        # crop_info = None
        crop_info = {"x": 200, "y": 100, "width": 300, "height": 300}

        if crop_info is not None:
            img = crop_img(img, crop_info)

        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(img_folder, f"image_{timestamp}.jpg")
        cv2.imwrite(img_path, img)
        print(f"Image saved at {img_path}")

        # Send to API and write to serial
        result = inference_reqeust(img, api_url)
        ser.write(b"1")
    else:
        pass
