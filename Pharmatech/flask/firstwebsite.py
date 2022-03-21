from flask import Flask, url_for, render_template, request, redirect
# import tensorflow as tf
import cv2 as cv
import numpy as np
import base64
from PIL import Image
from io import BytesIO


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about.html')
def about():
    return render_template("about.html")

@app.route('/result.html')
def result():
    return render_template("result.html")

@app.route('/upload.html',methods=['POST','GET'])
def upload():
    request_method = request.method
    if request_method == 'POST':
        image64 = request.get_data().decode('ascii').split(',')[1]
        im = Image.open(BytesIO(base64.b64decode(image64)))
        im.save("static/cropped_img/image.png",format = 'png')
        # filename = "image.png"
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template("upload.html")

@app.route('/instruction.html')
def instruction():
    return render_template("instruction.html")




if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)