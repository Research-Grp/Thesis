from flask import Flask, url_for, render_template, request, redirect, session
import cv2 as cv
import os
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import random
import string

import joblib
from sklearn import svm
import tensorflow as tf
import asrtoolkit as ak
from tensorflow.keras.layers import Input
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from sklearnex import patch_sklearn
patch_sklearn()


def get_random_string(length):
    # With combination of lower and upper case
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    # print random string
    return result_str

app = Flask(__name__)

new_model = keras.models.load_model('static/model09')
predict_model = keras.models.Model(
    new_model.get_layer(name="image").input, new_model.get_layer(name="dense2").output
)
svm_model = joblib.load('static/svm/model04.pkl')

path_to_cropped = "static/cropped_img/"
path_to_segmented_img = "static/segmented_img/"
app.secret_key = get_random_string(20)

Categories = ['printed', 'handwritten']
padding_token = 99
image_width = 360
image_height = 60
max_len = 21
AUTOTUNE = tf.data.AUTOTUNE
characters = ['m', 'd', 'l', 'f', 'P', '"', 'R', 'o', 'H', '8', 'W', 'n', 'N',
              'h', '*', 'I', 'y', '3', ',', 'X', '.', 'B', 'j', 'Q', ')', 'V',
              'M', 'x', '1', 'c', ':', 'T', '?', '!', 'F', "'", '9', '#', 'z',
              '6', '5', 'p', 'r', 'Y', '-', 'v', '&', 'O', 'U', '(', 'w', 'A',
              'i', 'Z', 'S', '7', 'q', 'G', 'D', 'E', 'J', 'b', '4', ' ', '/',
              '+', 'L', 'k', ';', 't', 'e', 'a', 'g', 'K', '2', '0', 's', 'u',
              '_', 'C', '%']

char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=False, beam_width=100)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def distortion_free_resize(image, img_size,to_rgb=True):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )
    if to_rgb:
        image = tf.image.grayscale_to_rgb(image)
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image_path, img_size=(image_width, image_height),delete=True,to_rgb=True):
    image = tf.io.read_file(image_path)
    #remove image after upload
    if delete:
        os.remove(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size,to_rgb=to_rgb)
    image = tf.cast(image, tf.float32) / 255.0
    return image

class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about.html')
def about():
    return render_template("about.html")

@app.route('/result.html')
def result():

    if "crop" in session:
        image_list = []#prediction storage
        image_predict = []#
        path_c = path_to_cropped+str(session["crop"])

        img = cv.imread(path_c)
        # im = Image.open(path_to_cropped+str(session["crop"]))
        os.remove(path_c)

        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_h = img.shape[0]
        img_w = img.shape[1]
        print("shape[0]h = ",img_h," shape[1]w = ",img_w)#tester

        if (img_h > 100 and img_w > 160): #if image height is greater than 100 pre-process
            print("true")
            blurred_img = cv.GaussianBlur(gray_img, (7, 7), 0)
            ret, threshed_img = cv.threshold(blurred_img, 0, 255,
                                            cv.THRESH_BINARY + cv.THRESH_OTSU)
            threshed_img = cv.bitwise_not(threshed_img)
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (31, 3))
            img_close = cv.morphologyEx(threshed_img, cv.MORPH_CLOSE, kernel)
            img_dilation = cv.dilate(img_close, kernel, iterations=1)

            contours, hierarchy = cv.findContours(img_dilation,
                                                  cv.RETR_TREE,
                                                  cv.CHAIN_APPROX_SIMPLE)
            count = 0
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                # print("height:",h," width:",w," area:",cv.contourArea(contour)) #tester
                if (cv.contourArea(contour) > 4000 and h > 50):
                    #segmented image to be predicted
                    roi = img[y:y + h, x:x + w]
                    #append to image_list for image showing in html
                    # image_list.append(roi)

                    #save image and delete
                    path_i = path_to_segmented_img + str(count) + str(
                        session["crop"])
                    cv.imwrite(path_i,roi)

                    #make image tf image
                    image = preprocess_image(path_i,delete=False)
                    image_svm = preprocess_image(path_i,img_size=(100,50)
                                                ,to_rgb=False)
                    image_svm = [tf.reshape(image_svm,[-1])]
                    image_expanded = tf.expand_dims(image, 0)

                    #predict image
                    svm_pred = Categories[svm_model.predict(image_svm)[0]]
                    pred = predict_model.predict(image_expanded)
                    pred_text = decode_batch_predictions(pred)
                    print("prediction:"+ str(count+1) ,pred_text, svm_pred)
                    count += 1

            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                if (cv.contourArea(contour) > 4000 and h > 50):
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv.imwrite("static/segmented_img/contoured_img.png",img)
        else: #else put inside processed_img folder
            print("false: smaller than 100,160")

            # save image and delete
            path_i = path_to_segmented_img + str(
                session["crop"])
            cv.imwrite(path_i,img)
            image = preprocess_image(path_i,delete=False)
            image_svm = preprocess_image(path_i, img_size=(100, 50),
                                         to_rgb=False)
            image_svm = [tf.reshape(image_svm, [-1])]
            image_expanded = tf.expand_dims(image, 0)

            # predict image
            svm_pred = Categories[svm_model.predict(image_svm)[0]]
            pred = predict_model.predict(image_expanded)
            pred_text = decode_batch_predictions(pred)

            print("prediction:", pred_text,svm_pred)

    else:
        return redirect(url_for('upload'))


    return render_template("result.html")

@app.route('/upload.html',methods=['POST','GET'])
def upload():
    request_method = request.method
    if request_method == 'POST':
        uri = get_random_string(13)+".png"
        session["crop"] = uri
        image64 = request.get_data().decode('ascii').split(',')[1]
        im = Image.open(BytesIO(base64.b64decode(image64)))

        im.save(path_to_cropped+session["crop"],format = 'png') #include
        # filename = "image.png"
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template("upload.html")

@app.route('/instruction.html')
def instruction():
    return render_template("instruction.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)