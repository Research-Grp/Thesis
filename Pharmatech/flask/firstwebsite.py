"""
HANDWRITTEN TEXT RECOGNITION OF MEDICAL PRESCRIPTION
USING CONVOLUTIONAL RECURRENT NERUAL NETWORK

Kaycee L. Ballesteros
Ivee L. Jintalan
Mendrix C. Manlangit
BSCS 4 - A

This code is for the backend of the WebView application.
This program includes:
    - Server Handling
    - Image Pre-processing
    - Trained models
"""

# Imported Libraries

# Server Handling Libraries
from flask import Flask, url_for, render_template, request, redirect, session
import os
import random
import string
import time

# Image Pre-processing Libraries
from PIL import Image
from io import BytesIO
import cv2 as cv
import numpy as np
import base64

# Model Libraries
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
import pandas as pd
import difflib
import joblib
import tensorflow as tf

# start server
app = Flask(__name__)


def get_random_string(length):
    # With combination of lower and upper case
    result_str = ''.join(
        random.choice(string.ascii_letters) for _ in range(length))
    # print random string
    return result_str


app.secret_key = get_random_string(20)

# load crnn and svm models
new_model = keras.models.load_model('static/crnn/model25')
predict_model = keras.models.Model(
    new_model.get_layer(name="image").input,
    new_model.get_layer(name="dense2").output
)
svm_model = joblib.load('static/svm/model08.pkl')

# load model dependencies
drugs = pd.read_csv("static/dictionary.csv", header=None)

path_to_cropped = "static/cropped_img/"
path_to_segmented_img = "static/segmented_img/"

Categories = ['printed', 'handwritten']
padding_token = 99
image_width = 360
image_height = 60
max_len = 21
AUTOTUNE = tf.data.AUTOTUNE

characters = [' ', '!', '"', '#', '%', '&', "'", '(', ')', '*', '+', ',', '-',
              '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':',
              ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
              'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
              'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
              'y', 'z']

# function to convert characters to numbers and vice versa
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# get confidence of recognized image
def get_confidence(pred, num_to_char=num_to_char):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    decoded = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
    accuracy = float(decoded[1][0][0])
    accuracy = pow(10, -(accuracy)) * 100

    return accuracy

# counts the same number of letters in dictionary and prediction
def compare_func(drugname, prediction):
    get_sum = 0
    letters = set()
    for char in drugname:
        if char in prediction and char not in letters:
            get_sum += 1
        letters.add(char)
    return get_sum

# function that gives a suggestion based on dictionary
# uses levenshtein distance for heuristic distance
def suggest(prediction):
    suggestions = []
    max_compare = 0
    for x in drugs.iterrows():
        drugname = str(x[1][0])
        LVD = levenshtein_distance(drugname, prediction.lower())
        if LVD < 5:
            if LVD == 1 or LVD == 0:
                return ["1" + drugname.title()]
            compare = compare_func(drugname, prediction)
            max_compare = max(compare, max_compare)
            dict_ = str(int(LVD)) + drugname
            suggestions.append(dict_)

    short_suggest = []
    for drugname in suggestions:
        compare = compare_func(drugname, prediction)
        if compare > max_compare - 2:
            short_suggest.append(drugname)

    short_suggest.sort()
    suggestions.sort()
    print("short_suggest", short_suggest)
    if len(suggestions) > 4:
        short_suggest = difflib.get_close_matches(prediction.title(),
                                                  short_suggest, n=8)
        short_suggest.sort()
    print("difflib", short_suggest)
    return short_suggest

# algorithm for getting the distance between words
# taken from https://blog.paperspace.com/implementing-levenshtein-distance-word-autocomplete-autocorrect/
def levenshtein_distance(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if a <= b and a <= c:
                    distances[t1][t2] = a + 1
                elif b <= a and b <= c:
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1
    return distances[len(token1)][len(token2)]

# taken from A_K_Nain
# get decoded text from 2D matrix
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred,
                                       input_length=input_len,
                                       greedy=True)[0][0][:, :max_len]

    confidence = get_confidence(pred)

    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    print("Confidence:", confidence)
    return output_text,confidence

# resize image containing aspect ratio
def distortion_free_resize(image, img_size, to_rgb=True, svm=False):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]
    pad_width_left = 0

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
        if not svm:
            pad_width_right = pad_width_left + pad_width_right

    else:
        if not svm:
            pad_width_right = pad_width
        else:
            pad_width_left = pad_width_right = pad_width // 2
    if not svm:  # padding right
        image = tf.pad(
            image,
            paddings=[
                [pad_height_top, pad_height_bottom],
                [0, pad_width_right],
                [0, 0],
            ],
        )
    else:  # padding middle
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

# preprocess image according to model
def preprocess_image(image_path, img_size=(image_width, image_height),
                     delete=True, to_rgb=True, svm=False):
    image = tf.io.read_file(image_path)
    # remove image after upload
    if delete:
        os.remove(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size, to_rgb=to_rgb, svm=svm)
    image = tf.cast(image, tf.float32) / 255.0
    return image

# sorts contours from left to right or top-to-bottom
# taken from https://stackoverflow.com/questions/39403183/python-opencv-sorting-contours
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

# ctc layer for crnn loaded model
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1),
                                              dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1),
                                              dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred

# server handling
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about.html')
def about():
    return render_template("about.html")


@app.route('/result.html', methods=['POST', 'GET'])
def result():
    if "crop" in session and "height" in session and "width" in session:
        # instantiate list of images to be passed on html
        image_list = []
        word_cnn_predict = []
        suggestion_list = []
        word_svm_predict = []
        confidence_list = []
        svm_confidence_list = []
        contoured_img = None
        path_c = path_to_cropped + str(session["crop"])

        # read image prescription
        img = cv.imread(path_c)

        # remove session when exiting result page
        try:
            os.remove(path_c)
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        except:
            session.pop("crop")
            session.pop("height")
            session.pop("width")
            return redirect(url_for('upload'))

        # get cropped image dimensions
        img_h = img.shape[0]
        img_w = img.shape[1]

        # start benchmark time
        start_time = time.time()

        # get image prescription dimensions
        IMG_WIDTH = int(session['width'])
        IMG_HEIGHT = int(session['height'])

        # set contour parameters based on image dimensions
        if IMG_HEIGHT < 800:
            kernelx = 7
            kernely = 11
            lower_area = 300
            upper_area = 30000
            contour_height = 8
        elif 800 <= IMG_HEIGHT <= 1200:
            kernelx = 9
            kernely = 13
            lower_area = 500
            upper_area = 50000
            contour_height = 10
            if IMG_HEIGHT > 1000:
                kernelx = 10
                contour_height = 15
        elif 1200 < IMG_HEIGHT <= 1600:
            kernelx = 23
            kernely = 7
            lower_area = 1000
            upper_area = 90000
            contour_height = 15
        else:
            kernelx = 30
            kernely = 9
            lower_area = 1500
            upper_area = 130000
            contour_height = 17


        print("Image height,width: ", IMG_HEIGHT,IMG_WIDTH) # tester
        print("kernel", kernelx) # tester

        if IMG_WIDTH > 300 and IMG_HEIGHT > 700:
            conf_flag = 1
        else:
            conf_flag = 0

        if (img_h > int(IMG_HEIGHT/5) or img_w > int(IMG_WIDTH/2)) and \
                conf_flag == 1:

            # image pre-processing and morphological transformations
            blurred_img = cv.GaussianBlur(gray_img, (7, 7), 0)
            ret, threshed_img = cv.threshold(blurred_img, 0, 255,
                                             cv.THRESH_BINARY + cv.THRESH_OTSU)
            threshed_img = cv.bitwise_not(threshed_img)
            kernel = cv.getStructuringElement(cv.MORPH_RECT,
                                              (kernelx, kernely))
            img_close = cv.morphologyEx(threshed_img, cv.MORPH_CLOSE, kernel)
            img_dilation = cv.dilate(img_close, kernel, iterations=1)

            contours, hierarchy = cv.findContours(img_dilation,
                                                  cv.RETR_TREE,
                                                  cv.CHAIN_APPROX_SIMPLE)

            (contours, _) = sort_contours(contours, method="top-to-bottom")

            # apply contour
            count = 0
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                area = cv.contourArea(contour)
                if (lower_area < area < upper_area) and h > contour_height:
                    # segmented image to be predicted
                    roi = img[y:y + h, x:x + w]

                    # append to image_list for image showing in html
                    # convert image to png before converting to base64
                    _, roi_buffered = cv.imencode('.png', roi)

                    roi_img = base64.b64encode(roi_buffered).decode("utf-8")
                    roi_img = "data:image/png;base64, " + roi_img

                    image_list.append(roi_img)

                    # save image and delete
                    path_i = path_to_segmented_img + str(count) + str(
                        session["crop"])
                    cv.imwrite(path_i, roi)

                    # make image tf image
                    image = preprocess_image(path_i, delete=False)
                    image_svm = preprocess_image(path_i,
                                                 img_size=(100, 50),
                                                 to_rgb=False,
                                                 svm=True)
                    image_svm = [tf.reshape(image_svm, [-1])]
                    image_expanded = tf.expand_dims(image, 0)

                    # predict image
                    svm_pred = Categories[
                        svm_model.predict(image_svm)[0]]  # svm prediction
                    pred = predict_model.predict(image_expanded)

                    pred_text, confidence = decode_batch_predictions(pred)
                    l = [tf.reshape(image_svm, [-1])]
                    probability = svm_model.predict_proba(l)
                    svm_confidence = []

                    for ind, val in enumerate(Categories):
                        svm_confidence.append(probability[0][ind] * 100)

                    confidence_s = max(svm_confidence)
                    confidence_s = f'{confidence_s:.2f}'
                    confidence = f'{confidence:.2f}'
                    print("prediction:" + str(count + 1),
                          pred_text,
                          svm_pred)  # tester
                    # send predictions to html
                    word_cnn_predict.append(pred_text[0])
                    word_svm_predict.append(svm_pred)
                    suggestion_list.append(suggest(pred_text[0]))
                    confidence_list.append(confidence)
                    svm_confidence_list.append(confidence_s)
                    count += 1

            count = 1
            # apply bounding box
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                area = cv.contourArea(contour)
                print("area", area)
                if (lower_area < area < upper_area) and h > contour_height:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv.putText(img, f'{count}', (x, y - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
                    count += 1

            # convert image to png before converting to base64
            # pass image to through jinja template
            _, cont_buffered_img = cv.imencode('.png', img)
            contoured_img = base64.b64encode(cont_buffered_img).decode("utf-8")
            contoured_img = "data:image/png;base64, " + contoured_img
            # print(contoured_img[:310],type(contoured_img)) #tester
            w_count = np.arange(1, len(word_cnn_predict) + 1)
            print("WORD COUNT:", w_count)

        else:  # else put inside processed_img folder
            # save image and delete
            path_i = path_to_segmented_img + str(
                session["crop"])
            cv.imwrite(path_i, img)
            image = preprocess_image(path_i, delete=False)
            image_svm = preprocess_image(path_i, img_size=(100, 50),
                                         to_rgb=False, svm=True)
            image_svm = [tf.reshape(image_svm, [-1])]
            image_expanded = tf.expand_dims(image, 0)

            # convert image to png before converting to base64
            # pass image to through jinja template
            _, roi_buffered = cv.imencode('.png', img)
            roi_img = base64.b64encode(roi_buffered).decode("utf-8")
            roi_img = "data:image/png;base64, " + roi_img
            image_list.append(roi_img)

            # predict image
            svm_pred = Categories[svm_model.predict(image_svm)[0]]
            pred = predict_model.predict(image_expanded)

            pred_text,confidence = decode_batch_predictions(pred)
            l = [tf.reshape(image_svm, [-1])]
            probability = svm_model.predict_proba(l)
            svm_confidence = []

            for ind, val in enumerate(Categories):
                svm_confidence.append(probability[0][ind] * 100)
            confidence_s = max(svm_confidence)
            confidence_s = f'{confidence_s:.2f}'
            confidence = f'{confidence:.2f}'

            print("prediction:", pred_text, svm_pred)  # tester
            # send predictions to html
            word_cnn_predict.append(pred_text[0])
            word_svm_predict.append(svm_pred)
            suggestion_list.append(suggest(pred_text[0]))
            confidence_list.append(confidence)
            svm_confidence_list.append(confidence_s)
        end_time = time.time()
        print(end_time - start_time, "s")

        w_count = np.arange(1, len(word_cnn_predict) + 1)
        print("ELSE WORD COUNT:", w_count)
    else:
        return redirect(url_for('upload'))

    return render_template("result.html", cont_img=contoured_img,
                           cnn_predict=word_cnn_predict,
                           svm_predict=word_svm_predict,
                           images=image_list,
                           suggestions=suggestion_list,
                           confidence=confidence_list,
                           svm_confidence=svm_confidence_list,
                           word_count=w_count)


@app.route('/upload.html', methods=['POST', 'GET'])
def upload():
    request_method = request.method
    if request_method == 'POST':
        uri = get_random_string(13) + ".png"
        session["crop"] = uri
        img_h = request.form.get('image_h')
        img_w = request.form.get('image_w')
        session["width"] = img_w
        session["height"] = img_h
        image64 = request.form.get('base64').split(',')[1]
        print(session['width'], session['height'], "size")
        print(image64[:25], type(image64))
        im = Image.open(BytesIO(base64.b64decode(image64)))

        # save cropped image
        im.save(path_to_cropped + session["crop"], format='png')  # include
    return render_template("upload.html")


@app.route('/instruction.html')
def instruction():
    return render_template("instruction.html")

# server start
if __name__ == "__main__":
    app.jinja_env.globals.update(suggest_function=suggest)
    app.run(host="0.0.0.0", debug=True)
