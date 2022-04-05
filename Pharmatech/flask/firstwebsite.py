from flask import Flask, url_for, render_template, request, redirect, session
import cv2 as cv
import os
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import random
import string
import pandas as pd
import time
import difflib
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from sklearnex import patch_sklearn

patch_sklearn()


def get_random_string(length):
    # With combination of lower and upper case
    result_str = ''.join(
        random.choice(string.ascii_letters) for _ in range(length))
    # print random string
    return result_str


app = Flask(__name__)

new_model = keras.models.load_model('static/crnn/model23')
predict_model = keras.models.Model(
    new_model.get_layer(name="image").input,
    new_model.get_layer(name="dense2").output
)
svm_model = joblib.load('static/svm/model08.pkl')

drugs = pd.read_csv("static/dictionary.csv", header=None)

path_to_cropped = "static/cropped_img/"
path_to_segmented_img = "static/segmented_img/"
app.secret_key = get_random_string(20)

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

char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


def compare_func(drugname, prediction):
    get_sum = 0
    letters = set()
    for char in drugname:
        if char in prediction and char not in letters:
            get_sum += 1
        letters.add(char)
    return get_sum


def suggest(prediction):
    suggestions = []
    max_compare = 0
    for x in drugs.iterrows():
        drugname = str(x[1][0])
        LVD = levenshtein_distance(drugname, prediction.lower())
        if LVD < 5:
            if LVD == 1:
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

    short_suggest = difflib.get_close_matches(prediction, short_suggest, n=8)
    print(short_suggest)
    return short_suggest


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


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = \
        keras.backend.ctc_decode(pred,
                                 input_length=input_len,
                                 greedy=True)[0][0][:, :max_len]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


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


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about.html')
def about():
    return render_template("about.html")


@app.route('/result.html')
def result():
    if "crop" in session:
        image_list = []  # image storage to be shown in html
        word_cnn_predict = []
        suggestion_list = []
        word_svm_predict = []
        contoured_img = None
        path_c = path_to_cropped + str(session["crop"])

        img = cv.imread(path_c)

        # noinspection PyBroadException
        try:
            os.remove(path_c)
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        except:
            session.pop("crop")
            return redirect(url_for('upload'))

        img_h = img.shape[0]
        img_w = img.shape[1]
        print("shape[0]h = ", img_h, " shape[1]w = ", img_w)  # tester

        start_time = time.time()
        if img_h > 100 and img_w > 160:  # if image height is greater
            # than 100 pre-process
            print("true")  # tester
            blurred_img = cv.GaussianBlur(gray_img, (7, 7), 0)
            ret, threshed_img = cv.threshold(blurred_img, 0, 255,
                                             cv.THRESH_BINARY + cv.THRESH_OTSU)
            threshed_img = cv.bitwise_not(threshed_img)
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (35, 3))
            img_close = cv.morphologyEx(threshed_img, cv.MORPH_CLOSE, kernel)
            img_dilation = cv.dilate(img_close, kernel, iterations=1)

            contours, hierarchy = cv.findContours(img_dilation,
                                                  cv.RETR_TREE,
                                                  cv.CHAIN_APPROX_SIMPLE)
            count = 0
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                # print("height:",h," width:",w," area:",cv.contourArea(
                # contour)) #tester
                if cv.contourArea(contour) > 4000 and h > 50:
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
                    pred_text = decode_batch_predictions(
                        pred)  # crnn prediction
                    print("prediction:" + str(count + 1), pred_text,
                          svm_pred)  # tester
                    # send predictions to html
                    word_cnn_predict.append(pred_text[0])
                    word_svm_predict.append(svm_pred)
                    suggestion_list.append(suggest(pred_text[0]))
                    count += 1

            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                if cv.contourArea(contour) > 4000 and h > 50:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # convert image to png before converting to base64
            # pass image to through jinja template
            _, cont_buffered_img = cv.imencode('.png', img)
            contoured_img = base64.b64encode(cont_buffered_img).decode("utf-8")
            contoured_img = "data:image/png;base64, " + contoured_img
            # print(contoured_img[:310],type(contoured_img)) #tester

        else:  # else put inside processed_img folder
            print("false: smaller than 100,160")

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
            pred_text = decode_batch_predictions(pred)
            print("prediction:", pred_text, svm_pred)  # tester
            # send predictions to html
            word_cnn_predict.append(pred_text[0])
            word_svm_predict.append(svm_pred)
            suggestion_list.append(suggest(pred_text[0]))
        end_time = time.time()
        print(end_time - start_time, "s")
    else:
        return redirect(url_for('upload'))

    return render_template("result.html", cont_img=contoured_img,
                           cnn_predict=word_cnn_predict,
                           svm_predict=word_svm_predict,
                           images=image_list,
                           suggestions=suggestion_list)


@app.route('/upload.html', methods=['POST', 'GET'])
def upload():
    request_method = request.method
    if request_method == 'POST':
        uri = get_random_string(13) + ".png"
        session["crop"] = uri
        image64 = request.get_data().decode('ascii').split(',')[1]
        im = Image.open(BytesIO(base64.b64decode(image64)))

        im.save(path_to_cropped + session["crop"], format='png')  # include
    return render_template("upload.html")


@app.route('/instruction.html')
def instruction():
    return render_template("instruction.html")


if __name__ == "__main__":
    app.jinja_env.globals.update(suggest_function=suggest)
    app.run(host="0.0.0.0", debug=True)
