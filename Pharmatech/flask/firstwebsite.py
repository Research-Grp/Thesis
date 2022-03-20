from flask import Flask, url_for, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about.html')
def about():
    return render_template("about.html")

@app.route('/upload.html',methods=['POST','GET'])
def upload():
    return render_template("upload.html")

@app.route('/instruction.html')
def instruction():
    return render_template("instruction.html")



if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)