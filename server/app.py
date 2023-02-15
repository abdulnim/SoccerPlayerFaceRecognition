from flask import Flask, request, jsonify
import util

app = Flask(__name__)



@app.route('/', methods=['GET'])
def hello_word():
    return 'Welcome'

@app.route('/hello')
def hello():
    return 'Hello abdul'

@app.route('/classify_image', methods = ['GET', 'POST'])
def classify_image():
    # util.initialize_variables()
    image_data = request.form['image_data']
    # print(image_data)
    prediction = jsonify(util.classify_image(image_data))
    return prediction


if __name__ == '__main__':
    print("Starting Python Flask Server for Footballer classification")
    util.initialize_variables()
    app.run(port=5000, debug=True)