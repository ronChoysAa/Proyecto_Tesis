from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import flask
import io

app = flask.Flask(__name__)
model = None
train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory('images/Images', target_size=(224, 224), batch_size=1, shuffle=False)

def load_my_model():
    global model
    model = load_keras_model('modelo.h5')

def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    data = {'success': False}

    if flask.request.method == 'POST':
        if flask.request.files.get('image'):
            image_file = flask.request.files['image'].read()
            image_array = preprocess_image(io.BytesIO(image_file))
            predictions = model.predict(image_array)
            predicted_class = np.argmax(predictions)
            predicted_breed = list(train_generator.class_indices.keys())[list(train_generator.class_indices.values()).index(predicted_class)]
            data['breed'] = predicted_breed
            data['success'] = True

    return flask.jsonify(data)

if __name__ == '__main__':
    load_my_model()
    app.run()