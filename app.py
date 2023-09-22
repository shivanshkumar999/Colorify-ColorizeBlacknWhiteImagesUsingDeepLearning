from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)

# Set paths
DIR = r"D:\Desktop\Colorify"
prototxt = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
points = os.path.join(DIR, r"model/pts_in_hull.npy")
model = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

# Load the convolutional deep neural networks model
print("Load Model")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)

# Load the centers for a b channel quantization and rebalancing
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

@app.route('/', methods=['GET', 'POST'])
def index():
    input_image_base64 = None
    output_image_base64 = None

    if request.method == 'POST':
        # Load the input image from the uploaded file
        uploaded_file = request.files['image']
        if uploaded_file.filename != '':
            image_data = uploaded_file.read()
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Process the image
            scaled = image.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2Lab)

            resized = cv2.resize(lab, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50

            print("Coloring the Image, Have some patience.")
            net.setInput(cv2.dnn.blobFromImage(L))

            ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

            L = cv2.split(lab)[0]

            colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
            colorized = cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR)
            colorized = np.clip(colorized, 0, 1)
            colorized = (255 * colorized).astype("uint8")

            # Encode images to base64 for HTML embedding
            _, input_image_encoded = cv2.imencode('.jpg', image)
            _, output_image_encoded = cv2.imencode('.jpg', colorized)

            input_image_base64 = base64.b64encode(input_image_encoded).decode('utf-8')
            output_image_base64 = base64.b64encode(output_image_encoded).decode('utf-8')

    return render_template('index.html', input_image=input_image_base64, output_image=output_image_base64)

if __name__ == '__main__':
    app.run(debug=True)
