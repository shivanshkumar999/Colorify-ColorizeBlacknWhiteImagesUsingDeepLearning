import cv2
import numpy as np
import argparse
import os
from PIL import Image

# set paths

DIR = r"D:\Desktop\Colorify"
prototxt = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
points = os.path.join(DIR, r"model/pts_in_hull.npy")
model = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")


# set arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, required=True, help="path for b/w image")
args = vars(ap.parse_args())


# load the convulational deep neural networks  model
print("Load Model")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)


# load the centres for a b channel quantization and rebalacing 
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2,313,1,1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1,313], 2.606, dtype = "float32")]


# load the input image
image_ = cv2.imread(args["image"])
image = cv2.resize(image_, (300,255))
scaled = image.astype("float32")/255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2Lab)

resized = cv2.resize(lab, (224,224))
L = cv2.split(resized)[0]
L -= 50

print("Coloring the Image, Have some patience.")
net.setInput(cv2.dnn.blobFromImage(L))

ab = net.forward()[0,:,:,:].transpose((1,2,0))
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]

colorized = np.concatenate((L[:, :, np.newaxis], ab), axis = 2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR)
colorized = np.clip(colorized, 0, 1)
colorized = (255*colorized).astype("uint8")

cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)

