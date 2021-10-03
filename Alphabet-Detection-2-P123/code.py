import cv2 
import numpy as np 
import pandas as pd
import seaborn as sns
import matploblib.pyplot as plt 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from skearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and 
	getattr(ssl, '_create_unverified_context', None)):
	ssl.create_default_https_context = ssl._create_unverified_context

#Downloading data
X = np.load('image.npz')['arr_0']
y = pd.read_csv("data.csv")['labels']
print(pd.Series(y).value_counts())
print(len(y))

#Splitting the data and scaling it 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

#Scaling features 
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0


#Fitting the data into the model 
clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scaled, y_train)

#Finding the accuracy of the model 
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("the accuracy is :", accuracy)

#Starting the camera
cap = cv2.VideoCapture(0)

while(True):
	try:
		ret, frame = cap.read()

		gray = cv2.cvtColor(frame, cv2.COLOUR_BG2GRAY)

		#Drawing a box in the center of the video 
		height, width = gray.shape 
		upper_left = (int(width / 2 - 56), int(height / 2 - 56))
		bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
		cv2.rectangle(gray, upper_left, bottom_right, (0, 25, 0), 2)

		#To only consider the area inside the box for detecting the digit 
		#roi = Region of Interest
		roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

		#Converting cv2 image pil format 
		im_pil = Image.fromarray(roi)

		#convert to grayscale image 

		image_bw = im_pil.convert('L')
		image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)

		image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
		pixel_filter = 20
		min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
		image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
		max_pixel = np.max(image_bw_resized_inverted)

		#Converting data to array 
		image_bw_resized_inverted_scaled = np.asrray(image_bw_resized_inverted_scaled)/max_pixel

		#Creating test sample and making a prediction 
		test_sample = np.array(image_bw_resized_inverted_scaled).reshap(1, 784)
		test_pred = clf.predict(test_sample)
		print("Predicted class is:	", test_pred)

		#Display
		cv2.imshow('frame', gray)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	except Exception as e:
		pass


#Release the capture 
cap.release()
cv2.destroyAllWindows()