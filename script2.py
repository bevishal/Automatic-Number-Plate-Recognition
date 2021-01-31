# Importing necessary packages
import cv2
import os
import sys
import matplotlib.pyplot as plt 
import numpy as np
import random
from matplotlib import gridspec
from collections import Counter
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder

# This function locates the plate by using Haar cascade classifier and returns the cropped plate image
def extract_plate(img):
	image_copy = img.copy()
	
	# Loading the pretrained haar cascade classifier
	plate_cascade = cv2.CascadeClassifier('./indian_license_plate.xml')

	# Detecting the candidate number plates and storing their co-ordinates
	plate_rect = list(plate_cascade.detectMultiScale(image_copy, scaleFactor = 1.3, minNeighbors = 7))
	
	# In case classifier is unable to detect the plate, we stop the program
	if len(plate_rect)==0:
		print('Sorry! Unable to detect plate for this image.')
		sys.exit()
		return -1
	
	# Sorting in decreasing order of area (w*h) to get the largest bounding rectangle
	plate_rect.sort(reverse=True,key=lambda x:x[2]*x[3])
	x,y,w,h = plate_rect[0]
	
	# Adjusting the co-ordinates so that the whole plate gets cropped
	x-=35 ; w+=70
	a,b = 2,2
	plate = image_copy[y+a:y+h-a, x+b:x+w-b, :]
	
	# Finally representing the detected contours by drawing rectangles around the edges.
	cv2.rectangle(image_copy, (x,y), (x+w, y+h), (51,255,25), 2)
	
	# Showing the detected plate with cropped plate
	cv2.imshow("Detected Plate", image_copy) 
	cv2.imshow("Cropped Plate", plate) 
	cv2.waitKey(0)
	
	# Returning the plate detected image and cropped plate image with its cordinates
	return image_copy,plate,x,y

# This function processes the plate image for character segmentation
def process_plate(plate_image):
	
	# Applying different filters to make it easy to detect contours of characters in the plate
	# These steps were found to be the most effective-
	# Cropped plate-->Grayscale-->Gaussian Blur-->Adaptive Thresholding-->Morphological Opening

	gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)
	binary = cv2.threshold(blur, 255, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	thre_mor = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel3)
	thresh2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 21, 5)
	opened = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel3)

	# Displaying various filtered plates
	cv2.imshow('Grayscale',gray)
	cv2.imshow('Gaussian Blurred',blur)
	cv2.imshow('Binary',binary)
	cv2.imshow('Eroded',thre_mor)
	cv2.imshow('Adaptive Thresholded',thresh2)
	cv2.imshow('Morphologically Opened',opened)
	cv2.waitKey(0)

	# Returning the processed plate
	return opened

# This function sorts the characters according to their x-coordinate i.e. from left to right
def sort_contours(cnts):
	# Creating bounding box for every character contour
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][0]))
	return cnts

# This function detects the contours of all characters and returns a list containing all the cropped characters
def character_segmentation(plate_image,processed_plate):
	try:	
		# Creating copies of plate_image to draw character contours and their bounding boxes
		test_roi = plate_image.copy()
		test_box = plate_image.copy()

		# Finding contours in the processed plate
		cont, _  = cv2.findContours(processed_plate, cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)

		# Drawing the contours in the copy image
		_=cv2.drawContours(test_roi,cont,-1,(255,0,255),2)

		# Initializing several lists-
		char_list=[]                # To store co-ordinates of all possible characters
		height_list=[]				# To store heights of all characters
		crop_characters=[]			# To store cropped images of all characters 

		# Here some approximations are used to filter out the character contours from other contours
		# Traversing in the sorted list of all contours
		for i,c in enumerate(sort_contours(cont)):
			# Getting co-ordinates of contours
			(x,y,w,h) = cv2.boundingRect(c)
			
			# Selecting contours with defined h/w ratio
			ratio=h/w
			if 1<=ratio<=10:

				# Selecting contours which have height larger than 50% of the plate but less than whole height
				if 0.35<=h/plate_image.shape[0]<0.9:             
					
					# To avoid redundant characters in case of O and D because inner and outer regions are detected as separate contours
					if len(char_list)>1:
						
						# If absolute difference of x-coodinates of two contours is less than 8 pixels than we simply skip the second one
						if abs(x-char_list[-1][0])<8:continue

					# Appending those contours which satisfies the above approximations
					char_list.append((x,y,w,h))
					height_list.append(h)

		# Still there might be some non-character contours present in char_list
		# Filtering out by selecting only those contours which are approximately same height as most of the contours present in the char_list
		apx_height=Counter(height_list).most_common()[0][0]
		for x,y,w,h in char_list:

			# Selecting only those which are approximately equal to the apx_height
			if apx_height-3<=h<=apx_height+3:
				# Cropping characters from processed_plate
				curr_num = processed_plate[y:y+h,x:x+w]    # pehle binary tha
				curr_num = cv2.resize(curr_num, dsize=(30,60))
				 # Removing blurness from characters
				_, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
				# Storing characters in the final list
				crop_characters.append(curr_num)
				# Drawing bounding boxes around detected charcters
				cv2.rectangle(test_box, (x, y), (x + w, y + h), (0, 255,0), 2)

		# Displaying contours and bounding boxes of characters in the plate
		cv2.imshow('Contours',test_roi)
		cv2.imshow('Bounding Boxes',test_box)
		cv2.waitKey(0)

		# Returning final list of characters
		return crop_characters

	# In case of error in detecting plate
	except:
		print("Sorry! The plate isn't detected properly")
		sys.exit()

# This function loads the MobileNets model pretrained on ImageNet dataset and fine tuned for detecting characters
def load_model():

	# Loading model architecture
	json_file = open('MobileNets_character_recognition.json','r')
	loaded_model_json = json_file.read()
	json_file.close()

	# Loading weights
	model = model_from_json(loaded_model_json)
	model.load_weights("License_character_recognition_weight.h5")
	print("[INFO] Model loaded successfully...")

	# Loading labels
	labels = LabelEncoder()
	labels.classes_ = np.load('license_character_classes.npy')
	print("[INFO] Labels loaded successfully...")

	return model,labels

# This function predicts the character in the image
def predict_from_model(image,model,labels):

	# Reshaping numpy array according to the input layer of the model
	image = cv2.resize(image,(80,80))

	# Transforming image to 3-D because the model takes 3-D images as input
	image = np.stack((image,)*3, axis=-1)

    # Inverse transforming the prediction to get its label
	prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])

	# Returning the prediction as a single character array
	return prediction

# This is the main function
def main(img):
	# Path of the image whose plate is to be detected
	path = './Dataset/'+img
	
	# Reading the image
	image = cv2.imread(path)

	# Displaying the image
	cv2.imshow("Initial Image", image)
	cv2.waitKey(0) 
	
	# Extracting the plate and its details from the image
	detected_image,plate_image,x,y = extract_plate(image)

	# Processing the plate before character segmentation
	processed_plate = process_plate(plate_image)

	# Performing character segmentation on processed_plate
	crop_characters = character_segmentation(plate_image,processed_plate)

	# Loading pre-trained model and labels
	model,labels = load_model()

	# Creating a figure to display segmented characters
	fig = plt.figure(figsize=(10,3))
	fig.suptitle('Segmented Characters')
	cols = len(crop_characters)

	# If all characters aren't detected properly we stop the program
	if cols<6:
		print("Sorry! The plate isn't detected properly")
		sys.exit()

	# Creating grid to hold subplots(character images)
	grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

	# Finally predicting each character image and displaying its predicted label as the title of subplot
	final_string = ''
	for i,character in enumerate(crop_characters):
	    fig.add_subplot(grid[i])
	    title = np.array2string(predict_from_model(character,model,labels))
	    plt.title('{}'.format(title.strip("'[]"),fontsize=20))
	    final_string+=title.strip("'[]")
	    plt.axis(False)
	    plt.imshow(character,cmap='gray')
	plt.show()

	# Displaying the final output image
	detected_image=cv2.putText(detected_image,final_string,(x+50,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
	print(final_string)
	cv2.imshow('Final Image', detected_image)
	cv2.waitKey(0)

if __name__=='__main__':

	# Setting path of the dataset
	dataset_path=os.path.join(os.getcwd(),'Dataset')

	# Acquiring all the images in the dataset
	image_list=os.listdir(dataset_path)

	# Selecting a random image from our dataset
	n=random.randrange(0,len(image_list))
	random_image=image_list[n]
	print(random_image)
	# Passing random_image for number plate recognition
	main(random_image)