# Importing necessary packages
import cv2
import os
import sys
import matplotlib.pyplot as plt 
import numpy as np 
import pytesseract
import random

# Setting path of the dataset
dataset_path=os.path.join(os.getcwd(),'Images')

# Acquiring all the images in the dataset
image_list=os.listdir(dataset_path)

# Adding path to Tessarct OCR
pytesseract.pytesseract.tesseract_cmd=r"U:/program files/Tesseract-OCR/tesseract.exe"

# Defining Kernals/Structuring elements
kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
kernel2=cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

def main(n):
	
	# Fetching the image to be processed 
	image=image_list[n-1]
	path=os.path.join(dataset_path,image)
	
	# Reading the image
	image=cv2.imread(path)

	# Converting image to Grayscale
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	# Removing Noise by Bilateral Filtering
	blur=cv2.bilateralFilter(gray,9,75,75)

	# Histogram Equalization of image
	histEqImage=cv2.equalizeHist(blur)

	# Morphological Opening of image
	openedImage=cv2.morphologyEx(histEqImage, cv2.MORPH_OPEN, kernel, iterations=15)

	# Subtract Morphologically Opened Image from Histogram Equalized Image
	subtractedImage=cv2.subtract(histEqImage, openedImage)

	# Thresholding the Image
	_,threshImage=cv2.threshold(subtractedImage, 0, 255, cv2.THRESH_OTSU)

	# Applying Canny Edge Detection
	cannyImage=cv2.Canny(threshImage, 250, 255)

	# Dilating image for Edge Strengthening
	dilatedImage=cv2.dilate(cannyImage, kernel2, iterations=1)
	
	# Finding Contours of the Edge Dilated image, which will find edges
	cnts,new=cv2.findContours(dilatedImage,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	
	image_copy=image.copy()
	imgcopy=image.copy()

	# Here an approximation is used that the required plate is a 4-sided polygon having maximum area
	# So all the contours are sorted according to their area in decreasing order
	cnts=sorted(cnts,key=cv2.contourArea,reverse=True)
	_=cv2.drawContours(image_copy,cnts,-1,(255,0,255),2)
	plate=None
	
	# Finding contour with possible 4-sides
	for c in cnts:
		perimeter=cv2.arcLength(c,True)
		edges_count=cv2.approxPolyDP(c,0.02*perimeter,True)
		
		if len(edges_count)==4:
			x,y,w,h=cv2.boundingRect(c)
			# Cropping number plate from image
			plate=image[y:y+h,x:x+w]
			name='plate.png'
			# Saving plate in a folder
			cv2.imwrite(name,plate)
			imgcopy=cv2.rectangle(imgcopy,(x,y),(x+w,y+h),(0,255,0),2)
			break

	# Displaying various Filtered images
	cv2.imshow("Initial Image", image)
	cv2.waitKey(0)	
	cv2.imshow("Grayscale Image", gray)
	cv2.waitKey(0)
	cv2.imshow("Bilateral Filtered Image", blur)
	cv2.waitKey(0)
	cv2.imshow("Histogram Equalized Image", histEqImage)
	cv2.waitKey(0)
	cv2.imshow("Opened Image", openedImage)
	cv2.waitKey(0)
	cv2.imshow("Subtracted Image", subtractedImage)
	cv2.waitKey(0)
	cv2.imshow("Thresholded Image", threshImage)
	cv2.waitKey(0)
	cv2.imshow("Edge Detected Image", cannyImage)
	cv2.waitKey(0)
	cv2.imshow("Dilated Edge Image", dilatedImage)
	cv2.waitKey(0)
	cv2.imshow("Plate Detected Image", imgcopy)
	cv2.waitKey(0)

	# Passing image for further processing and text recognition
	text=tesseract(plate)

	# Displaying the final output image
	detected_image=cv2.putText(imgcopy,text,(x+15,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
	print(text)
	cv2.imshow('Final Image', detected_image)
	cv2.waitKey(0)

	return

def tesseract(plate):

	# As every image needs different preprocessing steps
	# These steps were found to be the most effective for our dataset-
	# Cropped Plate-->Grayscale-->Bilateral Filter-->Thresholding-->Morphological Closing
	
	# Applying different filters to the plate
	gray=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
	blur=cv2.medianBlur(gray,5)
	blur2=cv2.bilateralFilter(gray,9,75,75)
	eroded=cv2.erode(blur, kernel, iterations = 1)
	_,thres=cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	opened=cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)
	dilated=cv2.dilate(thres, kernel, iterations = 1)
	closed=cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel2)

	# Displaying various Filtered plates
	cv2.imshow('Cropped Plate',plate)
	cv2.imshow('Grayscale Plate',gray)
	cv2.imshow('Median Blurred Plate',blur)	
	cv2.imshow('Bilateral Filtered Plate',blur2)	
	cv2.imshow('Thresholded Plate',thres)
	cv2.imshow('Dilated Plate',dilated)
	cv2.imshow('Eroded Plate',eroded)
	cv2.imshow('Opened Plate',opened)	
	cv2.imshow('Closed Plate',closed)
	cv2.waitKey(0)
 
	# Saving processed plate in a folder
	cv2.imwrite('processed_plate.png',closed)
	img=cv2.imread('./processed_plate.png')
	config=('-l eng --oem 1 --psm 3')
	
	# Detecting text using PyTesseract
	text=pytesseract.image_to_string(img,config=config)
	return text

if __name__=='__main__':
	# List of names of images
	images=[1,2,4,8,11,14,15,17,18]
	# Selecting a random image for number plate recognition
	main(random.choice(images))