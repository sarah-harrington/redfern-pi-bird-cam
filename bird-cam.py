import cv2
import time
import RPi.GPIO as GPIO
from ultralytics import YOLO

class cvfeeder:
	
	def __init__(self):
		# set our time between checks
		self.delay = 4

		# cat label in Coco dataset
		self.catLabel = 15

		# confidence standard
		self.goalConf = 0.35

		# Load a pretrained YOLO model (recommended for training)
		self.model = YOLO('yolov8n.pt')

		# open our video feed
		self.cam = None
		
		self.debug_mode = True
		
		#pins for motor driver inputs
		self.motorpin = 4
		
		# check if food has already dispensed
		self.alreadyFed = False;

		# Setup our motors / GPIO interface
		GPIO.setmode(GPIO.BCM)     #set numbering format
		GPIO.setup(self.motorpin, GPIO.OUT) # set as an output pin
		GPIO.output(self.motorpin, GPIO.LOW) # set state to low ( so we don't have it dispensing food forever until we see a cat and set this low )

		# startup our functions
		self.camera()
		self.main()
		
	def main(self):
		
		#run in bursts for efficiency
		last_run_time = time.time()
		
		while True:
			
			# TODO time stuff goes here, if time we want to feed:
			
			# Read in image from cam
			found, image = self.cam.read()
			
			# if our delay is up
			if time.time() - last_run_time > self.delay:
				
				#call model
				foundCat = self.findCat(image, found)
				
				# if we found a cat, do motor stuff
				print(f"Found cat: {foundCat}")
				
				if foundCat and  not self.alreadyFed:
					self.alreadyFed = self.motor()
					
				if self.alreadyFed:
					exit()
				
				# print(results.classes)
				last_run_time = time.time()
		
		
	def camera(self):
		for i in range(10):
			self.cam = cv2.VideoCapture(i)
			
			# check to see if it worked
			found, frame = self.cam.read()
			if found:
				print(f"Found camera at index {i}, connected successfully")
				break

		if self.cam == None:
			print("Unable to open cam")
			exit()
			
			
	def findCat(self, image, found):
		# set our variable for if we find something
		foundCat = False
		
		# only progress if the camera read was successful
		if found:
			
			# Perform object detection on an image using the model
			results = self.model([image])
			
			# get confidence and labels arrays
			confs = results[0].boxes.conf.numpy()
			labels = results[0].boxes.cls.numpy()
			
			# print(f"labels: {labels}")
			
			if self.catLabel in labels:
				# print("cat")
			
				# if we want to debug, show the video output
				#if self.debug_mode:
					#self.visualize(image)
				
				# check for confidence it's actually a cat	
				for i, label in enumerate(labels):
					if label == self.catLabel:
						if confs[i] > self.goalConf:
							print(f"{confs[i]}% sure it's a cat")
							foundCat = True
							# for now we can break if we just see one
							break
							
						else:
							print(f"Not sure it's a cat: {confs[i]}")

			
		return foundCat
		
	def motor(self):
		#GPIO.setmode(GPIO.BCM)     #set numbering format
		#GPIO.setup(self.motorpin, GPIO.OUT)
		
		#go forward
		GPIO.output(self.motorpin, GPIO.HIGH)

		time.sleep(1)     #motor will run for x seconds

		#stop
		GPIO.output(self.motorpin, GPIO.LOW)
		GPIO.cleanup()
		
		return True
		
	
	# example method
	def visualize(self, image):
		cv2.imshow("test", image)
		print(image.shape)
		cv2.waitKey(1)
		
	def time():
		pass
		
		
if __name__ == "__main__":
	#call to class
	obj = cvfeeder()
