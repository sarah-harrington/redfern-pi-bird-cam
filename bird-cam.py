import cv2
import time
from ultralytics import YOLO

class cvfeeder:
		
	def __init__(self):
		# set our time between checks in seconds
		self.delay = 4

		# bird label in Coco dataset
		self.birdLabel = 14

		# confidence standard
		self.goalConf = 0.60
		
		# Load a pretrained YOLO model (recommended for training)
		self.model = YOLO('yolov8n.pt')
		
		# Flag for using RTSP vs. USB camera
		self.use_rtsp = True
		
		# Link to connect to RTSP camera
		self.rtsp_url = ""
		
		# Variable for USB camera if found
		self.cam = None
		
		self.debug_mode = True
		
		# flag if bird is present, false on start up
		self.bird_present = False;
		
		self.found_bird_breeds = []
		self.bird_breed_already_found_today = False;
		
	def main(self):

		# open our video feed, if this fails program will exit
		self.get_camera()
		
		#run in bursts for efficiency
		last_run_time = time.time()
		
		while True:

			# Read in image from cam
			found, image = self.cam.read()
			
			# if our delay is up
			if time.time() - last_run_time > self.delay:
				
				#call model
				foundbird = self.findbird(image, found)
				
				# if we found a bird, do motor stuff
				print(f"Found bird: {foundbird}")
				
				if foundbird and not self.bird_breed_already_found_today:
					# TODO save off feed
				
				# print(results.classes)
				last_run_time = time.time()
		
		
	def get_camera(self):
		print("Searching for camera...")
		
		# Try RTSP, if none found use USB
		try:
			self.cam = cv2.VideoCapture(self.rtsp_url)
			
		except: 
			print("Failed to connect to RTSP, trying USB...")
			
			try: 
				for i in range(10):
					self.cam = cv2.VideoCapture(i)
					
					# check to see if it worked
					found, frame = self.cam.read()
					if found:
						print(f"Found camera at index {i}, connected successfully")
						break
			except: 
				print("Failed to find USB camera...")

		# If we didn't find a camera or it can't open a camera
		if self.cam == None or not self.cam.isOpened():
			print("ERROR: Unable to find camera")
			exit()
			
			
	def findbird(self, image, found):
		# local variable for if we find a bird at all
		foundbird = False
		
		# only progress if the camera read was successful
		if found:
			
			# Perform object detection on an image using the model
			results = self.model([image])
			
			# get confidence and labels arrays
			confs = results[0].boxes.conf.numpy()
			labels = results[0].boxes.cls.numpy()
			
			# print(f"labels: {labels}")
			
			if self.birdLabel in labels:
				# print("bird")
			
				# if we want to debug, show the video output
				if self.debug_mode:
					self.visualize(image)
				
				# check for confidence it's actually a bird	
				for i, label in enumerate(labels):
					if label == self.birdLabel:
						if confs[i] > self.goalConf:
							print(f"{confs[i]}% sure it's a bird")
							foundbird = True
							# for now we can break if we just see one
							break
							
						else:
							print(f"Not sure it's a bird: {confs[i]}")
			
		return foundbird
		
	# Debug method- displays what the camera is seeing and detecting
	def visualize(self, image):
		cv2.imshow("test", image)
		print(image.shape)
		cv2.waitKey(1)
		
	def time():
		pass
		
		
if __name__ == "__main__":
	print("Starting up....")
	#call to class
	obj = cvfeeder()
	obj.main()
