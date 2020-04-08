# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import os
import json

from peopleCounter_terminal import peopleDetector

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)

detector = peopleDetector()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to


@app.route("/")
def index():
	# return the rendered template
	
	return render_template("index.html", counting=str(detector.peopleCount))

def detectFunction():
	global detector
	dir_n = os.getcwd()    
	jsonpath = dir_n + "/data.json"
	with open(jsonpath) as data:
		DATA_INPUT = json.load(data)
	zonesList = [ zone for zone in DATA_INPUT['Zones'] ]
	detector.processVideoStream(zonesList, 0, TIME_SCHEDULER = 10)
	#detector.processVideoStream(zonesList, '/home/pi/test_crosswalk.webm', DISPLAY_IMG=False, TIME_SCHEDULER=10)
    #detector.processVideoStream("/home/pi/ssd_mobilenet_v1_coco_2018_01_28/dashcam2.mp4", DRAW_ZONES = True)
    

def generate():
	# grab global references to the output frame and lock variables
	global detector

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with detector.lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if detector.outputImage is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", detector.outputImage)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detectFunction, args=())
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)