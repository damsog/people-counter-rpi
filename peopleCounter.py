import cv2
import os
import dlib
from scipy.spatial import distance as dist
import numpy as np
import time
import sys
from imutils.video import FileVideoStream
from imutils.video import FPS
import threading
import json
from apscheduler.schedulers.background import BackgroundScheduler

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from utils.zones import selectPolygonZone
from utils.zones import designatedArea

#video_test = '/home/pi/ssd_mobilenet_v1_coco_2018_01_28/test4.mp4'


class peopleDetector:
    def __init__(self):
        self.outputImage = None
        self.peopleCount = 0
        self.outPermitedZone = False
        self.labels = ['background','person']
        self.allowedZones = []
        self.lock = threading.Lock()
        self.scheduler = BackgroundScheduler()
        self.color = lambda state: (0,255,0) if (state) else (0,0,255)


        # Load the model.

        #openvino people det retail
        self.net = cv2.dnn_DetectionModel('/home/pi/people_counting/person-detection-retail-0013.xml',
                                    '/home/pi/people_counting/person-detection-retail-0013.bin')

        # Specify target device.
        #net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

    def resetValues(self):
        self.outputImage = None
        self.peopleCount = 0
        self.outPermitedZone = 0
        self.labels = ['background','person']
        self.allowedZones = []


    def detectPeople(self, frame, threshold):
        idlabels, confidences,boxes = self.net.detect(frame, confThreshold=threshold)
        #print( [self.labels[idlabel[0]] for idlabel in idlabels ],confidences,boxes )
        # if not idlabels.any():
        #     return idlabels, confidences,boxes
        # else:
        return [self.labels[idlabel[0]] for idlabel in idlabels ], confidences,boxes
    
    def createTracker(self,tracker_type):
        if(tracker_type == 'KCF'):
            tracker = cv2.TrackerKCF_create()
        elif(tracker_type == 'TLD'):
            tracker = cv2.TrackerKCF_create()
        elif(tracker_type == 'MEDIANFLOW'):
            tracker = cv2.TrackerKCF_create()
        elif(tracker_type == 'MOSSE'):
            tracker = cv2.TrackerKCF_create()
        elif(tracker_type == 'Dlib'):
            tracker = dlib.correlation_tracker()
        return tracker

    def drawZones(self, jumpFrames=1 , VIDEO_PATH=0):
        #video = FileVideoStream(VIDEO_PATH).start()
        video = cv2.VideoCapture(VIDEO_PATH)
        
        MAX_ALLOWED = 4
        ALLOWED_ZONES = 0
        allowedZones = []
        print("Drawing Zones.")
        ALLOWED_ZONES = int(input("Input the number of allowed zones (green zones). 1 - 4: "))
        while( (ALLOWED_ZONES > MAX_ALLOWED) or (ALLOWED_ZONES < 1) ):
            print("Not a valid answer. type write exit to cancel. ")
            ALLOWED_ZONES = int( input(": ") )
            if(ALLOWED_ZONES == "exit"):
                sys.exit()
        
        
        for i in range( jumpFrames ):
            _,_ = video.read()
        _,paintFrame = video.read()
        for zone in selectPolygonZone(paintFrame,'green')[:ALLOWED_ZONES]:
            allowedZones.append( designatedArea(zone) )
        #video.stop()
        video.release()
        return allowedZones

    def zoneTrigger(self, zone=1):
        self.allowedZones[1].allowed = not self.allowedZones[1].allowed
    
    def processVideoStream(self, ZONES, VIDEO_PATH=0, SKIPFRAMES=1, TRACKER_CONF_THRESH = 10, HEAVY_TRACKER = False, SHOW_FPS=False, DISPLAY_IMG=False, TIME_SCHEDULER = 0, TIME="seconds"):
        
        if(TIME_SCHEDULER > 0):
            print("Setting timer every {} seconds ".format(TIME_SCHEDULER) )
            self.scheduler.add_job(lambda: self.zoneTrigger(1), 'interval', seconds=TIME_SCHEDULER, id='trigger')
            self.scheduler.start()

        self.allowedZones = [ designatedArea(inputZones) for inputZones in ZONES ]
        tracker_types = ['KCF'      , 'TLD' , 'MEDIANFLOW' , 'MOSSE', 'Dlib']
        tracker_type = tracker_types[4]

        # os.chdir(os.path.dirname(__file__))
        # os.chdir('..')
        # #dir_n = os.path.dirname(__file__)
        # dir_n = os.path.abspath(os.curdir)
        # os.chdir(os.path.dirname(__file__))
        STREAM = False
        if(VIDEO_PATH == 0):
            STREAM = True
        
        print('Live ', STREAM)
        if(STREAM):
            video = cv2.VideoCapture(VIDEO_PATH)        #this is the basic opencv video streaming.
        else:
            video = FileVideoStream(VIDEO_PATH).start()  #this is an improved version of opencv video streaming. smoother but adds lag on streamming
        #vs = VideoStream(usePiCamera=1).start()     #this is similar but uses a rpi cam. which is faster and more efficient than a webcam
        time.sleep(2.0)
        
        #videoFps = video.get(cv2.CAP_PROP_FPS)

        #videoLength = int(video.get(cv2.CAP_PROP_FRAME_COUNT) )

        DLIB_TOLERANCE = TRACKER_CONF_THRESH
        currentBoxes = []
        currentTypeList = []
        trackers = []
        maxDistance = 40
        maxDistance2 = 40
        maxDisaperance = 30

        SKIPFRAMES = SKIPFRAMES
        idFrame = 0

        ct = CentroidTracker( maxDisaperance, maxDistance )
        trackableObjects = {}

        #fps = FPS().start()

        #while video.more():
        #while video.isOpened():
        while True:
            #print('_____________',idFrame)
            if(STREAM):
                ret,frame = video.read()
                if not ret:
                    print("No more images.")
                    cv2.destroyAllWindows()
                    video.release()
                    break
            else:
                frame = video.read()

            frameHeight, frameWidth,_ = frame.shape

            frame2show = frame.copy()

            newBoxes = []
            newTypeList = []

            if trackers and HEAVY_TRACKER:
                timeI = time.time()
                currentBoxes = []
                todel = []

                for i,tracker in enumerate(trackers):
                    if(tracker_type == 'Dlib'):
                        ok = True
                        confidenceTracker = tracker.update(frame)
                        objectBox = tracker.get_position()
                        startX = int(objectBox.left())
                        startY = int(objectBox.top())
                        endX = int(objectBox.right())
                        endY = int(objectBox.bottom())
                        print('Tracker_conf',confidenceTracker)
                        if(confidenceTracker < DLIB_TOLERANCE ):
                            ok = False
                    else:
                        ok,objectBox = tracker.update(frame)
                    
                        # unpack the position object
                        startX = int(objectBox[0])
                        startY = int(objectBox[1])
                        endX = int(objectBox[0] + objectBox[2])
                        endY = int(objectBox[1] + objectBox[3])

                    if(endX > frameWidth):
                        endX = frameWidth
                    if(endY > frameHeight):
                        endY = frameHeight
                    
                    if(not ok):
                        todel.append(i)
                    else:
                        newBoxes.append((startX, startY, endX, endY))
                        currentBoxes.append((startX, startY, endX, endY))
                for i in range( len(todel) ):
                    del trackers[ todel[-i-1] ]
                    del currentTypeList[ todel[-i-1] ]
                #print(time.time() - timeI)
            if( (idFrame % SKIPFRAMES == 0) ):
                if(STREAM):
                    idFrame = 0
                

                newBoxes = []
                newTypeList = []
                #print( self.detectPeople(frame, threshold=0.5) )
                detClasses, detConfidences,detBoxes = self.detectPeople(frame, threshold=0.5)
                currentOutCount = 0
                for i in range( len(detBoxes) ):
                    center = ( int(detBoxes[i][0] + detBoxes[i][2]/2), int(detBoxes[i][1] + detBoxes[i][3]/2) )
                    if( self.allowedZones[0].contains( center ) ):
                        newBoxes.append( (detBoxes[i][0],detBoxes[i][1],detBoxes[i][0] + detBoxes[i][2], detBoxes[i][1] + detBoxes[i][3]) )
                        newTypeList.append( detClasses[i] )
                    if( not self.allowedZones[1].allowed ):
                        if( self.allowedZones[1].contains( center ) ):
                            cv2.rectangle(frame2show, (detBoxes[i][0],detBoxes[i][1]),(detBoxes[i][0] + detBoxes[i][2], detBoxes[i][1] + detBoxes[i][3]), (0,0,255) , 1)
                            currentOutCount += 1                      
                    else:
                        currentOutCount = 0
                
                self.outPermitedZone = True if currentOutCount>0 else False

                if(currentBoxes and HEAVY_TRACKER):
                    currentCenters = []
                    newCenters = []
                    if(newBoxes):
                        # Centers of the old boxes (The ones currently tracked)
                        for currentBox in currentBoxes:
                            currentCenters.append( ( int( (currentBox[0] + currentBox[2])/2 ),int( (currentBox[1] + currentBox[3])/2 ) ) )
                        for newBox in newBoxes:
                            newCenters.append( ( int( (newBox[0] + newBox[2])/2 ),int( (newBox[1] + newBox[3])/2 ) ) )
                        
                        distMatrix = dist.cdist( np.array(currentCenters), np.array(newCenters) )
                        distRows = distMatrix.min(axis=1).argsort()
                        distCols = distMatrix.argmin(axis=1)[distRows]

                        usedRows = set()
                        usedCols = set()

                        for (dRow, dCol) in zip(distRows, distCols):
                            if dRow in usedRows or dCol in usedCols:
                                continue
                            if distMatrix[dRow,dCol] > maxDistance2:
                                continue
                            usedRows.add(dRow)
                            usedCols.add(dCol)

                        unusedCols = set(range(0, distMatrix.shape[1])).difference(usedCols)
                        for dCol in unusedCols:
                            tracker = self.createTracker(tracker_type)
                            if( tracker_type == 'Dlib'):
                                drect = dlib.rectangle(newBoxes[dCol][0], newBoxes[dCol][1], newBoxes[dCol][2], newBoxes[dCol][3])
                                tracker.start_track(frame, drect)
                            else:
                                tracker.init(frame, (newBoxes[dCol][0],newBoxes[dCol][1],newBoxes[dCol][2]-newBoxes[dCol][0],newBoxes[dCol][3]-newBoxes[dCol][1]) )
                            trackers.append(tracker)
                            currentTypeList.append(newTypeList[dCol])
                            currentBoxes.append( newBoxes[dCol] ) 
                elif(HEAVY_TRACKER):
                    for newBox, newType in zip(newBoxes, newTypeList):
                        tracker = self.createTracker(tracker_type)
                        if( tracker_type == 'Dlib'):
                            drect = dlib.rectangle( newBox[0], newBox[1], newBox[2], newBox[3] )
                            tracker.start_track(frame, drect)
                        else:
                            tracker.init(frame, (newBox[0],newBox[1],newBox[2]-newBox[0],newBox[3]-newBox[1]) )
                        trackers.append(tracker)
                        currentTypeList.append(newType)
                    currentBoxes = newBoxes.copy()
                else:
                    currentTypeList = newTypeList.copy()
                    currentBoxes = newBoxes.copy()
            ctObjects, ctTypeObjects, ctBoxes = ct.update(currentBoxes, currentTypeList, frame)
            self.peopleCount = ct.nextObjectID

            for ((objectID, centroid)),((_,box)) in zip( ctObjects.items(),ctBoxes.items() ):
                    # check to see if a trackable object exists for the current
                    # object ID
                to = trackableObjects.get(objectID, None)
            
                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, centroid, (0,255,128), ctTypeObjects[objectID], ctBoxes[objectID])


                # store the trackable object in our dictionary
                trackableObjects[objectID] = to

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "{1}_{0} ".format(to.objectID,to.otype)

                
                for currentBox, currentType in zip(currentBoxes, currentTypeList):
                    cv2.rectangle(frame2show, (box[0],box[1]),(box[2], box[3]), (0,255,0) , 1)
                    cv2.rectangle(frame2show, (box[0],box[1]-10),(box[2], box[1] + 10), (0,255,0), -1)
                    cv2.putText(frame2show, text, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

            idFrame += 1


            if(DISPLAY_IMG):
                if(self.outPermitedZone):
                    cv2.putText(frame2show, "CROSSED THE RED LIGHT", (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    cv2.putText(frame2show, "{} pedestrians crossing the red ligth".format(currentOutCount), (15,28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    print("CROSSED THE RED LIGHT")
                    print("{} pedestrians crossing the red ligth".format(currentOutCount))
                cv2.polylines(frame2show, [np.array(self.allowedZones[0].points)], True, self.color(self.allowedZones[0].allowed), 2)
                cv2.polylines(frame2show, [np.array(self.allowedZones[1].points)], True, self.color(self.allowedZones[1].allowed), 2)
                cv2.imshow('peopleCounter', frame2show)
            with self.lock:
                self.outputImage = frame2show.copy()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.scheduler.remove_job('trigger')
                cv2.destroyAllWindows()
        
                if(STREAM):
                    video.release()
                else:
                    video.stop()
                break
            
            
            #fps.update()
        #fps.stop()
        #print(fps.fps())


if __name__=="__main__":
    detector = peopleDetector()

    #dir_n = os.path.dirname(__file__)
    dir_n = os.getcwd()
        
    jsonpath = dir_n + "/data.json"

    print("People Detector and Counter.")
    print("Draw a zone. (yes/no)")
    draw = input(": ")
    while(draw != "yes" and draw != "no"):
        print("Not a valid aswer. Please type yes or no to draw.")
        draw = input(": ")
    if(draw == "yes"):
        #drawnZones = detector.drawZones(35,VIDEO_PATH='/home/pi/test_crosswalk.webm')
        drawnZones = detector.drawZones(35,VIDEO_PATH=0)



        dictList = [ zone.points for zone in drawnZones ]
        # dictList = [ [{'x':zone.points[0][0],'y':zone.points[0][1]},
        # {'x':zone.points[1][0],'y':zone.points[1][1]},
        # {'x':zone.points[2][0],'y':zone.points[2][1]},
        # {'x':zone.points[3][0],'y':zone.points[3][1]}] for zone in drawnZones ]

        DATA_OUTPUT = {
            'Zones': dictList 
        }
        try:
            os.mknod(jsonpath)
        except:
            pass

        with open(jsonpath, 'w') as DATA_OUTPUTjson:
            json.dump(DATA_OUTPUT, DATA_OUTPUTjson)
        
        print("Zones were saved to: {}".format(jsonpath))
    print("Continue with the video processing. type. yes or no")
    continuea = input(": ")
    while(continuea != "yes" and continuea != "no"):
        print("Not a valid aswer. Please type yes or no to process.")
        continuea = input(": ")
    if(continuea == "yes"):

        with open(jsonpath) as data:
            DATA_INPUT = json.load(data)
        zonesList = [ zone for zone in DATA_INPUT['Zones'] ]
        #detector.processVideoStream("/home/pi/ssd_mobilenet_v1_coco_2018_01_28/dashcam2.mp4", DRAW_ZONES = True)
        #detector.processVideoStream(zonesList, 0, DISPLAY_IMG = True, TIME_SCHEDULER = 10)
        detector.processVideoStream(zonesList, '/home/pi/test_crosswalk.webm', DISPLAY_IMG = True, TIME_SCHEDULER = 10)




                



        




    
            




