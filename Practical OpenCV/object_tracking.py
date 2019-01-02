import cv2 
import sys 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':

    #setup tracker
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()
    
    #read video
    video = cv2.VideoCapture('./image/Alberto Mardegan-Selfie del futuro.ogv')

    #exit if video not open
    if not video.isOpend():
        print('Could not open video')
        sys.exit()
    
    #read first frame
    ok, frame = video.read()
    if not ok:
        print('Can not read video file')
        sys.exit()

    #define an initial bounding box
    bbox = (287, 23, 86, 320)

    #uncomment the line below to select different box
    bbox = cv2.selectROI(frame, False)

    #initial tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        #read new frame
        ok, frame = vidoe.read()
        if not ok:
            break
        
        #start timer
        timer = cv2.getTickCount()
    
        #update tracker
        ok, bbox = tracker.update(frame)

        #caculate frame per second
        fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)

        #draw bounding box
        if ok:
            #tracking success
            p1 = (int(bbox[0]), int(bbow[1]))
            p2 = (int(bbox[0] + bbox[2]),int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            #tracking fail
            cv2.putText(frame, 'Tracking failue', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)
        
        #Display tracker type on frame

        #display fps on frames

        #display result
        cv2.imshow('Tracking', frame)

        #exit if ESC press
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
