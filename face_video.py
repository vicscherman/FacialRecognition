import cv2
import sys

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame, check to see if we've run out of frames
    check, frame = video_capture.read()
    #flip the frame so it's mirrored
    frame= cv2.flip(frame,1 )
    #render video in grey for better results
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect faces with cascade reiteration of 1.1
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
       
    )

    # Draw a rectangle around the faces. Start point x,y, width is x + w, height is y + h, 
    for (x, y, w, h) in faces:                #color is pink cause it's fun!!!
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 153, 255), 3)

    # Display the resulting frame
    cv2.imshow('Face detector', frame)

    #rerender ever 1 ms
    key=cv2.waitKey(1)
    #to exit
    if key == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()