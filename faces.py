import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name" : 1}
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    # Capture farame by frame
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.5, minNeighbors= 5)

    for(x, y, w, h) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]   # (Y Cordinate Start, Y Cordinate End) # For Gray Color
        roi_color = frame[y:y+h, x:x+w] #For Color Image

        # Recogniz? Deep Learned Model
        id_,conf = recognizer.predict(roi_gray)
        if conf >= 50 and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "my-image.png"       #Imae name with Extentions
        cv2.imwrite(img_item, roi_color) # For Saving A Image

        # < For Rectangle Shape >
        color = (204, 102, 0) #BGR 0-255--> Blue,Green,Red
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
        # </ Rectangle Shape End>

    #Display The Frame Result
    if cv2.waitKey(20) & 0xFF == ord('e'):
        break
    cv2.imshow('frame',frame)

# When All The Works Are Done
cap.release()
cv2.destroyAllWindows()