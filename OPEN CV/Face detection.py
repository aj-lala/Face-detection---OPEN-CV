import cv2

face_cascode = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascode = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascode.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face_gray = gray[y:y+h, x:x+w]
        face_color = img[y:y+h, x:x+w]

        eyes = eye_cascode.detectMultiScale(face_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(1) & 0xff
    if k == 133:
        break

cap.release()
cv2.destroyAllWindows()

