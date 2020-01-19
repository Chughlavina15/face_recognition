import cv2
import os
import numpy as np

def faceDetection(test_img): #returns image and gray images with rectangles
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier('/Users/Lavina/PycharmProjects/face_recog/face_re/Haar Cascade/haarcascade_frontalface_default.xml')
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)
    return faces,gray_img #when our classifier will be drained we'll need the gray image

def labels_for_training_data(directory):  #creates labels for the trained data,argument passed is a directory containing images
    faces=[]
    faceid=[]

    for path,subdirnames,filenames in os.walk(directory):   #os.walk() works recursively to move into sub-directories until it reaches the files
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system filens")
                continue

            id=os.path.basename(path)   #stores the path of the directory
            img_path=os.path.join(path,filename)       #joins the filename with the path
            print("Image Path:",img_path)
            print("Id:",id)
            test_img=cv2.imread(img_path)   #loads an image from the specified file
            #there is a chance that imread() might not work properly and return null, hence to handle this case,
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect,gray_img=faceDetection(test_img)

            #since we are going to train our classifier with images of only one person,we skip the images containing multiple faces
            if len(faces_rect)!=1:
                continue
            (x,y,w,h)=faces_rect[0]  #co-ordinates of the rectangle detected
            #roi is to store the region of interest that is only the face from the gray_img which is to be fed to the classifier
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)       #since our labels are of the type int, we convert them to integer datatype
            faceid.append(int(id))

    return faces,faceid #returns the part of the face from the gray image and its label (faceid)

def train_classifier(faces,faceid): #argumenets returned by the function labels_for_training_data(directory)
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()   #LBPH is the type of recognizer used
    face_recognizer.train(faces,np.array(faceid))   #this recognizer takes input as a numpy array
    return face_recognizer

def draw_rect(test_img,face):
    (x,y,w,h)=face  #extracting co-ordinates of rectangle
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)
    # we did do this in the tester file, here we are creating a special function for creating bounding rectangles around
    # the faces as we need to call this function again anad again


def put_text(test_img,text,x,y):    #puts text for the image at points x and y
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),6)        #opencv function fir inserting text






