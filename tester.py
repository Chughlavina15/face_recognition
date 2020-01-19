import cv2
import os
import numpy as np
import face1 as fr

test_img=cv2.imread('/Users/Lavina/PycharmProjects/face_recog/face_re/test_images/dc2.jpg')  # function used to load the required images
faces_detected,gray_img=fr.faceDetection(test_img)
print("Faces Detected:",faces_detected)

#for (x,y,w,h) in faces_detected:  #trying to drawing rectangles around the faces
 #   cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)  #mentioning the diagonal co_ordinates,color and thickness of rectangle

#resized_img=cv2.resize(test_img,(1000,700)) #we resize our test image so that it fits in the given window size

#cv2.imshow("Face Detection tutorial",resized_img)    #method is used to display an image in a window
#cv2.waitKey(0)  #wait indefinitely until a key is pressed
#cv2.destroyAllWindows()  #destroys all the windows we created


faces,faceid= fr.labels_for_training_data('/Users/Lavina/PycharmProjects/face_recog/face_re/training_images')      #to call labels from the training data and generated labels
face_recognizer=fr.train_classifier(faces,faceid)    #will return the face recognizer of the image and we trained our classifier
face_recognizer.save('training_images.yml')
#face_recognizer=cv2.face.LBPHFaceRecognizer_create() #here we are not loading our classifier just taining our classifier with the alreadt trained image (yml file) of priyanka chopra
#face_recognizer.read('/Users/Lavina/PycharmProjects/face_recog/training_images.yml')
name={0:"Deepika Padukone", 1:"Priyanka Chopra",2:"Salman Khan"}     # a dictionary is created to tell which folder(present in the training images folder) contains whose images

for face in faces_detected:    #for images with multiple faces
    (x,y,w,h)=face      # co-ordinates of the face detected from the image
    roi_gray=gray_img[y:y+h,x:x+h]        # extracts the part of the face from the gray image, co-ordinates of gray image are given here
    # predict function returns one of the labels form 0(Deepika Padukone) and 1(Priyanka Chopra) and also the confidence value with which it is detetected
    label,confidence=face_recognizer.predict(roi_gray)
    print("Confidence Value:",confidence)
    print("Label:", label)
    fr.draw_rect(test_img,face)      # creates a bounding rectangle around all the detected images
    predicted_name=name[label]       # now we wnat to extract the name of the label from the name dictionary
    #if (confidence>80):  #higher confidence level means lesser is the accuracy
     #   continue
    fr.put_text(test_img,predicted_name,x,y)    # we want label at top left corner thus we passs x and y

# to show the image output:
resized_img=cv2.resize(test_img,(1000,700)) #we resize our test image so that it fits in the given window size

cv2.imshow("Face Detection tutorial",resized_img)    #method is used to display an image in a window
cv2.waitKey(0)  #wait indefinitely until a key is pressed
cv2.destroyAllWindows()  #destroys all the windows we created
