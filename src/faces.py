import cv2
import glob
import numpy as np
import os.path                                                                                                                                  
import csv

from face_detection import find_faces


def analyze_picture(model_emotion, model_gender, path, file_name, model):

    result_gender = ''
    result_emotion = ''
    result_faces = 0

    path += file_name
    #print(path)
    image = cv2.imread(path, 1)
    for normalized_face, (x, y, w, h) in find_faces(image):
        result_faces += 1
        emotion_prediction = model_emotion.predict(normalized_face)
        gender_prediction = model_gender.predict(normalized_face)
        if (gender_prediction[0] == 0):
            cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
        else:
            cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(image, emotions[emotion_prediction[0]], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        result_gender = gender_prediction[0]
        result_emotion = emotions[emotion_prediction[0]]

        with open('../data/results/results.csv', mode='a', newline='') as result_file:  
            results_writer = csv.writer(result_file, delimiter=',')
            results_writer.writerow([model, file_name, result_faces, result_gender, result_emotion])

    if not os.path.exists('../data/results/%s' % model):
        os.makedirs('../data/results/%s' % model)

    cv2.imwrite("../data/results/%s/%s" % (model, file_name), image)


def process_images(models):

    for model in models:
        #images = glob.glob('../data/sample/*.jpg')        
        images = [os.path.basename(x) for x in glob.glob('../data/sample/*.jpg')]
        
        # Load model
        fisher_face_emotion = cv2.face.FisherFaceRecognizer_create()
        fisher_face_emotion.read('models/emotion_classifier_model_all_%s.xml' % model)

        fisher_face_gender = cv2.face.FisherFaceRecognizer_create()
        fisher_face_gender.read('models/gender_classifier_model_%s.xml' % model)

        print(model)
        
        for file_name in images:
            #print(image)
            #print(model)
            analyze_picture(fisher_face_emotion, fisher_face_gender, '../data/sample/', file_name, model)

if __name__ == '__main__':
    emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]
    models = ["02", "05", "10", "35", "70"]

    process_images(models)