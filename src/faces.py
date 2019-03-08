#import dependences
import cv2
import glob
import numpy as np
import os.path                                                                                                                                  
import csv

from face_detection import find_faces

#detect faces and recognize the gender and emotion 
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

#detect faces and recognize emotion
def analyze_picture_emotion(model_emotion, path, file_name, model):

    result_emotion = ''
    result_faces = 0

    path += file_name

    image = cv2.imread(path, 1)
    for normalized_face, (x, y, w, h) in find_faces(image):
        result_faces += 1
        emotion_prediction = model_emotion.predict(normalized_face)
        cv2.putText(image, emotions[emotion_prediction[0]], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        result_emotion = emotions[emotion_prediction[0]]

        with open('../data/results/results.csv', mode='a', newline='') as result_file:  
            results_writer = csv.writer(result_file, delimiter=',')
            results_writer.writerow([model, file_name, result_faces, result_emotion])

    if not os.path.exists('../data/results/%s' % model):
        os.makedirs('../data/results/%s' % model)

    cv2.imwrite("../data/results/%s/%s" % (model, file_name), image)

#detect faces and recognize gender
def analyze_picture_gender(model_gender, path, file_name, model):
    result_gender = ''
    result_faces = 0

    path += file_name
    #print(path)
    image = cv2.imread(path, 1)
    for normalized_face, (x, y, w, h) in find_faces(image):
        result_faces += 1
        gender_prediction = model_gender.predict(normalized_face)
        if (gender_prediction[0] == 0):
            cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
        else:
            cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)

        result_gender = gender_prediction[0]

        with open('../data/results/results.csv', mode='a', newline='') as result_file:  
            results_writer = csv.writer(result_file, delimiter=',')
            results_writer.writerow([model, file_name, result_faces, result_gender])

    if not os.path.exists('../data/results/%s' % model):
        os.makedirs('../data/results/%s' % model)

    cv2.imwrite("../data/results/%s/%s" % (model, file_name), image)


#test all the models 
def process_images(models):

    for model in models:
        images = [os.path.basename(x) for x in glob.glob('../data/testing/*.jpg')]
        
        # Load model
        if model[1] == 1:        
            fisher_face_emotion = cv2.face.FisherFaceRecognizer_create()
            fisher_face_emotion.read('models/emotion_classifier_model_%s.xml' % (model[0]))
        else:
            fisher_face_gender = cv2.face.FisherFaceRecognizer_create()
            fisher_face_gender.read('models/gender_classifier_model_%s.xml' % model[0])        
        
        print(model)
        
        for file_name in images:
            #print(image)
            #print(model)
            if model[1] == 1:
                analyze_picture_emotion(fisher_face_emotion, '../data/testing/', file_name, model[0])    
            else:
                analyze_picture_gender(fisher_face_emotion, '../data/testing/', file_name, model[0])

if __name__ == '__main__':
    emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]
    models = [  ["all_multiple_02", 1],
                ["all_multiple_05", 1],
                ["all_multiple_10", 1],
                ["all_multiple_35", 1],
                ["all_multiple_70", 1],
                ["all_straight_02", 1],
                ["all_straight_05", 1],
                ["all_straight_10", 1],
                ["all_straight_35", 1],
                ["all_straight_70", 1],
                ["female_multiple_02", 1],
                ["female_multiple_05", 1],
                ["female_multiple_10", 1],
                ["female_multiple_35", 1],
                ["female_multiple_70", 1],
                ["female_straight_02", 1],
                ["female_straight_05", 1],
                ["female_straight_10", 1],
                ["female_straight_35", 1],
                ["female_straight_70", 1],                                
                ["male_multiple_02", 1],
                ["male_multiple_05", 1],
                ["male_multiple_10", 1],
                ["male_multiple_35", 1],
                ["male_multiple_70", 1],
                ["male_straight_02", 1],
                ["male_straight_05", 1],
                ["male_straight_10", 1],
                ["male_straight_35", 1],
                ["male_straight_70", 1],
                ["multiple_02",2],
                ["multiple_05",2],
                ["multiple_10",2],
                ["multiple_35",2],
                ["multiple_70",2],
                ["straight_02",2],
                ["straight_05",2],
                ["straight_10",2],
                ["straight_35",2],
                ["straight_70",2]]                                
    process_images(models)