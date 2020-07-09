# INCASE YOU RECIEVE ANY MODULE IMPORT ERROR, KINDLY RUN THE FOLLOWING COMMAND IN YOUR TERMINAL IF YOU HAVE PYTHON ALREADY INSTALLED ON YOUR MACHINE.
# pip install numpy, opencv-python, keras
# HOPEFULLY IT SHOULD WORK NOW.

# I have pre-trained the CNN Model used for this project.

import numpy as np
import cv2
from keras.models import load_model

CLASS_LABELS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

number_plate_cascade = cv2.CascadeClassifier('indian_number_plate.xml')

# Reading the Image
def read_image(path):
    image = cv2.imread(path)
    return image

# Detecting the number plates from the Image.
# This haarcascade that I picked up is pretty erroneous and so it might sometimes fail to detect the correct number plates.
def detect_plates(original_image):
    
    detection_img = cv2.threshold(cv2.equalizeHist(cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)), 110, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    plate_img = original_image.copy()
    coordinates = number_plate_cascade.detectMultiScale(detection_img)
    
    for x,y,z,w in coordinates:
        cv2.rectangle(plate_img, (x, y), (x+z, y+w), thickness=3, color=(0, 0, 0))
    
    cropped_plates = []
    for x,y,z,w in coordinates:
        cropped_plates.append(original_image[y:y+w, x:x+z])
    
    return plate_img, cropped_plates

def sort_on_tuples(tpl):
    return tpl[0]

def apt_image_shape(img):
    x,y,z = img.shape
    if x in range(14,51) and y in range(7,25) and z == 3:
        return True
    else:
        return False

# Extracting characters from the number plates
def extract_characters(original_images):
    
    characters_list_for_all_plates = []
    
    for original_image in original_images:
        
        original_image_bw = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(original_image_bw, 110, 255, cv2.THRESH_BINARY_INV)[1]

        _, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        characters_list = []
        coordinates_list = []
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1:    
                (x,y,z,w) = cv2.boundingRect(contours[i])
                coordinates_list.append((x,y,z,w))

        coordinates_list = sorted(coordinates_list, key=sort_on_tuples)

        for coor in coordinates_list:
            x,y,z,w = coor
            roi = original_image[y:y+w, x:x+z]
            if apt_image_shape(roi):
                characters_list.append(roi)

        characters_list_for_all_plates.append(characters_list)    
        
    return characters_list_for_all_plates

# Processing the number plate images
def image_processing_on_digits(char_list_for_all_plates):
    
    th_list_for_all_plates = []
    
    for char_list in char_list_for_all_plates:   
        th_list = []
        for char in char_list:
            _, th = cv2.threshold(char, 65, 255, cv2.THRESH_BINARY_INV)
            th_list.append(th)
        th_list_for_all_plates.append(th_list)
    
    return th_list_for_all_plates

# Reading the text written on the number plates
def read_number_plate(original_car_image):
    
    _, plates = detect_plates(original_car_image)
#     plt.imshow(plate)
    list_of_list_of_characters_on_number_plates = extract_characters(plates)
#     plt.imshow(list_of_characters_on_number_plate[1])
    list_of_processed_characters_list = image_processing_on_digits(list_of_list_of_characters_on_number_plates)
#     plt.imshow(processed_characters_list[1])
    model_numplate = load_model('character_classification_new.h5')
    
    global CLASS_LABELS
    
    def predict(img, model, restriction='none'):
        img = img/255
        img = cv2.resize(img, (20,20))
        img = img.reshape(1,20,20,3)
        
        probabilities = model.predict_proba(img)[0]
        
        if restriction == 'sec1':
            max_idx = 10 + np.argmax(probabilities[10:])
        
        elif restriction == 'sec2':
            max_idx = np.argmax(probabilities[:10])
        
        elif restriction == 'sec3':
            max_idx = 10 + np.argmax(probabilities[10:])
            
        elif restriction == 'sec4':
            max_idx = np.argmax(probabilities[:10])
            
        else:
            max_idx = np.argmax(probabilities)
        
        return CLASS_LABELS[max_idx]
    
    list_of_number_plates = []
    
    for processed_characters_list in list_of_processed_characters_list:
    
        number_plate = []
        length = len(processed_characters_list)

        for idx in range(length):

            char = processed_characters_list[idx]

            if length == 10:

                if idx in [0,1]:
                    prediction = predict(char, model_numplate, restriction='sec1')
                    number_plate.append(prediction)

                elif idx in [2,3]:
                    prediction = predict(char, model_numplate, restriction='sec2')
                    number_plate.append(prediction)

                elif idx in [4,5]:
                    prediction = predict(char, model_numplate, restriction='sec3')
                    number_plate.append(prediction)

                elif idx in [6,7,8,9]:
                    prediction = predict(char, model_numplate, restriction='sec4')
                    number_plate.append(prediction)

            elif length == 9:

                if idx in [0,1]:
                    prediction = predict(char, model_numplate, restriction='sec1')
                    number_plate.append(prediction)

                elif idx in [5,6,7,8]:
                    prediction = predict(char, model_numplate, restriction='sec4')
                    number_plate.append(prediction)

                else:
                    prediction = predict(char, model_numplate)
                    number_plate.append(prediction)

            else:
                number_plate = 'Nothing Found'
        
        number_plate = ''.join(number_plate)
        list_of_number_plates.append(number_plate)

    return list_of_number_plates

if __name__ == '__main__':

    path = 'NumberPlateImages/fiesta.jpg'



    try:
	    carImage = read_image(path)
	    
    except:
        print('File Not Found')

    

    try:
    	list_of_texts_read = read_number_plate(carImage)

    except:
    	print('Number Plate Reading Error')



    number_plate = ''
    if len(list_of_texts_read) != 0:
        for text_read in list_of_texts_read:
            if text_read != 'Nothing Found' or text_read != '':
                number_plate = text_read
                break
            else:
                number_plate = text_read
    else:
        number_plate = "Couldn't detect any number plate"



    print(number_plate)
    cv2.imshow('CAR IMAGE',carImage)
    cv2.waitKey(0)