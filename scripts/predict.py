from tensorflow.keras.preprocessing import image 
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pickle

def validate_path(img_path, filename):
    flag = os.path.exists(img_path + filename + '.jpg')
    if not flag:
        print('File ' + img_path + filename + '.jpg does not exist' )
        return False
    return True

def validate_answer(ans):
    message = 'Invalid answer, please enter Y or N'
    if not type(ans) == str:
        print(message)
        return 'x'
    if not len(ans) == 1:
        print(message)
        return 'x'
    if not ans.upper() == 'Y' and not ans.upper() == 'N':
        print(message)
        return 'x'
    return ans.upper()

def main():
    model = models.load_model("calc/vgg_fruit_veggies.h5")
    img_path = 'testing/test/'
    infile = open('labels.pkl', 'rb')
    labels = pickle.load(infile)
    labels = dict((v,k) for k,v in labels.items())
    #print(labels)
    infile.close()
    flag = True
    while flag:
        print('Fruit and vegetable classifier')
        ans = False
        path = ''
        while not ans:
            path = input('Filename of image without extension: ')
            ans = validate_path(img_path, path)
        img = image.load_img(img_path + path + '.jpg', target_size=(224,224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis = 0)
        #img_tensor = np.expand_dims(img_tensor, axis = 0)
        #img_tensor /= 255.

        #model response
        confidence = model.predict(img_tensor)
        confidence = np.argmax(confidence,axis=1)
        print(confidence)
        pred = [labels[k] for k in confidence]
        print('The predicted class is ' + pred[0])

        plt.imshow(img_tensor[0])
        plt.savefig('resources/pred.png')
        ans = 'x'
        while not ans == 'Y' and not ans == 'N':
            ans = input('Would you like another prediction: ')
            ans = validate_answer(ans)
        if ans == 'N':
            flag = False
        else:
            print('\n')

main()