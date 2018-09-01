import numpy as np
from PIL import Image
import os
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

from matplotlib import pyplot as plt


base_model = ResNet50(include_top = False, pooling = 'max')

predictions = Dense(4, activation='softmax')(base_model.output)

model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights('./weights.h5')

model.compile(optimizer=optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=["acc"])

########################
#write the image name in .txt
########################

'''
dataset_dir = 'fish/test_data/1dpf'
file_list=[]
write_file = open('fish/test_data/unmber1.txt', 'w')
print(write_file)
for file in os.listdir(dataset_dir):
    if file.endswith(".jpg"):
        write_name = file
        file_list.append(write_name)
        sorted(file_list)
        number_of_lines = len(file_list)
for current_line in range(number_of_lines):
    write_file.write(file_list[current_line] + '\n')

write_file.close()

dataset_dir = 'fish/test_data/2dpf'
file_list=[]
write_file = open('fish/test_data/unmber2.txt', 'w')
print(write_file)
for file in os.listdir(dataset_dir):
    if file.endswith(".jpg"):
        write_name = file
        file_list.append(write_name)
        sorted(file_list)
        number_of_lines = len(file_list)
for current_line in range(number_of_lines):
    write_file.write(file_list[current_line] + '\n')

write_file.close()

dataset_dir = 'fish/test_data/3dpf'
file_list=[]
write_file = open('fish/test_data/unmber3.txt', 'w')
print(write_file)
for file in os.listdir(dataset_dir):
    if file.endswith(".jpg"):
        write_name = file
        file_list.append(write_name)
        sorted(file_list)
        number_of_lines = len(file_list)
for current_line in range(number_of_lines):
    write_file.write(file_list[current_line] + '\n')

write_file.close()

dataset_dir = 'fish/test_data/4dpf'
file_list=[]
write_file = open('fish/test_data/unmber4.txt', 'w')
print(write_file)
for file in os.listdir(dataset_dir):
    if file.endswith(".jpg"):
        write_name = file
        file_list.append(write_name)
        sorted(file_list)
        number_of_lines = len(file_list)
for current_line in range(number_of_lines):
    write_file.write(file_list[current_line] + '\n')

write_file.close()


'''
#######################
#count images in the files
#######################
from os.path import join as PJ
import os

path, dirs, files = next(os.walk("./fish/test_data/1dpf"))
file_count1 = len(files)

path, dirs, files = next(os.walk("./fish/test_data/2dpf"))
file_count2 = len(files)

path, dirs, files = next(os.walk("./fish/test_data/3dpf"))
file_count3 = len(files)

path, dirs, files = next(os.walk("./fish/test_data/4dpf"))
file_count4 = len(files)

file_count = [file_count1 ,file_count2 ,file_count3 ,file_count4]

arr_dpf = [file_count1 ,file_count2 ,file_count3 ,file_count4]
arr_file = ['1dpf' ,'2dpf' ,'3dpf' ,'4dpf']

###########################
#put the image name in array
###########################

a = os.path.abspath('.')
arr1 = np.genfromtxt('fish/test_data/unmber1.txt',dtype= str)
arr2 = np.genfromtxt('fish/test_data/unmber2.txt',dtype= str)
arr3 = np.genfromtxt('fish/test_data/unmber3.txt',dtype= str)
arr4 = np.genfromtxt('fish/test_data/unmber4.txt',dtype= str)
arr_txt = [arr1,arr2,arr3,arr4]


#################################
#test the image  and make the bar summery graph
#################################

toImage = Image.new('RGBA',(1280,960))

for j in range(4):
    print("{:d}dpf".format(j+1) )
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    arr = arr_txt[j]


    for i in range(file_count[j]):

        b = PJ("fish/test_data", arr_file[j], arr[i])

        img = image.load_img(b , target_size = (800,800))

        x = image.img_to_array(img)

        x = np.expand_dims(x , axis = 0)

        preds = model.predict(x)

        a = np.argmax(preds)

        if a == 0:
             c1+=1
        elif a == 1:
            c2+=1
        elif a == 2:
            c3 +=1
        elif a== 3:
            c4+=1

    print(c1, c2 ,c3, c4)
    n = 4
    X = np.arange(n)+1
    Y1 = [c1,c2,c3,c4]
    fig, ax = plt.subplots()
    plt.bar(X, Y1 , width = 0.35, facecolor = 'lightskyblue', edgecolor = 'white', label='one', lw=1)
    plt.title("{:d}dpf".format(j+1))
    plt.xticks(X, ('1dpf', '2dpf', '3dpf', '4dpf'))
    for a,b in zip(X,Y1):
        plt.text(a-0.2 ,b+0.5 ,"{:.3f}%".format((b/file_count[j])*100))
    plt.savefig('plot.png')
    fromImge = Image.open('plot.png')
    if j < 2:
        loc = ((int(j) * 640), 0)
    else:
        loc = ((int(j-2) * 640), 480)
    print(loc)
    toImage.paste(fromImge, loc)

toImage.save('merged2.png')
