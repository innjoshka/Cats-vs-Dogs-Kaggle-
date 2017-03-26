import numpy as np
import cv2
import os
import tflearn
from random import shuffle
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


cwd = os.getcwd()
TRAIN_DIR = cwd + '/train'
TEST_DIR = cwd + '/test'

IMAGE_SIZE = 44


#creating labels for cats&dogs
def label_img(img):
	word_label = img.split('.')[0]

	if word_label == 'cat':
		return [1,0]

	elif word_label == 'dog':
		return [0,1]



#dealing with directories
list_of_train_images = os.listdir(TRAIN_DIR)
list_of_train_images.remove('.DS_Store') #COMMENT IF THE SYSTEM IS NOT MAC OS , removing hidden file of metadata from list_of_train_images.
list_of_test_images = os.listdir(TEST_DIR)
list_of_test_images.remove('.DS_Store') #COMMENT IF THE SYSTEM IS NOT MAC OS , removing hidden file of metadata from list_of_test_images.



#creating training data from images
def creating_train_data():
	train_data = []
	for img in list_of_train_images:
		label = label_img(img)
		img_path = os.path.join(TRAIN_DIR,img)
		img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE), interpolation = cv2.INTER_AREA) #interpolation for shrinking img
		train_data.append([np.array(img),np.array(label)])
	shuffle(train_data)
	np.save('train_data.npy', train_data)
	return train_data

#train_data = creating_train_data()
#if dataset has already created - comment it out
train_data = np.load('train_data.npy')


#creating testing data from images
def creating_test_data():
	test_data = []
	for img in list_of_test_images:
		img_path = os.path.join(TEST_DIR,img)
		img_numb = img.split('.')[0]
		img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
		test_data.append([np.array(img), img_numb])
	shuffle(test_data)
	np.save('test_data.npy', test_data)
	return test_data

#test_data = creating_test_data()
#if dataset has already created - comment it out
test_data = np.load('test_data.npy')

train = train_data
test = test_data

#creating data for feeding into Convolutional 2d NN
train_X = [i[0] for i in train]
train_X = np.array(train_X).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)
train_y = [i[1] for i in train]


#creating Convolutional Neural Network
CNN = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input') #first element should be None (representing batch size)

# Input For Convolutional 2d NN  = 4-D Tensor [batch, height, width, in_channels].
# Output For Convolutional 2d NN  = 4-D Tensor [batch, new height, new width, nb_filter].

#         incoming, #nb_filter, #Size of filters
CNN = conv_2d(CNN, 32, 2, activation='relu') # [., ., ., .] , 32 - The number of convolutional filters[neurons],  2x2 -  Size of filters.
#            incoming, #kernel_size
CNN = max_pool_2d(CNN, 2)

CNN = conv_2d(CNN, 64, 2, activation='relu')
CNN = max_pool_2d(CNN, 2)

CNN = conv_2d(CNN, 128, 2, activation='relu')
CNN = max_pool_2d(CNN, 2)

CNN = conv_2d(CNN, 64, 2, activation='relu')
CNN = max_pool_2d(CNN, 2)

CNN = conv_2d(CNN, 32, 2, activation='relu')
CNN = max_pool_2d(CNN, 2)

#                 incoming, n_units
CNN = fully_connected(CNN, 1024, activation='relu')
CNN = dropout(CNN, 0.85)

CNN = fully_connected(CNN, 2, activation='softmax')
CNN = regression(CNN, optimizer='Adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(CNN)

model.fit({'input': train_X}, {'targets': train_y}, n_epoch=3, batch_size=96, snapshot_step=1000, show_metric=True)

#model.save('CNN.model')

model.load('CNN.model')


with open('submission_file.csv','w') as f:
	f.write('id,label\n')

with open('submission_file.csv','a') as f:
	for data in test_data:
		img_numb = data[1]
		img_data = data[0]
		data = img_data.reshape(IMAGE_SIZE,IMAGE_SIZE,1)
		model_submit = model.predict([data])[0]
		f.write('{},{}\n'.format(img_numb,model_submit[1]))
