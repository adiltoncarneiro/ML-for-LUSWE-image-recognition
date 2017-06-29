'''''
=======
blob detection for surface wave ultrasound elastography
=======
given the data of lung, eye and tendon testing, several points for region of interest were selected to mearure the wave speed of tissue.
the coordinates of these points can be extracted or the configuration of these points can be recognized and used for machine learning.  
Determinant of Hessian (DoH)
----------------------------
This is the fastest approach. It detects blobs by finding maximas in the
matrix of the Determinant of Hessian of the image. The detection speed is
independent of the size of blobs as internally the implementation uses
box filters instead of convolutions. Bright on dark as well as dark on
bright blobs are detected. The downside is that small blobs (<3px) are not
detected accurately. See :py:meth:`skimage.feature.blob_doh` for usage.
'''''

import numpy as np
from math import sqrt
import math
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xlrd
from skimage import io 
import os
from multiprocessing import Pool
import re
import glob
from scipy import misc
# Hyper Parameters
PATCH_SIZE = 21
BATCH_SIZE = 64
LR = 0.001              # learning rate
'''
#load image without points
d = os.chdir('//ressrv08/ultrasound/ZHANG/SHARED/Zhang_Lung_R01/Data/Patient_data/Patient base test/Data_LSWE_P82')
d1 = '//ressrv08/ultrasound/ZHANG/SHARED/Zhang_Lung_R01/Data/Patient_data/Patient base test/Data_LSWE_P82'
pat = '20161205T'
#w = os.listdir(d1)
w = glob.glob('20161205T*.jpg')
print(w)


for i in range(len(w)):
    fname = str(w[i]) 
    #filename = os.path.join(d1, fname)
    f = io.imread(fname)
    with Image.open(fname) as img:
        img.show()
'''
'''
#load image with circles
d = os.chdir('//ressrv08/ultrasound/ZHANG/SHARED/Zhang_Lung_R01/Data/Patient_data/Patient base test/Data_LSWE_P82')
d1 = '//ressrv08/ultrasound/ZHANG/SHARED/Zhang_Lung_R01/Data/Patient_data/Patient base test/Data_LSWE_P82'
a = list(range(1,91))
for i in a:
    fname = str(i) + '.jpg'
    #filename = os.path.join(d1, fname)
    #f = io.imread(filename)
    f = io.imread(fname)
    #with Image.open(fname) as img:
        #img.show()
'''
# load images as training data
# step 1
pool = Pool()
d = os.chdir('//ressrv08/ultrasound/ZHANG/SHARED/Zhang_Lung_R01/Data/Patient_data/Patient base test/Data_LSWE_P82')
d1 = '//ressrv08/ultrasound/ZHANG/SHARED/Zhang_Lung_R01/Data/Patient_data/Patient base test/Data_LSWE_P82'
a = list(range(1,91))
filenames = []
for i in a:
    filenames.append( str(i) + '.jpg')
    '''fname = str(i) + '.jpg'
    f = io.imread(fname)
    with Image.open(fname) as img:
        img.show()'''
for i in range(1,len(filenames)):
    im_array = []
    im_array.append(Image.open(filenames[i]))
    with Image.open(filenames[i]) as img:
        width, height = img.size # get the shape of image
    #print(width,height) # 1200, 900
    image = misc.imread(filenames[i])

# step 2
'''
filename_queue = tf.train.string_input_producer(filenames)
#Output strings (e.g. filenames) to a queue for an input pipeline.

# step 3: read, decode and resize images
reader = tf.WholeFileReader()
#A Reader that outputs the entire contents of a file as a value.
filename, content = reader.read(filename_queue)
#Returns the next record (key, value pair) produced by a reader.
image = tf.image.decode_jpeg(content, channels=3)
#Decode a JPEG-encoded image to a uint8 tensor.
#The attr channels indicates the desired number of color channels for the decoded image.

image = tf.cast(image, tf.float32)
#Casts a tensor to a new type.
resized_image = tf.image.resize_images(image, [224, 224])
#Resize images to size using the specified method.
# step 4: Batching
#image_batch = tf.train.batch([resized_image], batch_size=8)
#Creates batches of tensors in tensors.
'''
# read label (coordinates of selected points for each image) from excel
data = xlrd.open_workbook('coords.xlsx')
listoflists = []

for j in range(0,90):
    a_list = []
    table = data.sheets()[j]
    nrows = table.nrows
    ncols = table.ncols
    for i in range(0, nrows):
        #print (table.row_values(i))
        a_list.append((table.row_values(i)))
    listoflists.append(a_list)
    #print(a_list)
    #listoflists.append(a_list[:])
#print(listoflists[89])

# select some patches from selected points of the image
#grass_locations = [(465, 425), (471,445), (477, 465), (483, 485), (489, 505), (495, 525), (501, 545), (507, 565)]  # [y,x]
grass_locations = a_list[0]
grass_patches = []
for loc in grass_locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

" read images and corresponding coordinates of selected points for the model"
test_x = im_array[:]
test_y = listoflists[:]

tf_x = tf.placeholder(tf.float32, [None, 1200*900])
image = tf.reshape(tf_x, [-1, 900, 1200, 1])              # (batch, height, width, channel)
tf_y = tf.placeholder(tf.float32, [None, 8])            # input y

# CNN
conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
    inputs=image,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)           # -> (28, 28, 16)
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)           # -> (14, 14, 16)
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)    # -> (14, 14, 32)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 32)
flat = tf.reshape(pool2, [-1, 7*7*32])          # -> (7*7*32, )
output = tf.layers.dense(flat, 10)              # output layer

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

# generate coordinates of selected points for test image
# x_train is images, y_train is coordinates of points, x_test is image, y_test is the predicted coordinates of points

for step in range(1200):    # training
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:      # testing
        accuracy_ = sess.run(accuracy, {tf_x: test_x, tf_y: test_y})
        print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

# print 10 predictions from test data
test_output = sess.run(output, {tf_x: test_x[:10]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], 1), 'real number')
