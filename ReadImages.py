from PIL import Image
import glob
import numpy
import numpy as np
from keras.utils import np_utils
import os

from pickle import dump, load
import random

# fix random seed for reproducibility
seed = 1320
np.random.seed(seed)

batch_size = 10
nb_classes = 10
nb_epoch = 1

#setup a standard image size;
STANDARD_SIZE = (640, 480)

def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    if verbose==True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = numpy.array(img)
    return img

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

y_train = []
image = []
print "load train data"
for filename in glob.glob("train/c0/*.jpg"):
    #print "0"
    img = img_to_matrix(filename)
    #print len(img),len(img[0])
    img = flatten_image(img)
    #print len(img)
    # pylab.subplot(1, 3, 1)
    # pylab.axis("off")
    # pylab.imshow(img)
    # pylab.gray()
    # pylab.show()
    #img = numpy.asarray(img, dtype="float64")
    #img = img.transpose(2, 0, 1).reshape(1, 3, 640, 480)
    # box = (0, 0, 640, 480)
    # im_crop = img.crop(box)
    # im_crop.show()
    image.append( img)
    y_train.append(0)

for filename in glob.glob("train/c1/*.jpg"):
   # print "1"
    img = img_to_matrix(filename)
    img = flatten_image(img)
    image.append( img)
    y_train.append(1)


for filename in glob.glob("train/c2/*.jpg"):
  #  print "2"
    img = img_to_matrix(filename)
    img = flatten_image(img)
    image.append( img)
    y_train.append(2)

for filename in glob.glob("train/c3/*.jpg"):
  # print "3"
    img = img_to_matrix(filename)
    img = flatten_image(img)
    image.append( img)
    y_train.append(3)

for filename in glob.glob("train/c4/*.jpg"):
 #   print "4"
    img = img_to_matrix(filename)
    img = flatten_image(img)
    image.append( img)
    y_train.append(4)

for filename in glob.glob("train/c5/*.jpg"):
#    print "5"
    img = img_to_matrix(filename)
    img = flatten_image(img)
    image.append( img)
    y_train.append(5)

for filename in glob.glob("train/c6/*.jpg"):
#    print "6"
    img = img_to_matrix(filename)
    img = flatten_image(img)
    image.append( img)
    y_train.append(6)

for filename in glob.glob("train/c7/*.jpg"):
#    print "7"
    img = img_to_matrix(filename)
    img = flatten_image(img)
    image.append( img)
    y_train.append(7)

for filename in glob.glob("train/c8/*.jpg"):
#    print "8"
    img = img_to_matrix(filename)
    img = flatten_image(img)
    image.append( img)
    y_train.append(8)

for filename in glob.glob("train/c9/*.jpg"):
#    print "9"
    img = img_to_matrix(filename)
    img = flatten_image(img)
    image.append( img)
    y_train.append(9)
print "Train data loaded"
X_test = []
listeImageTest = []


os.chdir("C:/Users/adm-loc/PycharmProjects/State Farm Distracted Driver Detection/test")
for filename in glob.glob("*.jpg"):
    #print "test"
    img = img_to_matrix(filename)
    img = flatten_image(img)
    X_test.append( img)
    listeImageTest.append(filename)

os.chdir("C:/Users/adm-loc/PycharmProjects/State Farm Distracted Driver Detection/")

output = open("Images.pkl", "wb")
dump(listeImageTest, output, -1)
output.close()

image = numpy.asarray(image, dtype="float32")
X_test = numpy.asarray(X_test, dtype="float32")

output = open("Test.pkl", "wb")
dump(X_test, output, -1)
output.close()
#print type(X_test)
#print filename,len(image),len(image[0])

X = image
#print type(X)

output = open("Data.pkl", "wb")
dump(X, output, -1)
output.close()



# print y_train
# random.shuffle(y_train)
# print y_train

# Do not use shuffle on Y !!
Y = np_utils.to_categorical(y_train, nb_classes)
#print Y,type(Y),"\n\n"

output = open("Target.pkl", "wb")
dump(Y, output, -1)
output.close()





