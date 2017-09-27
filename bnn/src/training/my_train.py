from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234) # for reproducibility?

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip
import glob
from scipy import misc

import binary_net
import cnv

from pylearn2.datasets.zca_dataset import ZCA_Dataset   
from pylearn2.datasets.cifar10 import CIFAR10 
from pylearn2.utils import serial

from collections import OrderedDict

def loaddata():
    '''
    Loads the NIST SD19 Character dataset, which must can be downloaded from https://www.nist.gov/srd/nist-special-database-19
    Assumes dataset is downloaded in the current directory (..../bnn/src/training) and ordered by class.
    '''

    #classes = ["30", "31", "32", "33", "34", "35", "36", "37", "38", "39", #Digits
#"41", "42", "43", "44", "45", "46", "47", "48", "49", "4a", "4b", "4c", "4d", "4e", "4f", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "5a", #Upper case
#"61", "62", "63", "64", "65", "66", "67", "68", "69", "6a", "6b", "6c", "6d", "6e", "6f", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "7a"] #Lower case
    path = "/home/cp612sh/dataset/train_set"
    classes = os.listdir(path)
    #classes = ["daisy", "dandelion","roses", "sunflowers", "tulips"]

    NumImagesPerClassTrain = 1500
    NumImagesPerClassValidation = 400
    NumImagesPerClassTest = 100

    # NumImagesPerClassTrain = 300
    # NumImagesPerClassTest = 100
    # NumImagesPerClassValidation = 50

    pngTrain = []
    pngTest = []
    pngValidation = []
    labelsTrain = []
    labelsTest = []
    labelsValidation = []

    for glyph in classes:
        i = 0
        print("Loading Glyph code: "+glyph)
        for image_path in glob.glob( path + "/" + glyph+"/*.jpg"):
        #for image_path in glob.glob("./by_class/"+glyph+"/train_"+glyph+"/*.png"):
            if (i < NumImagesPerClassTrain):
                pic_train = misc.imread(image_path)
                picture_train = misc.imresize(pic_train, (32,32))
                #misc.imshow(picture_train)
                pngTrain.append(picture_train) 
                labelsTrain.append(classes.index(glyph))
                i=i+1
            elif(i < (NumImagesPerClassTrain + NumImagesPerClassValidation)):
                pic_valid = misc.imread(image_path)
                picture_valid = misc.imresize(pic_valid, (32,32))
                pngValidation.append(picture_valid) 
                labelsValidation.append(classes.index(glyph))
                i=i+1
            elif(i < (NumImagesPerClassTrain + NumImagesPerClassValidation + NumImagesPerClassTest)):
                pic_test = misc.imread(image_path)
                picture_test = misc.imresize(pic_test, (32,32))
                pngTest.append(picture_test) 
                labelsTest.append(classes.index(glyph))
                i=i+1
            else:
                break
        # k = 0
        # for image_path in glob.glob("/home/cp612sh/wsy/BNN-PYNQ-master/dataset/test/"+glyph+"/*.jpg"):
        # #for image_path in glob.glob("./by_class/"+glyph+"/hsf_4/*.png"):
        #     if (k < NumImagesPerClassTest):
        #         pic_test = misc.imread(image_path)
        #         picture_test = misc.imresize(pic_test, (32,32))
        #         pngTest.append(picture_test) 
        #         labelsTest.append(classes.index(glyph))
        #         k=k+1
        #     else:
        #         break

    pngTrain = np.reshape(np.subtract(np.multiply(2./255.,pngTrain),1.),(-1,3,32,32))
    pngValidation = np.reshape(np.subtract(np.multiply(2./255.,pngValidation),1.),(-1,3,32,32))
    pngTest = np.reshape(np.subtract(np.multiply(2./255.,pngTest),1.),(-1,3,32,32))

    
    return (pngTrain, labelsTrain, pngTest, labelsTest, pngValidation, labelsValidation)

    


if __name__ == "__main__":
    
    learning_parameters = OrderedDict()
    # BN parameters
    batch_size = 50
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    learning_parameters.alpha = .1
    print("alpha = "+str(learning_parameters.alpha))
    learning_parameters.epsilon = 1e-4
    print("epsilon = "+str(learning_parameters.epsilon))
    
    # W_LR_scale = 1.    
    learning_parameters.W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(learning_parameters.W_LR_scale))
    
    # Training parameters
    num_epochs = 500
    print("num_epochs = "+str(num_epochs))
    
    # Decaying LR 
    LR_start = 0.001
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    save_path = "clothes_parameters.npz"
    print("save_path = "+str(save_path))
    
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    print('Loading clothes dataset...')
    train_setX, train_sety, test_setX, test_sety, valid_setX, valid_sety = loaddata()
        
    # bc01 format
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    train_sety = np.hstack(train_sety)
    valid_sety = np.hstack(valid_sety)
    test_sety = np.hstack(test_sety)
    
    # Onehot the targets
    train_sety = np.float32(np.eye(30)[train_sety])    
    valid_sety = np.float32(np.eye(30)[valid_sety])
    test_sety = np.float32(np.eye(30)[test_sety])
    
    # for hinge loss
    train_sety = 2* train_sety - 1.
    valid_sety = 2* valid_sety - 1.
    test_sety = 2* test_sety - 1.

    print('Building the CNN...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = cnv.genCnv(input, 30, learning_parameters)

    train_output = lasagne.layers.get_output(cnn, deterministic=False)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    # W updates
    W = lasagne.layers.get_all_params(cnn, binary=True)
    W_grads = binary_net.compute_grads(loss,cnn)
    updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
    updates = binary_net.clipping_scaling(updates,cnn)
    
    # other parameters updates
    params = lasagne.layers.get_all_params(cnn, trainable=True, binary=False)
    updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
    binary_net.train(
            train_fn,val_fn,
            cnn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_setX,train_sety,
            valid_setX,valid_sety,
            test_setX,test_sety,
            save_path=save_path,
            shuffle_parts=shuffle_parts)
