import os
import sys
from finnthesizer import *

if __name__ == "__main__":
    bnnRoot = "."
    npzFile = bnnRoot + "/clothes_parameters.npz"
    #npzFile = bnnRoot + "/cifar10_parameters.npz"
    targetDirBin = bnnRoot + "/binparam-cnv-pynq"
    
    peCounts = [16, 32, 16, 16, 4, 1, 1, 1, 4]
    simdCounts = [3, 32, 32, 32, 32, 32, 4, 8, 1]
    
    path = "/home/cp612sh/dataset/train_set"
    classes = os.listdir(path)
    #classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    if not os.path.exists(targetDirBin):
      os.mkdir(targetDirBin)
      
    rHW = BNNWeightReader(npzFile, True)
    
    # TODO:
    # - generalize and move into library
    # - spit out config header
    # - add param generation for SVHN
    
    # process convolutional layers
    for convl in range(0, 6):
      peCount = peCounts[convl]
      simdCount = simdCounts[convl]
      print "Using peCount = %d simdCount = %d for engine %d" % (peCount, simdCount, convl)
      if convl == 0:
        # use fixed point weights for the first layer
        (w,t) = rHW.readConvBNComplex(usePopCount=False)
        # compute the padded width and height
        paddedH = padTo(w.shape[0], peCount)
        paddedW = padTo(w.shape[1], simdCount)
        # compute memory needed for weights and thresholds
        neededWMem = (paddedW * paddedH) / (simdCount * peCount)
        neededTMem = paddedH / peCount
        print "Layer %d: %d x %d" % (convl, paddedH, paddedW)
        print "WMem = %d TMem = %d" % (neededWMem, neededTMem)
        m = BNNProcElemMem(peCount, simdCount, neededWMem, neededTMem, numThresBits=24, numThresIntBits=16)
        m.addMatrix(w,t)
        m.createBinFiles(targetDirBin, str(convl))
      else:
        # regular binarized layer
        (w,t) = rHW.readConvBNComplex()
        # compute the padded width and height
        paddedH = padTo(w.shape[0], peCount)
        paddedW = padTo(w.shape[1], simdCount)
        # compute memory needed for weights and thresholds
        neededWMem = (paddedW * paddedH) / (simdCount * peCount)
        neededTMem = paddedH / peCount
        print "Layer %d: %d x %d" % (convl, paddedH, paddedW)
        print "WMem = %d TMem = %d" % (neededWMem, neededTMem)
        m = BNNProcElemMem(peCount, simdCount, neededWMem, neededTMem)
        m.addMatrix(w,t)
        m.createBinFiles(targetDirBin, str(convl))
    
    # process fully-connected layers
    for fcl in range(6,9):
      peCount = peCounts[fcl]
      simdCount = simdCounts[fcl]
      print "Using peCount = %d simdCount = %d for engine %d" % (peCount, simdCount, fcl)
      (w,t) =  rHW.readFCBNComplex()
      # compute the padded width and height
      paddedH = padTo(w.shape[0], peCount)
      paddedW = padTo(w.shape[1], simdCount)
      # compute memory needed for weights and thresholds
      neededWMem = (paddedW * paddedH) / (simdCount * peCount)
      neededTMem = paddedH / peCount
      print "Layer %d: %d x %d" % (fcl, paddedH, paddedW)
      print "WMem = %d TMem = %d" % (neededWMem, neededTMem)
      m = BNNProcElemMem(peCount, simdCount, neededWMem, neededTMem)
      m.addMatrix(w,t)
      m.createBinFiles(targetDirBin, str(fcl))
    
    with open(targetDirBin + "/classes.txt", "w") as f:
        f.write("\n".join(classes))
