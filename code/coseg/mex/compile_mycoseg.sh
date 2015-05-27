#!/bin/bash

mex -cxx -largeArrayDims myCoseg.cpp MxArray.cpp  -lopencv_core -lopencv_imgproc
