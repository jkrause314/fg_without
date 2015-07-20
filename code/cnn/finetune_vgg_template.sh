#!/bin/bash

gpu=GPU

CAFFE_ROOT=CAFFE_LOC

solver=SOLVER_LOC
prev_model=CAFFE_LOC/models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers.caffemodel

$CAFFE_ROOT/build/tools/caffe train -solver $solver -weights $prev_model -gpu $gpu
