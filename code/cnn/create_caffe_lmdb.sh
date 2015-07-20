#!/bin/bash
# Create the imagenet lmdb inputs

CAFFE_ROOT="${1}"
DATA_DIR="${2}"

EXAMPLE=$CAFFE_ROOT/examples/imagenet
TOOLS=$CAFFE_ROOT/build/tools
DATA_ROOT=/

echo "Creating train joint lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=0 \
    --resize_width=0 \
    --shuffle \
    $DATA_ROOT \
    $DATA_DIR/train_labels_joint.txt \
    $DATA_DIR/train_joint_lmdb


echo "Creating train bbox lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=0 \
    --resize_width=0 \
    --shuffle \
    $DATA_ROOT \
    $DATA_DIR/train_labels_bbox.txt \
    $DATA_DIR/train_bbox_lmdb

echo "Done."
