#!/bin/bash
# Create the caffe mean

CAFFE_ROOT="${1}"
DATA_DIR="${2}"

EXAMPLE=$CAFFE_ROOT/examples/imagenet
TOOLS=$CAFFE_ROOT/build/tools

echo "Creating train bbox mean..."
$TOOLS/compute_image_mean $DATA_DIR/train_bbox_lmdb \
  $DATA_DIR/train_bbox_mean.binaryproto

echo "Creating train joint mean..."
$TOOLS/compute_image_mean $DATA_DIR/train_joint_lmdb \
  $DATA_DIR/train_joint_mean.binaryproto
