


-download/install caffe
  testing with rc2
  get caffenet
-download/install r-cnn
  need to replace some files?
-need open cv dev package?

tested with matlab r2012b

http://www.mathworks.com/matlabcentral/fileexchange/10922-matlabbgl
  put it in thirdparty
  make a script that does this automatically?
requires sed :(
  make my own replacement!

export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libstdc++.so.6

assignmentoptimal
  where did I get this from
  nedes to be compiled

sc code adapted form serge belongie

compile coseg code

remove /scail/scratch/u/jkrause/cvpr15_release/code/thirdparty/matlab_bgl/test/assert.m
get ./data/caffe_nets/finetune_voc_2007_trainval_iter_70k by running ./data/fetch_models.sh

replace rcnn_load_model with our rcnn_load_model
replace rcnn_cache_pool5_features.m
