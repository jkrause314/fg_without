============================================================
Installation Instructions
============================================================
First you'll need to install some pre-requisite packages and compile certain
files:

---------------
Caffe:
---------------
We need caffe for training CNNs and extracting features. The main caffe
webpage is:

http://caffe.berkeleyvision.org/

We'll be using caffe rc2, which you can download at
https://github.com/BVLC/caffe/archive/rc2.zip

Our code uses a slightly-modified version of matcaffe, which is included
under code/cnn/matcaffe.cpp. Please copy that file into the 'matlab/caffe'
folder within your caffe directory before compiling matcaffe.

Note: caffe has a bunch of its own pre-requisites and can be a pain to install
if you're missing them or don't have a gpu-enabled machine. Good luck.

---------------
OpenCV development files
---------------
This is a prerequisite for the co-segmentation code. Consult your own
OS's documentation for details. Look for a package like 'opencv-devel' or
'libopencv-dev'.

---------------
Matlab
---------------
This is all in matlab (except caffe stuff). Sorry folks!
Tested with matlab r2012b.

---------------
R-CNN
---------------
Follow the installation instructions at https://github.com/rbgirshick/rcnn

Note that R-CNN is pretty slow, easily the slowest part of this code. If you're
feeling ambitious and really want to make this code as good as possible, try
replacing R-CNN with the new Fast R-CNN at
https://github.com/rbgirshick/fast-rcnn.

Since we're using a slightly different version of Caffe than R-CNN expects, some
files need to be modified. Replace 'rcnn_load_model.m' and
'rcnn_cache_pool5_features.m' with the versions provided in code/rcnn.

---------------
sed
---------------
This code requires sed to be available via the command line. If your OS doesn't
have sed (most linux distributions have it), you'll need to either install it or
replace that small bit of code.

---------------
MatlabBGL
---------------
Download it from:
http://www.mathworks.com/matlabcentral/fileexchange/10922-matlabbgl
and add it to the path or put it in the 'thirdparty' folder, which will
automatically be added to the path.

To work with Matlab and other libraries used, delete the file:
matlab_bgl/test/assert.m

---------------
assignmentoptimal
---------------
Download code for the matching problem from:
http://www.mathworks.com/matlabcentral/fileexchange/6543-functions-for-the-rectangular-assignment-problem

Compile 'assignmentoptimal.c' with mex and make sure it's accessible on the path.

---------------
liblinear-dense-float
---------------
This is a more efficient version of liblinear which uses floats instead of
doubles, saving a factor of two in memory.

Download from:
https://github.com/BVLC/DPD/tree/master/thirdparty/kdes_2.0/liblinear-1.5-dense-float
and follow the installation instructions.

---------------
Compile co-segmentation code
---------------
Run code/coseg/mex/compile_mycoseg.sh

That should, hopefully, "just work". :)

---------------
Download pre-trained CNNs.
---------------
We need the CaffeNet, VGGNet, and fine-tuned PASCAL models. You should be able
to get these with the following commands:

CaffeNet:
(from the caffe root directory)
scripts/download_model_binary.py models/bvlc_reference_caffenet

VGGNet:
(from the caffe root directory)
wget -P models/3785162f95cd2d5fee77/ http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel

fine-tuned PASCAL:
(from the R-CNN root directory)
data/fetch_models.sh


============================================================
How to run
============================================================
Set the variables in the top half of set_config.m

Run either run_cub.m for CUB-200-2011 or run_car.m for cars-196.

============================================================
Other Notes
============================================================
-If you don't run this on a machine with a GPU, it's going to be very slow.

-Some particularly slow parts of the code (fine-tuning multiple networks,
extracting features, etc.) can be sped up relatively easily if you have multiple
GPUs.

-R-CNNs need a lot of memory to train. Expect it to take up several tens of GB
of RAM while running.

-The co-segmentation code uses code from mexopencv by Kota Yamaguchi:
https://github.com/kyamagu/mexopencv

-The Shape Context code has been modified from the original version provided
by Serge Belongie at 
https://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sc_digits.html

============================================================
Citation
============================================================
If you use this code please cite the corresponding CVPR paper:

Fine-Grained Recognition without Part Annotations
 Jonathan Krause, Hailin Jin, Jianchao Yang, Li Fei-Fei.
 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2015.

============================================================
Contact
============================================================
Have questions? Find a bug? Something not working? E-mail me at:

jkrause@cs.stanford.edu

