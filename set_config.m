function config = set_config(domain)

config = [];

% Location of your CUB_200_2011 directory
config.cub_root = '/scail/scratch/u/jkrause/temp/cub/CUB_200_2011';

% Segmentation directory for CUB. Only necessary if evaluating segmentation.
config.cub_seg_dir = '/scail/scratch/u/jkrause/temp/cub/segmentations';

% Location of the directory containing cars_annos.mat
config.car_root = '/scail/scratch/u/jkrause/temp/car/';


% Location of caffe install.
config.caffe_root = '/home/jkrause/cvpr15_caffe';

% Location of rcnn install
config.rcnn_root = '/home/jkrause/cvpr15_rcnn';

% Which gpu to use (0-indexed)
config.gpu_num = 3;





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Don't touch anything below this line
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
config.cnn_padding = 16;
config.ilsvrc_mean_loc = fullfile(config.caffe_root, 'matlab', 'caffe', 'ilsvrc_2012_mean.mat');
config.caffenet_deploy_loc = fullfile(config.caffe_root, 'models', 'bvlc_reference_caffenet', 'deploy.prototxt');
config.caffenet_model_loc = fullfile(config.caffe_root, 'models', 'bvlc_reference_caffenet', 'bvlc_reference_caffenet.caffemodel');
config.cnn_input_size = 256;


% Pose graph/matching parameters
config.num_msts = 5;
config.pose_graph_layer = 'conv4';
config.large_part_pool_size = 500;
config.final_part_pool_size = 31;
config.part_context = .25;

config.root = pwd();
config.im_base_cub = fullfile(config.cub_root, 'images');
config.im_base_car = config.car_root;

% Add some paths, set up the matlab pool
addpath(genpath(fullfile(config.root, 'code')));
addpath(genpath(fullfile(config.caffe_root, 'matlab', 'caffe')));

if matlabpool('size') == 0
  fprintf('Opening matlab pool\n');
  matlabpool('open');
end

config.domain = domain;
config.im_base = config.(sprintf('im_base_%s', domain));
config.image_fname = fullfile(config.root, 'processed', 'data', sprintf('%s_images.mat', domain));
config.train_image_fname = fullfile(config.root, 'processed', 'data', sprintf('%s_images_train.mat', domain));
config.test_image_fname = fullfile(config.root, 'processed', 'data', sprintf('%s_images_test.mat', domain));
config.train_imagedata_fname = fullfile(config.root, 'processed', 'data', sprintf('%s_imagedata_train.mat', domain));
config.test_imagedata_fname = fullfile(config.root, 'processed', 'data', sprintf('%s_imagedata_test.mat', domain));
config.coseg_out_dir = fullfile(config.root, 'processed', 'coseg', domain);
config.coseg_im_save_dir = fullfile(config.coseg_out_dir, 'ims');
config.coseg_save_fname = fullfile(config.coseg_out_dir, 'segs.mat');
config.cnn_bbox_fname = fullfile(config.root, 'processed', 'cnn', 'features', sprintf('%s_%s_train.mat', config.pose_graph_layer, domain));
config.mst_save_fname = fullfile(config.root, 'processed', 'posegraph', sprintf('%s_msts.mat', domain));
config.matching_im_out_dir = fullfile(config.root, 'processed', 'posegraph', domain, 'im_out');
config.alignment_fname = fullfile(config.root, 'processed', 'posegraph', domain, 'alignments.mat');
config.propagate_save_fname = fullfile(config.root, 'processed', 'posegraph', sprintf('keypoints_%s.mat', domain));
config.tight_part_fname = fullfile(config.root, 'processed', 'posegraph', sprintf('regions_%s.mat', domain));
config.ft_patch_folder = fullfile(config.root, 'processed', 'cnn', 'data', domain);
