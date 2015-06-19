function [res_test, res_train] = rcnn_fg_train_and_test(config)
% Runs an experiment that trains an R-CNN model and tests it.

layer = 6; % 5=pool5,6=fc6, 7=fc7
gpu_num = config.gpu_num;

% -------------------- CONFIG --------------------
cache_name   = sprintf('%s_cache', config.domain);
net_file     = './data/caffe_nets/finetune_voc_2007_trainval_iter_70k';

crop_mode    = 'warp';
crop_padding = 16;
k_folds      = 0;

% ------------------------------------------------

imdb_train = imdb_from_fg_domain(config, 'train');
imdb_test  = imdb_from_fg_domain(config, 'test');

caffe('set_device', gpu_num);

[rcnn_model, rcnn_k_fold_model] = ...
    my_rcnn_train(imdb_train, ...
      'layer',           layer, ...
      'k_folds',         k_folds, ...
      'cache_name',      cache_name, ...
      'net_file',        net_file, ...
      'crop_mode',       crop_mode, ...
      'crop_padding',    crop_padding);

if k_folds > 0
  res_train = rcnn_test(rcnn_k_fold_model, imdb_train, '', cachesub_suffix);
else
  res_train = [];
end

%res_test = rcnn_test(rcnn_model, imdb_train, '');
res_test = rcnn_test(rcnn_model, imdb_test, '');
