function res = rcnn_fg_bbox_reg_train_and_test(config)
% Runs an experiment that trains a bounding box regressor and
% tests it.


done_fname = fullfile(config.rcnn_result_folder, 'bboxreg_rcnn.done');
if exist(done_fname, 'file')
  fprintf('Base R-CNN already trained!\n');
  return;
end

layer = 6; % 5=pool5,6=fc6, 7=fc7
gpu_num = config.gpu_num;

cache_name   = sprintf('%s_cache', config.domain);

imdb_train = imdb_from_fg_domain(config, 'train');
imdb_test  = imdb_from_fg_domain(config, 'test');

caffe('set_device', gpu_num);

conf = rcnn_config('sub_dir', imdb_train.name);

ld = load([conf.cache_dir 'rcnn_model']);

% train the bbox regression model
fprintf('train bbox regressor\n');
bbox_reg = rcnn_train_bbox_regressor(imdb_train, ld.rcnn_model, ...
    'min_overlap', 0.6, ...
    'layer', 5, ...
    'lambda', 1000, ...
    'robust', 0, ...
    'binarize', false);

% test the bbox regression model
res = my_rcnn_test_bbox_regressor(imdb_test, ld.rcnn_model, bbox_reg, 'bbox_reg');

[pardir, ~, ~] = fileparts(done_fname);
if ~exist(pardir, 'dir')
  mkdir(pardir);
end
fclose(fopen(done_fname, 'w'));
