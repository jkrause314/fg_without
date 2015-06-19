function my_run_rcnn(config)

old_dir = pwd();
rcnn_dir = config.rcnn_root;

% Run R-CNN code in that directory.
% Our code is already in the path and should be fine.
cd(rcnn_dir);
startup;

% Make the imdbs
imdb_from_fg_domain(config, 'train');
imdb_from_fg_domain(config, 'test');

% Run selective search
make_fg_ssearch(config);

%% Extract features. You might want to parallelize this across gpus if possible.
%rcnn_fg_cache_features(config, 'train_1');
%rcnn_fg_cache_features(config, 'train_2');
%rcnn_fg_cache_features(config, 'test_1');
%rcnn_fg_cache_features(config, 'test_2');

% Train the r-cnn
rcnn_fg_train_and_test(config);
% Train bounding box regression

cd(old_dir);
