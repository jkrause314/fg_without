function ft_create_prototxt(config)
% Create the prototxt files.
% Super hacky, requires sed.

fprintf('Create prototxt files\n');

% Get number of classes
image_data = load(config.train_image_fname);
images = image_data.images;
num_classes = max([images.class]);

if ~exist(config.cnn_ft_dir, 'dir')
  mkdir(config.cnn_ft_dir);
end

[curdir, ~, ~] = fileparts(mfilename('fullpath'));


regions = {'bbox', 'joint'};
for i = 1:numel(regions)
  region = regions{i};
  % Make deploy
  old_deploy_loc = fullfile(curdir, 'finetune_deploy_vgg_template.prototxt');
  new_deploy_loc = fullfile(config.cnn_ft_dir, sprintf('%s_deploy.prototxt', region));
  copyfile(old_deploy_loc, new_deploy_loc);
  system(sprintf('sed -i ''s#NUM_CLASSES#%d#g'' %s', num_classes, new_deploy_loc));

  % Make train_test
  old_tt_loc = fullfile(curdir, 'finetune_train_test_vgg_template.prototxt');
  new_tt_loc = fullfile(config.cnn_ft_dir, sprintf('%s_train_test.prototxt', region));
  train_lmdb_fpath = fullfile(config.ft_patch_folder, sprintf('train_%s_lmdb', region));
  train_mean_fpath = fullfile(config.ft_patch_folder, sprintf('train_%s_mean.binaryproto', region));
  copyfile(old_tt_loc, new_tt_loc);
  system(sprintf('sed -i ''s#NUM_CLASSES#%d#g'' %s', num_classes, new_tt_loc));
  system(sprintf('sed -i ''s#TRAIN_LMDB#%s#g'' %s', train_lmdb_fpath, new_tt_loc));
  system(sprintf('sed -i ''s#TRAIN_MEAN#%s#g'' %s', train_mean_fpath, new_tt_loc));

  % Make solver
  old_solver_loc = fullfile(curdir, 'finetune_solver_vgg_template.prototxt');
  new_solver_loc = fullfile(config.cnn_ft_dir, sprintf('%s_solver.prototxt', region));
  snapshot_pref = fullfile(config.cnn_ft_dir, sprintf('%s_ft_model', region));
  copyfile(old_solver_loc, new_solver_loc);
  system(sprintf('sed -i ''s#TRAIN_TEST_LOC#%s#g'' %s', new_tt_loc, new_solver_loc));
  system(sprintf('sed -i ''s#SNAPSHOT_PREFIX#%s#g'' %s', snapshot_pref, new_solver_loc));

  % Make bash script
  old_script_loc = fullfile(curdir, 'finetune_vgg_template.sh');
  new_script_loc = fullfile(config.cnn_ft_dir, sprintf('%s_script.sh', region));
  copyfile(old_script_loc, new_script_loc);
  system(sprintf('sed -i ''s#GPU#%d#g'' %s', config.gpu_num, new_script_loc));
  system(sprintf('sed -i ''s#CAFFE_LOC#%s#g'' %s', config.caffe_root, new_script_loc));
  system(sprintf('sed -i ''s#SOLVER_LOC#%s#g'' %s', new_solver_loc, new_script_loc));
  system(sprintf('chmod u+x %s', new_script_loc));
end

done_fname = fullfile(config.ft_patch_folder, 'prototxt.done');
fprintf('Make mean files\n');
if exist(done_fname, 'file')
  fprintf('Already made mean\n');
  return;
end

fclose(fopen(done_fname, 'w'));
