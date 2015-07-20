function ft_create_mean(config)
% Create the mean files

done_fname = fullfile(config.ft_patch_folder, 'mean.done');
fprintf('Make mean files\n');
if exist(done_fname, 'file')
  fprintf('Already made mean\n');
  return;
end

[pardir, ~, ~] = fileparts(mfilename('fullpath'));
script_loc = fullfile(pardir, 'create_caffe_mean.sh');
cmd = sprintf('"%s" "%s" "%s"', script_loc, config.caffe_root, config.ft_patch_folder);

fprintf('%s\n', cmd);
system(cmd);

assert(logical(exist(fullfile(config.ft_patch_folder, 'train_bbox_mean.binaryproto'), 'file')));

% vanilla VGG Mean

mean_pix = [103.939, 116.779, 123.68];
image_mean = zeros(256,256,3, 'single');
image_mean(:,:,1) = mean_pix(1);
image_mean(:,:,2) = mean_pix(2);
image_mean(:,:,3) = mean_pix(3);
save(fullfile(config.ft_patch_folder, 'vgg_mean.mat'), 'image_mean');


orig_mean_fname = fullfile(config.ft_patch_folder, 'train_joint_mean.binaryproto');
new_mean_fname = fullfile(config.ft_patch_folder, 'train_joint_mean.mat');
image_mean = caffe('read_mean', orig_mean_fname);
save(new_mean_fname, 'image_mean');

orig_mean_fname = fullfile(config.ft_patch_folder, 'train_bbox_mean.binaryproto');
new_mean_fname = fullfile(config.ft_patch_folder, 'train_bbox_mean.mat');
image_mean = caffe('read_mean', orig_mean_fname);
save(new_mean_fname, 'image_mean');

fclose(fopen(done_fname, 'w'));
