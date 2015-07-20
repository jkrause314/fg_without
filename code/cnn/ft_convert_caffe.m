function ft_convert_caffe(config)
% Convert the saved image regions into lmdb

done_fname = fullfile(config.ft_patch_folder, 'lmdb.done');
fprintf('Convert to lmdb\n');
if exist(done_fname, 'file')
  fprintf('Already made lmdb\n');
  return;
end

[pardir, ~, ~] = fileparts(mfilename('fullpath'));
script_loc = fullfile(pardir, 'create_caffe_lmdb.sh');
cmd = sprintf('"%s" "%s" "%s"', script_loc, config.caffe_root, config.ft_patch_folder);

fprintf('%s\n', cmd);
system(cmd);
assert(logical(exist(fullfile(config.ft_patch_folder, 'train_bbox_lmdb'), 'dir')));
fclose(fopen(done_fname, 'w'));
