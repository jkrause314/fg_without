function vgg_finetune(config)
% Do all the fine-tuning with a VGGNet

fprintf('VGG finetune %s\n', config.domain);
done_fname = fullfile(config.cnn_ft_dir, 'finetuning.done');
if exist(done_fname, 'file')
  fprintf('Already done\n');
  return;
end

% Get region crops for parts and bbox
ft_prep_data_ims(config);

% Convert to caffe format
ft_convert_caffe(config);

% Get the mean
ft_create_mean(config);

% Prepare prototxt files
ft_create_prototxt(config);

% Do the fine-tuning
run_ft(config);

fclose(fopen(done_fname, 'w'));
