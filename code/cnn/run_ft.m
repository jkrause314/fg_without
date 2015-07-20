function run_ft(config)

fprintf('Run fine-tuning\n');

regions = {'bbox', 'joint'};
for i = 1:numel(regions)
  region = regions{i};
  final_model_fname = fullfile(config.cnn_ft_dir, sprintf('%s_ft_model_iter_120000.caffemodel', region));
  if exist(final_model_fname, 'file')
    fprintf('%s fine-tuning already done!\n', region);
    continue;
  end
  script_loc = fullfile('./', config.cnn_ft_dir, sprintf('%s_script.sh', region));
  system(script_loc);
end

