function run_coseg(config)

% Co-segmentation parameters
params = [];
params.im_meta_fname = config.train_image_fname;
params.im_base = config.im_base;
params.out_dir = config.coseg_out_dir;
params.im_out_dir = config.coseg_im_save_dir;
params.coseg_save_fname = config.coseg_save_fname;

if exist(params.coseg_save_fname, 'file')
  fprintf('Already did coseg.\n');
  return;
end

params.coseg_iters = 5;
params.resize_area = 100000;
params.bbox_context = 1;
params.fg_prior = .5;

params.bbox_min_fg_area = .10;
params.bbox_max_fg_area = .90;
params.bbox_min_fg_length = .5;
params.bbox_min_fg_height = .5;

params.use_class_fg = true;
params.use_class_bg = false;
params.class_weight = .5;
params.do_refine = true;

params

coseg_main(params);
