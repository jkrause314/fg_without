function extract_feats_bbox_domain(config)

% Used for the pose graph

% For train, test
% Crop each image to bbox + padding
% Extract the specified features. Flips/Crops?
% Make extra labels if we need to
% Save to a file


options.pix_padding = config.cnn_padding;
options.layer = config.pose_graph_layer;

options.use_whole = true;
options.use_center = false;
options.use_corners = false;
options.use_flips = false;

options.use_gpu = true;
options.gpu_num = config.gpu_num;
options.temp_model_def_loc = tempname();

options.net = 'caffenet';
options.mean_fname = config.ilsvrc_mean_loc;
options.model_def_file = config.caffenet_deploy_loc;
options.model_file = config.caffenet_model_loc;

split = 'train';


save_fname = config.cnn_bbox_fname;

if exist(save_fname, 'file')
  fprintf('Already done!\n');
  return;
end

% Load
fprintf('Load images...');
im_data = load(config.train_imagedata_fname);
images = im_data.images;
ims = im_data.ims;
clear('im_data');
fprintf('done.\n');

labels = [images.class];
all_feats = cell(1, numel(images));
all_labels = cell(1, numel(images));
s=tic;
for i = 1:numel(images)
  if mod(i, 256) == 1
    fprintf('cnn feats %s %d/%d\n', split, i, numel(images));
    toc(s);
    s=tic;
  end

  % Crop
  im = ims{i};
  bbox = images(i).bbox;
  x1 = max(1, bbox.x1 - options.pix_padding);
  x2 = min(size(im, 2), bbox.x2 + options.pix_padding);
  y1 = max(1, bbox.y1 - options.pix_padding);
  y2 = min(size(im, 1), bbox.y2 + options.pix_padding);
  im = im(y1:y2,x1:x2,:);

  % Extract features
  features = extract_caffe(im, options);
  if isempty(features)
    fprintf('bad layer\n');
    continue;
  end
  all_feats{i} = single(features);
  all_labels{i} = repmat(images(i).class, 1, size(features, 2));
end
[pardir,~,~] = fileparts(save_fname);
if ~exist(pardir, 'dir')
  mkdir(pardir);
end
save(save_fname, 'all_feats', 'all_labels', 'options', '-v7.3')

if exist(options.temp_model_def_loc)
  delete(options.temp_model_def_loc);
end
