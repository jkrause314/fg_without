function extract_part_feats(config, split, net)

fprintf('Extract part features, %s, %s\n', split, net);

% Check if already done
done_fname = fullfile(config.cnn_feat_dir, sprintf('%s.%s.done', split, net));
if exist(done_fname, 'file')
  fprintf('Already done!\n');
  return;
end

if ~exist(config.cnn_feat_dir, 'dir')
  mkdir(config.cnn_feat_dir);
end

% Feature extraction options
options = [];
options.layer = 'fc6';
options.net = net;
options.use_whole = true;
options.use_flips = true;
options.use_gpu = true;
options.temp_model_def_loc = tempname();
options.padding = config.cnn_padding;
options.use_corners = false;
options.use_center = false;
options.gpu_num = config.gpu_num;

fprintf('split: %s\n', split);

% Load up the data
if strcmp(split, 'train')
  im_data = load(config.train_imagedata_fname);
else
  im_data = load(config.test_imagedata_fname);
end
images = im_data.images;
ims = im_data.ims;
num_parts = config.final_part_pool_size;

bboxes_to_extract = cell(1, numel(images));

% Figure out which bounding boxes to extract from the images.
if strcmp(split, 'train')
  fprintf('Calculate extract boxes\n');
  train_reg_data = load(config.tight_part_fname);
  train_regions = train_reg_data.regions;
  for i = 1:numel(images)
    bbox = images(i).bbox;
    x1 = max(1, bbox.x1 - options.padding);
    x2 = min(size(ims{i}, 2), bbox.x2 + options.padding);
    y1 = max(1, bbox.y1 - options.padding);
    y2 = min(size(ims{i}, 1), bbox.y2 + options.padding);
    to_extract = zeros(1+num_parts, 4);
    to_extract(1,:) = [x1 y1 x2 y2];
    for k = 1:num_parts
      x1 = max(1, train_regions(k,1,i) - options.padding);
      y1 = max(1, train_regions(k,2,i) - options.padding);
      x2 = min(size(ims{i}, 2), train_regions(k,3,i) + options.padding);
      y2 = min(size(ims{i}, 1), train_regions(k,4,i) + options.padding);
      assert(x2 > x1 && y2 > y1 && x1 > 0 && y1 > 0 && x1 <= size(ims{i},2) && y1 <= size(ims{i}, 1));
      to_extract(k+1,:) = [x1 y1 x2 y2];
    end
    bboxes_to_extract{i} = round(to_extract);
    obj_dec_vals{i} = 1;
  end
elseif strcmp(split, 'test')
  fprintf('Use det bboxes...');
  det_data = load(config.rcnn_result_fname);
  det_bboxes = det_data.bboxes;

  for i = 1:numel(images)
    num_dets = size(det_bboxes{1}{i}, 1);
    assert(num_dets == 1);
    to_extract = zeros(1+num_parts, 4);
    % If no detection, set everything to whole image
    if isempty(det_bboxes{1}{i})
      fprintf('Image %d has no detection\n', i);
      to_extract(:,1) = 1;
      to_extract(:,2) = 1;
      to_extract(:,3) = images(i).width;
      to_extract(:,4) = images(i).height;
      obj_dec_vals{i} = -100;
      bboxes_to_extract{i} = round(to_extract);
      continue;
    end
    obj_det = det_bboxes{1}{i}(1,1:4);
    x1 = max(1, obj_det(1) - options.padding);
    y1 = max(1, obj_det(2) - options.padding);
    x2 = min(size(ims{i}, 2), obj_det(3) + options.padding);
    y2 = min(size(ims{i}, 1), obj_det(4) + options.padding);
    to_extract(1,:,1) = [x1 y1 x2 y2];
    for k = 1:num_parts
      part_det = det_bboxes{k+1}{i}(1,1:4);
      % No part detected.
      if all(part_det == 0)
        continue;
      end
      x1 = max(1, part_det(1) - options.padding);
      y1 = max(1, part_det(2) - options.padding);
      x2 = min(size(ims{i}, 2), part_det(3) + options.padding);
      y2 = min(size(ims{i}, 1), part_det(4) + options.padding);
      to_extract(k+1,:,1) = [x1 y1 x2 y2];
    end
    obj_dec_vals{i} = det_bboxes{1}{i}(:, 5);
    bboxes_to_extract{i} = round(to_extract);
  end
end
fprintf('done. Let''s extract some features now.\n');

for part_num = 1:num_parts+1
  fprintf('Region %d/%d\n', part_num, num_parts+1);
  clear functions; % Necessary to reset caffe

  % Check if already done.
  save_fname = fullfile(config.cnn_feat_dir, sprintf('feats.%s.part.%d.net.%s.mat', split, part_num, options.net));
  if exist(save_fname, 'file')
    fprintf('Already done!\n');
    continue;
  end
  working_fname = [save_fname '.working'];
  if exist(working_fname, 'file')
    fprintf('Already being worked on...\n');
    continue;
  end
  fclose(fopen(working_fname, 'w'));

  % Get paths to caffe parameters and such
  if part_num == 1
    region = 'bbox';
  else
    region = 'joint';
  end
  switch options.net
  case 'vgg'
    options.mean_fname = fullfile(config.ft_patch_folder, 'vgg_mean.mat');
    options.model_def_file = fullfile(config.caffe_root, 'models', '3785162f95cd2d5fee77', 'VGG_ILSVRC_19_layers_deploy.prototxt');
    options.model_file = fullfile(config.caffe_root, 'models', '3785162f95cd2d5fee77', 'VGG_ILSVRC_19_layers.caffemodel');

  case 'vgg-ft'
    options.mean_fname = fullfile(config.ft_patch_folder, sprintf('train_%s_mean.mat', region));
    options.model_def_file = fullfile(config.cnn_ft_dir, sprintf('%s_deploy.prototxt', region));
    options.model_file = fullfile(config.cnn_ft_dir, sprintf('%s_ft_model_iter_120000.caffemodel', region));
  end

  all_feats = cell(1, numel(images));
  all_labels = cell(1, numel(images));

  s=tic;
  for i = 1:numel(images)
    if mod(i, 64) == 1
      fprintf('cnn feats %s part %d, %s %d/%d\n', config.domain, part_num, split, i, numel(images));
      toc(s);
      s=tic;
    end

    im = ims{i};
    all_labels{i} = double(images(i).class);

    to_extract = bboxes_to_extract{i};
    num_dets = size(to_extract, 3);
    assert(num_dets <= 1); % Might be 0 if no test-time detections
    det_feats = cell(1, num_dets);
    for det_ind = 1:num_dets
      if all(to_extract(part_num,:, det_ind) == 0)
        continue;
      end
      x1 = to_extract(part_num, 1, det_ind);
      y1 = to_extract(part_num, 2, det_ind);
      x2 = to_extract(part_num, 3, det_ind);
      y2 = to_extract(part_num, 4, det_ind);
      im_crop = im(y1:y2,x1:x2,:);
      im_feats = extract_caffe(im_crop, options);
      assert(~isempty(im_feats));
      feat_dim = size(im_feats, 1);
      rep_factor = max(1, size(im_feats, 2));
      det_feats{det_ind} = im_feats;
    end
    % Fill in empty within an image
    feat_dim = max(cellfun(@(x)size(x,1), det_feats));
    rep_factor = max(cellfun(@(x)size(x,2), det_feats));
    det_feats(cellfun(@isempty, det_feats)) = {zeros(feat_dim, rep_factor, 'single')};
    all_feats{i} = cat(2, det_feats{:});
  end
  % Fill in empty across images
  feat_dim = max(cellfun(@(x)size(x,1), all_feats));
  rep_factor = max(cellfun(@(x)size(x,2), all_feats));
  all_feats(cellfun(@isempty, all_feats)) = {zeros(feat_dim, rep_factor, 'single')};
  for i = 1:numel(images)
    all_labels{i} = double(all_labels{i}) * ones(1, size(all_feats{i}, 2));
  end
  [pardir,~,~] = fileparts(save_fname);
  if ~exist(pardir, 'dir')
    mkdir(pardir);
  end
  save(save_fname, 'all_feats', 'all_labels', 'obj_dec_vals', 'options', '-v7.3')

  if exist(options.temp_model_def_loc)
    delete(options.temp_model_def_loc);
  end

  delete(working_fname);
end
fclose(fopen(done_fname, 'w'));
