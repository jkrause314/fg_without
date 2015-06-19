function ft_prep_data_ims(config);
% Prepare image data to be converted to caffe's format

fprintf('Prepare images for fine-tuning\n');


domain = config.domain;
OUT_DIR = config.ft_patch_folder;
TRAIN_DIR = fullfile(OUT_DIR, 'images', 'train');
if ~exist(TRAIN_DIR, 'dir')
  mkdir(TRAIN_DIR);
end

done_fname = fullfile(OUT_DIR, 'ims.done');
if exist(done_fname, 'file')
  fprintf('Already extracted images!\n')
  return;
end

pix_padding = config.cnn_padding;
resize_dim = config.cnn_input_size;


% Load image metadata and regions
% Load training images and regions
fprintf('loading...');
train_data = load(config.train_imagedata_fname);
train_images = train_data.images;
train_ims = train_data.ims;
train_part_data = load(config.tight_part_fname);
train_regions = train_part_data.regions;
num_train = numel(train_images);
num_regions = size(train_regions, 1);

train_save_ind = 1;



train_label_bbox_f = fopen(fullfile(OUT_DIR, 'train_labels_bbox.txt'), 'w');
train_label_joint_f = fopen(fullfile(OUT_DIR, 'train_labels_joint.txt'), 'w');

train_labels = [train_images.class];
num_classes = numel(unique(train_labels));

% Class mapping
domain_classes = unique([train_images.class]);
caffe_classes = 0:numel(domain_classes)-1;
domain_to_caffe = containers.Map(domain_classes, caffe_classes);
caffe_to_domain = containers.Map(caffe_classes, domain_classes);
map_f1 = fopen(fullfile(OUT_DIR, sprintf('%s_to_caffe.txt', domain)), 'w');
map_f2 = fopen(fullfile(OUT_DIR, sprintf('caffe_to_%s.txt', domain)), 'w');
for i = 1:numel(domain_classes)
  fprintf(map_f1, '%d %d\n', domain_classes(i), domain_to_caffe(domain_classes(i)));
  fprintf(map_f2, '%d %d\n', caffe_classes(i), caffe_to_domain(caffe_classes(i)));
end
fclose(map_f1);
fclose(map_f2);


for region_ind = 1:num_regions
  fprintf('Region %d/%d\n', region_ind, num_regions);
  tic;

  % Pick the appropriate box
  if region_ind == 1
    train_boxes = zeros(4, num_train);
    for i = 1:num_train
      bbox = train_images(i).bbox;
      train_boxes(:, i) = [bbox.x1 bbox.y1 bbox.x2 bbox.y2];
    end
  else
    train_boxes = squeeze(train_regions(region_ind-1,:,:)); % (x1 y1 x2 y2) x num_ims
  end

  % Write out everything
  % Get each image, with padding, resized
  % Write out the images and labels
  % Write out mappings from domain classes to caffe and vice versa

  % First pass to write stuff and record it
  write_inds = train_save_ind:train_save_ind+num_train-1;
  train_save_ind = train_save_ind + num_train;

  parfor i = 1:num_train
    if mod(i, 256) == 1
      fprintf('write im: %d\n', i);
    end
    im = train_ims{i};
    bbox = train_boxes(:, i);
    if all(bbox == 0)
      continue;
    end
    x1 = max(1, bbox(1) - pix_padding);
    x2 = min(size(im, 2), bbox(3) + pix_padding);
    y1 = max(1, bbox(2) - pix_padding);
    y2 = min(size(im, 1), bbox(4) + pix_padding);
    if size(im, 3) == 1
      im = repmat(im, [1,1,3]);
    end
    save_fname = fullfile(TRAIN_DIR, sprintf('%06d.png', write_inds(i)));
    im = im(y1:y2,x1:x2,:);
    im = imresize(im, [resize_dim, resize_dim]);
    imwrite(im, save_fname);
  end

  fprintf('Write to label files\n');
  for i = 1:num_train
    bbox = train_boxes(:, i);
    if all(bbox == 0)
      continue;
    end
    save_fname = fullfile(TRAIN_DIR, sprintf('%06d.png', write_inds(i)));
    if region_ind == 1
      fprintf(train_label_bbox_f, sprintf('%s %d\n', save_fname, domain_to_caffe(train_images(i).class)));
    else
      fprintf(train_label_joint_f, sprintf('%s %d\n', save_fname, domain_to_caffe(train_images(i).class)));
    end
  end
  toc;
end
fclose(train_label_bbox_f);
fclose(train_label_joint_f);
fclose(fopen(done_fname, 'w'));
