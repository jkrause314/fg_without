function make_cub_anno_files(config)
% Makes annotation files useful for the CUB 2011 dataset

fprintf('Making cub annotation files...\n');
CUB_DIR = config.cub_root;

[out_dir,~,~] = fileparts(config.train_image_fname);
if ~exist(out_dir, 'dir')
  mkdir(out_dir);
end

done_fname = fullfile(out_dir, 'cub.done');
if exist(done_fname, 'file')
  fprintf('Already done!\n');
  return;
end

% Keypoints, bounding boxes, paths, classes, split for every image
im_part_fname = fullfile(CUB_DIR, 'parts', 'part_locs.txt');
im_bbox_fname = fullfile(CUB_DIR, 'bounding_boxes.txt');
im_path_fname = fullfile(CUB_DIR, 'images.txt');
im_split_fname = fullfile(CUB_DIR, 'train_test_split.txt');
im_label_fname = fullfile(CUB_DIR, 'image_class_labels.txt');

path_data = textscan(fopen(im_path_fname, 'r'), '%d %s');
im_ids = path_data{1};
rel_paths = path_data{2};
images = repmat(struct('id', [], 'rel_path', [], 'train', [], 'test', [], ...
  'class', [], 'bbox', [], 'width', [], 'height', []), 1, numel(im_ids));
% Paths first
for i = 1:numel(im_ids)
  image_id = im_ids(i);
  rel_path = rel_paths{i};
  images(image_id).id = image_id;
  images(image_id).rel_path = rel_path;
end

% Image sizes
all_ims = cell(1, numel(im_ids));
for i = 1:numel(im_ids)
  if mod(i, 256) == 0
    fprintf('im size %d/%d\n', i, numel(im_ids));
  end
  im_path = fullfile(CUB_DIR, 'images', images(i).rel_path);
  im = imread(im_path);
  if size(im, 3) == 1
    im = repmat(im, [1,1,3]);
  end
  all_ims{i} = im;
  images(i).width = size(im, 2);
  images(i).height = size(im, 1);
end

% Splits
split_data = textscan(fopen(im_split_fname, 'r'), '%d %d');
im_ids = split_data{1};
is_training = split_data{2};
for i = 1:numel(im_ids)
  images(im_ids(i)).train = logical(is_training(i));
  images(im_ids(i)).test = ~logical(is_training(i));
end

% Classes
class_data = textscan(fopen(im_label_fname, 'r'), '%d %d');
im_ids = class_data{1};
labels = class_data{2};
for i = 1:numel(im_ids)
  images(im_ids(i)).class = labels(i);
end

% Bboxes
bbox_data = textscan(fopen(im_bbox_fname, 'r'), '%d %f %f %f %f');
im_ids = bbox_data{1};
xs = bbox_data{2};
ys = bbox_data{3};
widths = bbox_data{4};
heights = bbox_data{5};
for i = 1:numel(im_ids)
  x1 = int32(xs(i));
  y1 = int32(ys(i));
  x2 = int32(x1 + widths(i) - 1);
  y2 = int32(y1 + heights(i) - 1);
  images(im_ids(i)).bbox = struct('x1', x1, 'y1', y1, 'x2', x2, 'y2', y2);
end


im_save_fname = config.image_fname;
save(im_save_fname, 'images');

all_images = images;

% Make testing, training files
fprintf('Saving...');
images = all_images([all_images.test]);
save(config.test_image_fname, 'images');
images = all_images([all_images.train]);
save(config.train_image_fname, 'images');

% Save files including images. Both of these files are a couple gigs
images = all_images([all_images.test]);
ims = all_ims([all_images.test]);
save(config.train_imagedata_fname, 'images', 'ims', '-v7.3');

images = all_images([all_images.train]);
ims = all_ims([all_images.train]);
save(config.test_imagedata_fname, 'images', 'ims', '-v7.3');
fprintf('done.\n');

fclose(fopen(done_fname, 'w'));
