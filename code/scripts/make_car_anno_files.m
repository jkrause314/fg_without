function make_car_anno_files(config)

fprintf('Making car annotation files...\n');

CAR_DIR = config.car_root;

[OUT_DIR,~,~] = fileparts(config.train_image_fname);
if ~exist(OUT_DIR, 'dir')
  mkdir(OUT_DIR);
end

done_fname = fullfile(OUT_DIR, 'car.done');
if exist(done_fname, 'file')
  fprintf('Already done!\n');
  return;
end

IMAGE_DIR = config.im_base_car;

if ~exist(OUT_DIR, 'dir')
  mkdir(OUT_DIR);
end


anno_data = load(fullfile(CAR_DIR, 'cars_annos.mat'));
% Remap classes

images = repmat(struct('rel_path', [], 'train', [], 'test', [], ...
  'class', [], 'bbox', [], 'width', [], 'height', []), 1, numel(anno_data.annotations));
all_ims = cell(1, numel(anno_data.annotations));

% Paths first
num_ims = numel(images);

for i = 1:numel(anno_data.annotations)
  if mod(i, 256) == 0
    fprintf('im size %d/%d\n', i, num_ims);
  end
  anno = anno_data.annotations(i);
  images(i).rel_path = anno.relative_im_path;
  c = anno.class;
  if c > 123 % Skip a class
    c = c - 1;
  end
  images(i).class = c;
  images(i).train = ~anno.test;
  images(i).test = anno.test;
  im_path = fullfile(IMAGE_DIR, images(i).rel_path);
  im = imread(im_path);
  if size(im, 3) == 1
    im = repmat(im, [1,1,3]);
  end
  all_ims{i} = im;
  images(i).width = size(im, 2);
  images(i).height = size(im, 1);
  x1 = anno.bbox_x1;
  x2 = anno.bbox_x2;
  y1 = anno.bbox_y1;
  y2 = anno.bbox_y2;
  assert(x1 > 0);
  assert(x2 > x1);
  assert(x2 <= images(i).width);
  assert(y1 > 0);
  assert(y2 > y1);
  assert(y2 <= images(i).height);
  images(i).bbox = struct('x1', x1, 'y1', y1, 'x2', x2, 'y2', y2);
end
all_images = images;

im_save_fname = config.image_fname;
save(im_save_fname, 'images');

% Make testing, training files
fprintf('Saving...');
images = all_images([all_images.test]);
save(config.test_image_fname, 'images');
images = all_images([all_images.train]);
save(config.train_image_fname, 'images');

% Save files including images. Both of these files are about 7 gigs
images = all_images([all_images.test]);
ims = all_ims([all_images.test]);
save(config.test_imagedata_fname, 'images', 'ims', '-v7.3');

images = all_images([all_images.train]);
ims = all_ims([all_images.train]);
save(config.train_imagedata_fname, 'images', 'ims', '-v7.3');
fprintf('done.\n');

fclose(fopen(done_fname, 'w'));
