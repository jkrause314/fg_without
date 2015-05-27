function check_cub_coseg(config)

fprintf('Evaluating cub segmentation\n');
gt_fname = fullfile(config.root, 'processed', 'data', 'cub_segmentations_train.mat');
im_fname = fullfile(config.root, 'processed', 'data', 'cub_images_train.mat');

im_data = load(im_fname);
images = im_data.images;
classes = [images.class];
num_classes = numel(unique(classes));

gt_data = load(gt_fname);
gt_segs = gt_data.segmentations;

coseg_fname = fullfile(config.root, 'processed', 'coseg', 'cub', 'segs.mat');

coseg_data = load(coseg_fname);
cosegs = coseg_data.segmentations;

fprintf('Separating class annotations\n');
all_class_gt = cell(1, num_classes);
all_class_cosegs = cell(1, num_classes);
for i = 1:num_classes
  im_inds = find([images.class] == i);
  all_class_gt{i} = gt_segs(im_inds);
  all_class_cosegs{i} = cosegs(im_inds);
end

class_accs = zeros(1, num_classes);
class_jacs = zeros(1, num_classes);
parfor i = 1:num_classes
  fprintf('%d/%d\n', i, num_classes);
  accs = zeros(size(im_inds));
  jacs = zeros(size(im_inds));
  class_gt = all_class_gt{i};
  class_coseg = all_class_cosegs{i};
  for j = 1:numel(class_gt)
    gt_seg = class_gt{j};
    coseg = class_coseg{j};
    if any(size(gt_seg) ~= size(coseg))
      %gt_seg = imresize(gt_seg, size(coseg));
      coseg = imresize(coseg, size(gt_seg));
    end
    gt_vec = gt_seg(:);
    coseg_vec = coseg(:);
    accs(j) = mean(gt_vec == coseg_vec);
    jacs(j) = sum(gt_vec & coseg_vec) ./ sum(gt_vec | coseg_vec);
  end
  class_accs(i) = 100*mean(accs);
  class_jacs(i) = 100*mean(jacs);
end
fprintf('Acc: %g\n', mean(class_accs));
fprintf('Jac: %g\n', mean(class_jacs));
