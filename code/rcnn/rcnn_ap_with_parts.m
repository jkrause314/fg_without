clear all;

%cache_dir = '/mnt/ilcompf1d0/user/krause/project/code/cnn/rcnn/cachedir/cub_kpdisc_keypoints_regions_cub_test_fc6';
%save_fname = 'part_preds_reg_mult_cub.mat';
%im_data = load('/mnt/ilcompf1d0/user/krause/project/processed/data/cub_images_test.mat');

cache_dir = '/mnt/ilcompf1d0/user/krause/project/code/cnn/rcnn/cachedir/car_kpdisc_keypoints_regions_car_test_fc6';
save_fname = 'part_preds_reg_mult_car.mat';
im_data = load('/mnt/ilcompf1d0/user/krause/project/processed/data/car_images_test.mat');

%cache_dir = '/mnt/ilcompf1d0/user/krause/project/code/cnn/rcnn/cachedir/dog_kpdisc_keypoints_regions_dog_test_fc6';
%save_fname = 'part_preds_reg_mult_dog.mat';
%im_data = load('/mnt/ilcompf1d0/user/krause/project/processed/data/dog_images_test.mat');

num_keep = 1;

bbox_reg = true;
bbox_ext_pix = 10;


images = im_data.images;
gt_boxes = arrayfun(@(x)double([x.bbox.x1 x.bbox.y1 x.bbox.x2 x.bbox.y2]), images, 'uniformoutput', false);

bboxes = merge_rcnn_det_parts_bboxes(cache_dir, bbox_reg, bbox_ext_pix);
%bboxes = merge_rcnn_det_parts_bboxes(cache_dir, bbox_reg, bbox_ext_pix, gt_boxes);

%%gmm_data = load('cub_gmms.mat');
%%gmm = gmm_data.gmms(4);
%%addpath('../util');
%%bboxes = merge_rcnn_det_parts_bboxes_gmm(cache_dir, bbox_reg, bbox_ext_pix, gmm);


obj_bboxes = bboxes{1};

% Do NMS
nms_thresh = 0.3;
addpath('rcnn');
addpath('rcnn/nms');
addpath('rcnn/utils');
boxes_keep = cell(size(obj_bboxes));
for i = 1:numel(obj_bboxes)
  bbox = obj_bboxes{i};
  keep = nms(bbox, nms_thresh);
  boxes_keep{i} = bbox(keep,:);
end 

% Get AP

recall = [];
prec = [];
ap = 0;
ap_auc = 0;

[recall, prec, ap] = myevaldet(boxes_keep, gt_boxes);
ap_auc = xVOCap(recall, prec);

ap_all = ap;
ap_auc_all = ap_auc;

% Take top 1 per image
for i = 1:numel(boxes_keep)
  [~, ind] = max(boxes_keep{i}(:, 5));
  boxes_keep{i} = boxes_keep{i}(ind,:);
end
[recall, prec, ap] = myevaldet(boxes_keep, gt_boxes);
ap_auc = xVOCap(recall, prec);

ap_single = ap;
ap_auc_single = ap_auc;

fprintf('\n');
fprintf('AP all boxes: %g\n', ap_all);
fprintf('AP AUC all boxes: %g\n', ap_auc_all);
fprintf('\n');
fprintf('AP top box: %g\n', ap_single);
fprintf('AP AUC top box: %g\n', ap_auc_single);

fprintf('Saving top preds\n');

do_nms = true;
nms_thresh = 0.8; % Arbitrary
for im_ind = 1:numel(bboxes{1})
  % NMS
  if do_nms
    bbox = bboxes{1}{im_ind};
    nms_mask = nms(bbox, nms_thresh);
  end
  obj_boxes = bboxes{1}{im_ind};
  obj_boxes = obj_boxes(nms_mask,:);
  [~, inds] = sort(obj_boxes(:, 5), 'descend');
  inds = inds(1:min(num_keep, numel(inds)));

  for reg_ind = 1:numel(bboxes)
    boxes = bboxes{reg_ind}{im_ind};
    if isempty(boxes)
      continue;
    end
    boxes = boxes(nms_mask,:);
    num_rows = min(numel(inds), size(boxes, 1));
    boxes = boxes(inds(1:num_rows),:);
    bboxes{reg_ind}{im_ind} = boxes;
  end
end
save(save_fname, 'images', 'gt_boxes', 'bboxes', 'ap_all', 'ap_auc_all', 'ap_single', 'ap_auc_single', '-v7.3');
