function merge_part_detections(config);

fprintf('Merge part detections\n');
done_fname = fullfile(config.rcnn_result_folder, 'merged.done');
if exist(done_fname, 'file')
  fprintf('Already merged!\n');
  return;
end

bboxes = merge_rcnn_det_parts_bboxes(config);
obj_bboxes = bboxes{1};

% Do NMS for the purpose of evaluation
nms_thresh = 0.3;
boxes_keep = cell(size(obj_bboxes));
for i = 1:numel(obj_bboxes)
  bbox = obj_bboxes{i};
  keep = nms(bbox, nms_thresh);
  boxes_keep{i} = bbox(keep,:);
end 

% Get AP
im_data = load(config.test_image_fname);
images = im_data.images;
gt_boxes = arrayfun(@(x)double([x.bbox.x1 x.bbox.y1 x.bbox.x2 x.bbox.y2]), images, 'uniformoutput', false);
[recall, prec, ap] = myevaldet(boxes_keep, gt_boxes);
ap_auc = xVOCap(recall, prec);

ap_all = ap;
ap_auc_all = ap_auc;

% Evaluate AP with top 1 per image
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

% Only save top one
for im_ind = 1:numel(bboxes{1})
  obj_boxes = bboxes{1}{im_ind};
  [~, ind] = max(obj_boxes(:, 5));

  for reg_ind = 1:numel(bboxes)
    boxes = bboxes{reg_ind}{im_ind};
    if isempty(boxes)
      continue;
    end
    bboxes{reg_ind}{im_ind} = boxes(ind, :);
  end
end
save(config.rcnn_result_fname, 'bboxes', 'ap_all', 'ap_auc_all', 'ap_single', 'ap_auc_single', '-v7.3');
fclose(fopen(done_fname, 'w'));
