function res = imdb_eval_fg_domain(config, cls, boxes, imdb, conf, suffix, nms_thresh)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Add a random string ("salt") to the end of the results file name
% to prevent concurrent evaluations from clobbering each other
use_res_salt = true;

% save results
if ~exist('suffix', 'var') || isempty(suffix) || strcmp(suffix, '')
  suffix = '';
else
  if suffix(1) ~= '_'
    suffix = ['_' suffix];
  end
end

if ~exist('nms_thresh', 'var') || isempty(nms_thresh)
  nms_thresh = 0.3;
end

image_ids = imdb.image_ids;

if use_res_salt
  prev_rng = rng;
  rng shuffle;
  salt = sprintf('%d', randi(100000));
  res_id = salt;
  rng(prev_rng);
else
  res_id = '';
end

% Apply NMS
boxes_keep = cell(size(boxes));
for i = 1:numel(image_ids)
  bbox = boxes{i};
  keep = nms(bbox, nms_thresh);
  boxes_keep{i} = bbox(keep,:);
end
% bbox(:,end) are decision values
% bbox(:,1:4) are the actual bboxes

% Load up GT
% Note that we only have GT for the whole object, not the parts
if strcmp(imdb.image_set, 'train')
  im_data = load(config.train_image_fname);
elseif strcmp(imdb.image_set, 'test')
  im_data = load(config.test_image_fname);
end
images = im_data.images;
gt_boxes = arrayfun(@(x)[x.bbox.x1 x.bbox.y1 x.bbox.x2 x.bbox.y2], images, 'uniformoutput', false);

recall = [];
prec = [];
ap = 0;
ap_auc = 0;

% Bug in VOCevaldet requires that tic has been called first
[recall, prec, ap] = myevaldet(boxes_keep, gt_boxes);
ap_auc = xVOCap(recall, prec);

% force plot limits
ylim([0 1]);
xlim([0 1]);

print(gcf, '-djpeg', '-r0', ...
    [conf.cache_dir cls '_pr_' imdb.name suffix '.jpg']);

fprintf('!!! %s : %.4f %.4f\n', cls, ap, ap_auc);

res.recall = recall;
res.prec = prec;
res.ap = ap;
res.ap_auc = ap_auc;

save([conf.cache_dir cls '_pr_' imdb.name suffix], ...
    'res', 'recall', 'prec', 'ap', 'ap_auc');
