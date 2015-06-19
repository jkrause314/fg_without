function roidb = roidb_from_fg_domain(config, imdb)
% roidb = roidb_from_voc(imdb)
%   Builds an regions of interest database from imdb image
%   database. Uses precomputed selective search boxes available
%   in the R-CNN data package.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

cache_file = ['./imdb/cache/roidb_' imdb.name '.mat'];
fprintf('roidb %s %s\n', config.domain, imdb.image_set);
try
  load(cache_file);
catch
  roidb.name = imdb.name;

  fprintf('Loading region proposals...');
  regions_file = fullfile('data', 'selective_search_data', sprintf('%s_%s.mat', config.domain, imdb.image_set));
  regions = load(regions_file);
  fprintf('done\n');

  % Load image data
  image_set = imdb.image_set;
  im_data = load(imdb.im_data_fname);
  images = im_data.images;
  use_parts = strcmp(imdb.image_set, 'train');
  if use_parts
    part_data = load(config.tight_part_fname);
    train_regions = part_data.regions;
    num_parts = size(train_regions, 1);
  end

  for i = 1:length(imdb.image_ids)
    tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
    % Need objects.class and objects.bbox
    bbox = images(i).bbox;
    our_rec = [];
    our_rec.objects.class = imdb.classes{1};
    our_rec.objects.bbox = double([bbox.x1 bbox.y1 bbox.x2 bbox.y2]);

    % Add in parts
    if use_parts
      our_rec.objects = repmat(our_rec.objects, [1, 1+num_parts]);
      part_good = false(1, num_parts);
      for j = 1:num_parts
        part_x1 = train_regions(j,1,i);
        part_y1 = train_regions(j,2,i);
        part_x2 = train_regions(j,3,i);
        part_y2 = train_regions(j,4,i);
        part_good(j) = all(train_regions(j,:,i) ~= 0);
        if ~part_good(j)
          continue;
        end
        assert(part_x2 > part_x1);
        assert(part_y2 > part_y1);
        our_rec.objects(j+1).class = imdb.classes{j+1};
        our_rec.objects(j+1).bbox = double([part_x1 part_y1 part_x2 part_y2]);
      end
      our_rec.objects(~part_good) = [];
    end

    roidb.rois(i) = attach_proposals(our_rec, regions.boxes{i}, imdb.class_to_id);
  end

  fprintf('Saving roidb to cache...');
  save(cache_file, 'roidb', '-v7.3');
  fprintf('done\n');
end


% ------------------------------------------------------------------------
function rec = attach_proposals(voc_rec, boxes, class_to_id)
% ------------------------------------------------------------------------

% change selective search order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
boxes = boxes(:, [2 1 4 3]);

%           gt: [2108x1 double]
%      overlap: [2108x20 single]
%      dataset: 'voc_2007_trainval'
%        boxes: [2108x4 single]
%         feat: [2108x9216 single]
%        class: [2108x1 uint8]
if isfield(voc_rec, 'objects')
  gt_boxes = cat(1, voc_rec.objects(:).bbox);
  all_boxes = cat(1, gt_boxes, boxes);
  gt_classes = class_to_id.values({voc_rec.objects(:).class});
  gt_classes = cat(1, gt_classes{:});
  num_gt_boxes = size(gt_boxes, 1);
else
  gt_boxes = [];
  all_boxes = boxes;
  gt_classes = [];
  num_gt_boxes = 0;
end
num_boxes = size(boxes, 1);

rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
rec.overlap = zeros(num_gt_boxes+num_boxes, class_to_id.Count, 'single');
for i = 1:num_gt_boxes
  rec.overlap(:, gt_classes(i)) = ...
      max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end
rec.boxes = single(all_boxes);
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));
