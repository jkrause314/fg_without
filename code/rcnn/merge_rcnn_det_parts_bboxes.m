function bboxes = merge_rcnn_det_parts_bboxes(config)
% Reorder/score things based on the joint configuration
% bboxes is a cell array of length (1 + num_keypoints)
% The first cell array is the whole object detector
% Subsequent cell arrays are for the parts, and are aligned with the whole object bboxes
% Each element is a cell array of bboxes for each image
% The overall scores are in the object detector cell array
% If a part doesn't fire, then its bounding box is all zeros

bbox_ext_pix = 10;

% Load up the results
cachedir = fullfile(config.rcnn_root, 'cachedir', sprintf('%s_test', config.domain));
obj_fname = fullfile(cachedir, sprintf('%s_boxes_%s_test_bbox_reg.mat', config.domain, config.domain));

obj_bbox_file = load(obj_fname);
obj_boxes = obj_bbox_file.boxes;

fprintf('Load\n');
num_parts = config.final_part_pool_size;
part_boxes = cell(1, num_parts);
for i = 1:num_parts
  part_fname = fullfile(cachedir, sprintf('part_%d_boxes_%s_test_bbox_reg.mat', i, config.domain));
  part_file = load(part_fname);
  part_boxes{i} = part_file.boxes;
end

% Apply sigmoid
fprintf('Sigmoid\n');
sigmoid = @(x)1./(1+exp(-x));
num_ims = numel(obj_boxes);
for i = 1:num_ims
  obj_boxes{i}(:,end) = sigmoid(obj_boxes{i}(:,end));
  for j = 1:num_parts
    part_boxes{j}{i}(:,end) = sigmoid(part_boxes{j}{i}(:,end));
  end
end

% Get joint configurations
fprintf('Joint\n');
obj_inds = cell(1, num_ims);
part_inds = cell(1, num_ims);
bboxes = cell(1, 1 + num_parts);
for i = 1:numel(bboxes)
  bboxes{i} = cell(num_ims, 1);
end

bboxes_sep = cell(1, num_ims);
for i = 1:num_ims
  bboxes_sep{i} = cell(1, numel(bboxes));
  for k = 1:numel(bboxes)
    bboxes_sep{i}{k} = bboxes{k}{i};
  end
end


parfor i = 1:num_ims
  if mod(i, 256) == 1
    fprintf('joint %d/%d\n', i, num_ims);
  end
  obj_preds = obj_boxes{i};
  scores = zeros(size(obj_preds, 1), 1);
  obj_inds = zeros(size(obj_preds, 1), 1);
  part_inds = zeros(size(obj_preds, 1), num_parts);
  for j = 1:size(obj_preds, 1)
    obj_inds(j) = j;
    obj_bbox = obj_preds(j,1:4); % x1 y1 x2 y2
    obj_score = obj_preds(j,5);
    part_scores = zeros(1, num_parts);
    for k = 1:num_parts
      part = part_boxes{k}{i};
      if isempty(part)
        part_scores(k) = .1; % Doesn't matter
        continue;
      end
      ok_mask = ...
        obj_bbox(1) - part(:,1) <= bbox_ext_pix & ...
        obj_bbox(2) - part(:,2) <= bbox_ext_pix & ...
        part(:,3) - obj_bbox(3) <= bbox_ext_pix & ...
        part(:,4) - obj_bbox(4) <= bbox_ext_pix;
      if ~any(ok_mask)
        part_scores(k) = min(part(:,5));
        continue;
      else
        [part_scores(k), part_inds(j,k)] = max(part(:, 5) .* ok_mask);
      end
    end
    % Weigh scores for bbox and parts equally
    scores(j) = log(obj_score) + 1./num_parts * sum(log(part_scores));
  end
  % Align everything with the object bounding boxes
  bboxes_sep{i}{1} = obj_preds;
  for k = 1:num_parts
    bboxes_sep{i}{k+1} = zeros(size(bboxes_sep{i}{1}, 1), 4);
    has_part_mask = part_inds(:, k)>0;
    parti = part_inds(part_inds(:,k)>0, k);
    bboxes_sep{i}{k+1}(has_part_mask,:) = part_boxes{k}{i}(parti,1:4);
  end

  % Reorder based on the new scores
  if isempty(scores) % No detections
    continue;
  end
  [sorted_scores, inds] = sort(scores, 'descend');
  for j = 1:numel(bboxes_sep{i})
    bboxes_sep{i}{j} = bboxes_sep{i}{j}(inds,:);
  end


  % Remap scores to original space to keep AP evaluation sane.
  new_max = max(log(obj_boxes{i}(:,5)));
  new_min = min(log(obj_boxes{i}(:,5)));
  sorted_scores = (sorted_scores - min(sorted_scores)) ./ (range(sorted_scores)+eps);
  sorted_scores = sorted_scores * (new_max - new_min) + new_min;
  bboxes_sep{i}{1}(:, end) = sorted_scores;
end

for i = 1:num_ims
  for k = 1:numel(bboxes)
    bboxes{k}{i} = bboxes_sep{i}{k};
  end
end
