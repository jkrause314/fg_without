function part_propagate(config)

num_keypoints = config.large_part_pool_size;

expand_frac_lower = .4;
expand_frac_upper = .8;
num_keypoints_subsample = config.final_part_pool_size;

save_fname = config.propagate_save_fname;
if exist(save_fname, 'file')
  fprintf('Already done!\n');
  return;
end


fprintf('Part propagate, loading...\n');
align_data = load(config.alignment_fname);
seg_data = load(config.coseg_save_fname);
segs = seg_data.segmentations;
mst_data = load(config.mst_save_fname);
msts = mst_data.msts;
im_data = load(config.train_imagedata_fname);
images = im_data.images;
all_ims = im_data.ims;
fprintf('Done.\n');

% Form pose graph
graph = msts{1};
for i = 2:numel(msts)
  graph = graph + msts{i};
end
graph = graph>0;
num_ims = size(graph, 1);

% Select the parameters that are actually part of the graph
all_warp_params = align_data.warp_params;
all_to_inds = align_data.to_inds;
all_from_inds = align_data.from_inds;
for i = 1:num_ims
  old_to = all_to_inds{i};
  actual_to = find(graph(i,:));
  keep_mask = ismember(old_to, actual_to);
  all_warp_params{i} = all_warp_params{i}(keep_mask);
  all_to_inds{i} = actual_to;
  all_from_inds{i} = actual_to; % Undirected edges
end

% Pick an image to sample from initially based on connectivity
[~, root_ind] = max(sum(graph, 1));

root_im = all_ims{root_ind};
root_seg = imresize(segs{root_ind}, [size(root_im, 1) size(root_im, 2)]);
bbox_context = .05;
resize_height = 150;
bbox = images(root_ind).bbox;
x1 = round(max(1, bbox.x1 - bbox_context * (bbox.x2 - bbox.x1 + 1)));
x2 = round(min(size(root_im, 2), bbox.x2 + bbox_context * (bbox.x2 - bbox.x1 + 1)));
y1 = round(max(1, bbox.y1 - bbox_context * (bbox.y2 - bbox.y1 + 1)));
y2 = round(min(size(root_im, 1), bbox.y2 + bbox_context * (bbox.y2 - bbox.y1 + 1)));
root_im = root_im(y1:y2,x1:x2,:);
scale = resize_height / size(root_im, 1);
root_im = imresize(root_im, scale);
root_seg = imresize(root_seg(y1:y2,x1:x2), scale);


% Sample some points using our segmentation
fprintf('keypoint sample\n');
rand('seed', 0);
s = root_seg;
inds = find(s==1);
if numel(inds) > 5000
  perm = randperm(numel(inds));
  inds = inds(perm(1:5000));
elseif numel(inds) < num_keypoints
  perm = randperm(numel(s));
  inds = perm(1:5000);
end
[rows,cols] = ind2sub(size(s), inds);
X = [rows(:)'; cols(:)'];
%[center, label] = litekmeans(X, num_keypoints, 1000, 0);
[label, energy, index] = kmedoids(X, num_keypoints, 1000, 0);
center = X(:,index);
center = round(center);
x = center(2,:);
y = center(1,:);

num_incoming_edges = zeros(1, num_ims);
done = false(1, num_ims);
done(root_ind) = true;
num_done = 1;
num_incoming_edges(graph(root_ind,:)) = num_incoming_edges(graph(root_ind,:)) + 1;
keypoints = zeros(numel(x), 2, num_ims, 'single');
keypoints(:,1,root_ind) = x;
keypoints(:,2,root_ind) = y;
rand('seed', 0);


fprintf('resize ims,segs\n');
if ~exist('ims_resized', 'var')
  ims_resized = cell(1, num_ims);
  segs_resized = cell(1, num_ims);
  % TODO: Parallelize
  for i = 1:num_ims
    if mod(i, 128) == 0
      fprintf('resize %d/%d\n', i, num_ims);
    end
    im = all_ims{i};
    seg = imresize(segs{i}, [size(im, 1), size(im, 2)]);
    bbox = images(i).bbox;
    x1 = round(max(1, bbox.x1 - bbox_context * (bbox.x2 - bbox.x1 + 1)));
    x2 = round(min(size(im, 2), bbox.x2 + bbox_context * (bbox.x2 - bbox.x1 + 1)));
    y1 = round(max(1, bbox.y1 - bbox_context * (bbox.y2 - bbox.y1 + 1)));
    y2 = round(min(size(im, 1), bbox.y2 + bbox_context * (bbox.y2 - bbox.y1 + 1)));
    im = im(y1:y2,x1:x2,:);
    scale = resize_height / size(im, 1);
    resize_scales(i) = scale;
    ims_resized{i} = imresize(im, scale);
    segs_resized{i} = imresize(seg(y1:y2,x1:x2), scale);
  end
end
clear('all_ims');


while ~all(done)
  fprintf('Propagate %d/%d\n', num_done, num_ims);
  % Pick the image to propagate to
  max_inc = max(num_incoming_edges(~done));
  cand_inds = find(num_incoming_edges == max_inc & ~done);
  to = cand_inds(randi(numel(cand_inds)));
  from_inds = intersect(all_from_inds{to}, find(done));

  % Propagate
  kp_preds = zeros(num_keypoints, 2, numel(from_inds));
  for from_ind = 1:numel(from_inds)
    from = from_inds(from_ind);
    from_x = keypoints(:,1,from);
    from_y = keypoints(:,2,from);
    wp = all_warp_params{from}{find(all_to_inds{from}==to)};
    [fx, fy] = do_point_warp(from_x, from_y, wp, size(segs_resized{from}), size(segs_resized{to}));

    assert(isreal(fx));
    kp_preds(:,1,from_ind) = fx;
    kp_preds(:,2,from_ind) = fy;
  end

  % Get rid of failed propagations
  fx = zeros(size(kp_preds, 1), 1);
  fy = zeros(size(kp_preds, 1), 1);
  for i = 1:size(kp_preds, 1)
    % If things are only a little outside the window, crop them.
    tol = 10;
    kp_preds(i,1,(kp_preds(i,1,:) < 1 & kp_preds(i,1,:)>-tol)) = 1;
    kp_preds(i,2,(kp_preds(i,2,:) < 1 & kp_preds(i,2,:)>-tol)) = 1;
    kp_preds(i,1,(kp_preds(i,1,:) > size(segs_resized{to},2) & ...
      kp_preds(i,1,:)<size(segs_resized{to},2)+tol)) = size(segs_resized{to},2);
    kp_preds(i,2,(kp_preds(i,2,:) > size(segs_resized{to},1) & ...
      kp_preds(i,2,:)<size(segs_resized{to},1)+tol)) = size(segs_resized{to},1);

    good_mask = kp_preds(i,1,:) >= 1 & kp_preds(i,1,:) <= size(segs_resized{to}, 2) & ...
      kp_preds(i,2,:) >= 1 & kp_preds(i,2,:) <= size(segs_resized{to}, 1);
    assert(any(good_mask));
    fx(i) = median(kp_preds(i,1,good_mask), 3);
    fy(i) = median(kp_preds(i,2,good_mask), 3);
  end

  % Make sure occupies some fraction of the image. Should be very rare.
  xmin = min(fx);
  xmax = max(fx);
  ymin = min(fy);
  ymax = max(fy);
  if (xmax - xmin + 1) < expand_frac_lower * size(segs_resized{to}, 2)
    fprintf('expand x\n');
    % xmin -> (1/2 - expand_frac/2) * width
    % xmax -> (1/2 + expand_frac/2) * width
    new_min = (1/2 - expand_frac_upper/2) * size(segs_resized{to}, 2);
    new_max = (1/2 + expand_frac_upper/2) * size(segs_resized{to}, 2);
    m = (new_max - new_min) / (xmax - xmin);
    b = new_min - m * xmin;
    fx = m * fx + b;
  end
  if (ymax - ymin + 1) < expand_frac_lower * size(segs_resized{to}, 1)
    fprintf('expand y\n');
    new_min = (1/2 - expand_frac_upper/2) * size(segs_resized{to}, 1);
    new_max = (1/2 + expand_frac_upper/2) * size(segs_resized{to}, 1);
    m = (new_max - new_min) / (ymax - ymin);
    b = new_min - m * ymin;
    fy = m * fy + b;
  end

  keypoints(:,1,to) = fx;
  keypoints(:,2,to) = fy;
  done(to) = true;
  num_incoming_edges(graph(to,:)) = num_incoming_edges(graph(to,:)) + 1;
  num_done = num_done + 1;
end

% Put back in the coordinates of the original images
fprintf('Coordinate change\n');
for i = 1:num_ims
  keypoints(:,:,i) = keypoints(:,:,i) ./ resize_scales(i);
  keypoints(:,1,i) = keypoints(:,1,i) + double(images(i).bbox.x1) - 1;
  keypoints(:,2,i) = keypoints(:,2,i) + double(images(i).bbox.y1) - 1;
end

% Cluster based on trajectories
num_ims = size(keypoints, 3);
dim = 2 * num_ims;
X = zeros(dim, size(keypoints, 1));
for i = 1:size(keypoints, 1)
  X(:, i) = reshape(keypoints(i, :, :), [], 1);
end
[center, label] = litekmeans(X, num_keypoints_subsample, 100, 0);
sampled_keypoints = zeros(num_keypoints_subsample, 2, num_ims);
for i = 1:num_keypoints_subsample
  sampled_keypoints(i,:,:) = reshape(center(:,i), 2, num_ims);
end
keypoints = sampled_keypoints;

save(save_fname, 'keypoints');
