function run_matching(config)

bbox_context = .05;
resize_height = 150;
try_flip = true;
mst_fname = config.mst_save_fname;
seg_fname = config.coseg_save_fname;

im_out_dir = config.matching_im_out_dir;
imdata_fname = config.train_imagedata_fname;
im_base = config.im_base;

save_fname = config.alignment_fname;

if exist(save_fname, 'file')
  fprintf('Matching already done!\n');
  return;
end

if ~exist(im_out_dir, 'dir')
  mkdir(im_out_dir);
end

% Get the pose graph
fprintf('Matching, loading files...\n');
mst_data = load(mst_fname);
graph = mst_data.msts{1};
for i = 2:numel(mst_data.msts)
  graph = graph + mst_data.msts{i};
end

% Load segmentations and images
seg_data = load(seg_fname);
segmentations = seg_data.segmentations;
im_data = load(imdata_fname);
images = im_data.images;
all_ims = im_data.ims;
clear('im_data');
fprintf('Done loading.\n');

for i = 1:numel(images)
  fprintf('Matching %d/%d\n', i, numel(images));
  out_base = sprintf('%06d', i);
  out_fname = fullfile(im_out_dir, [out_base '.mat']);
  if exist(out_fname, 'file')
    continue;
  end

  from = i;
  to_inds = find(graph(from,:)>0);

  inds = [from to_inds];
  local_images = images(inds);
  segs = segmentations(inds);
  ims = all_ims(inds);
  for j = 1:numel(local_images)
    im = ims{j};
    segs{j} = imresize(segs{j}, [size(im, 1) size(im, 2)]);
    bbox = local_images(j).bbox;
    x1 = round(max(1, bbox.x1 - bbox_context * (bbox.x2 - bbox.x1 + 1)));
    x2 = round(min(size(im, 2), bbox.x2 + bbox_context * (bbox.x2 - bbox.x1 + 1)));
    y1 = round(max(1, bbox.y1 - bbox_context * (bbox.y2 - bbox.y1 + 1)));
    y2 = round(min(size(im, 1), bbox.y2 + bbox_context * (bbox.y2 - bbox.y1 + 1)));
    im = im(y1:y2,x1:x2,:);
    scale = resize_height / size(im, 1);
    im = imresize(im, scale);
    ims{j} = im;
    segs{j} = imresize(segs{j}(y1:y2,x1:x2), scale);
  end
  % Do alignment
  % TODO: Parallelize this better
  warp_params = compute_warp_params(segs{1}, segs, struct('try_flip', try_flip));
  assert(numel(warp_params) == numel(to_inds)+1);
  warp_params = warp_params(2:end);
  save(out_fname, 'warp_params', 'to_inds', 'from');
end


% Aggregate
fprintf('Aggregate matching files...\n');
num_ims = numel(images);
warp_params = cell(1, num_ims);
to_inds = cell(1, num_ims);
from_inds = cell(1, num_ims);
for i = 1:num_ims
  if mod(i, 256) == 0
    fprintf('%d/%d\n', i, num_ims);
  end
  align_data = load(fullfile(im_out_dir, sprintf('%06d.mat', i)));
  from = i;
  warp_params{from} = align_data.warp_params;
  to_inds{from} = align_data.to_inds;
  for to = align_data.to_inds
    from_inds{to} = [from_inds{to} from];
  end
end
fprintf('Saving...\n');
save(save_fname, 'warp_params', 'to_inds', 'from_inds');
