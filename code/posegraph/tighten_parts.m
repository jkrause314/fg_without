function tighten_parts(config)
fprintf('Tighten parts\n');

region_out_fname = config.tight_part_fname;

if exist(region_out_fname, 'file')
  fprintf('Already done!\n');
  return;
end

fprintf('Loading...\n');
seg_data = load(config.coseg_save_fname);
kp_data = load(config.propagate_save_fname);
im_data = load(config.train_image_fname);
fprintf('Done\n');

part_context = config.part_context;

segs = seg_data.segmentations;
keypoints = kp_data.keypoints;
images = im_data.images;

num_kps = size(keypoints, 1);
num_ims = size(keypoints, 3);
regions = zeros(num_kps, 4, num_ims);

for i = 1:num_ims
  if mod(i, 128) == 0
    fprintf('Tighten part: %d/%d\n', i, num_ims);
  end
  bbox = images(i).bbox;

  im = images(i);
  seg = segs{i};
  if size(seg,1) ~= im.height || size(seg,2) ~= im.width
    seg = imresize(seg, [im.height, im.width]);
  end
  kp_x = keypoints(:, 1, i);
  kp_y = keypoints(:, 2, i);
  [x1 y1 x2 y2] = kps_to_region(kp_x, kp_y, seg, part_context);
  regions(:,:,i) = [x1 y1 x2 y2];
  % Kill any regions which end up too small
  if any(x2 - x1 < 10) || any(y2 - y1 < 10)
    regions((x2-x1<10)|(y2-y1<10),:,i) = 0;
  end
end
save(region_out_fname, 'regions');
end



function [x1 y1 x2 y2] = kps_to_region(kp_x, kp_y, seg, part_context)
  kp_x = max(1, min(size(seg, 2), round(kp_x)));
  kp_y = max(1, min(size(seg, 1), round(kp_y)));

  % Project keypoint?
  [dist,dist_ind] = bwdist(seg);
  ind = sub2ind(size(seg), kp_y, kp_x);
  [row, col] = ind2sub(size(seg), dist_ind(ind));
  kp_x = col;
  kp_y = row;

  in_col = any(seg, 1);
  bbox_x1 = find(in_col, 1, 'first');
  bbox_x2 = find(in_col, 1, 'last');
  in_row = any(seg, 2);
  bbox_y1 = find(in_row, 1, 'first');
  bbox_y2 = find(in_row, 1, 'last');

  % Make a box around the keypoint using the aspect ratio of the bbox
  % Shrink each side until it's in the foreground by some margin

  bbox_width  = (bbox_x2 - bbox_x1 + 1);
  bbox_height = (bbox_y2 - bbox_y1 + 1);

  sl = sqrt(bbox_width * bbox_height);
  bbox_width = sl;
  bbox_height = sl;

  x1 = round(max(1,            kp_x - part_context * bbox_width));
  x2 = round(min(size(seg, 2), kp_x + part_context * bbox_width));
  y1 = round(max(1,            kp_y - part_context * bbox_height));
  y2 = round(min(size(seg, 1), kp_y + part_context * bbox_height));

  num_kps = numel(x1);

  edge_fg = .0001;
  y1_temp = y1;
  y2_temp = y2;
  x1_temp = x1;
  x2_temp = x2;
  for i = 1:num_kps
    while mean(seg(y1_temp(i),x1(i):x2(i))) < edge_fg && y1_temp(i) < kp_y(i)
      y1_temp(i) = y1_temp(i) + 1;
    end
    while mean(seg(y2_temp(i),x1(i):x2(i))) < edge_fg && y2_temp(i) > kp_y(i)
      y2_temp(i) = y2_temp(i) - 1;
    end
    while mean(seg(y1(i):y2(i),x1_temp(i))) < edge_fg && x1_temp(i) < kp_x(i)
      x1_temp(i) = x1_temp(i) + 1;
    end
    while mean(seg(y1(i):y2(i),x2_temp(i))) < edge_fg && x2_temp(i) > kp_x(i)
      x2_temp(i) = x2_temp(i) - 1;
    end
  end
  x1 = x1_temp;
  x2 = x2_temp;
  y1 = y1_temp;
  y2 = y2_temp;
end
