function do_coseg(params)

coseg_save_dir = params.im_out_dir;

im_data = load(params.im_meta_fname);
images = im_data.images;
classes = unique([images.class]);

% Options for the mex
options = struct(...
  'num_iters', params.coseg_iters, ...
  'class_weight', params.class_weight, ...
  'use_class_fg', params.use_class_fg, ...
  'use_class_bg', params.use_class_bg, ...
  'fg_prior', params.fg_prior, ...
  'do_refine', params.do_refine ...
);

class_cosegs = cell(1, numel(classes));

parfor class_ind = 1:numel(classes)
  test_class = classes(class_ind); 
  fprintf('Coseg class %d\n', test_class);

  % Load images and set them up for coseg
  stime = tic;
  im_inds = find([images.class] == test_class);
  class_ims = cell(1, numel(im_inds));
  class_gt_masks = cell(1, numel(im_inds));
  class_min_fg_areas = zeros(1, numel(im_inds));
  class_max_fg_areas = 1e6*ones(1, numel(im_inds));
  class_min_fg_lengths = zeros(1, numel(im_inds));
  class_min_fg_heights = zeros(1, numel(im_inds));
  for i = 1:numel(im_inds)
    ind_str = sprintf('%06d', im_inds(i));

    % Get the image
    im = imread(fullfile(params.im_base, images(im_inds(i)).rel_path));
    if size(im, 3) == 1
      im = repmat(im, [1,1,3]);
    end
    scale = sqrt(params.resize_area/(size(im,1) * size(im,2)));
    im_scaled = imresize(im, scale);
    class_ims{i} = im_scaled;

    % Set up GT mask using the bounding box
    class_gt_masks{i} = 10*ones(size(im_scaled, 1), size(im_scaled, 2), 'uint8');
    % Bbox init?
    bbox = images(im_inds(i)).bbox;
    x1 = max(1, round((bbox.x1-1)*scale+1 - params.bbox_context));
    x2 = min(size(im_scaled, 2), round((bbox.x2-1)*scale+1 + params.bbox_context));
    y1 = max(1, round((bbox.y1-1)*scale+1 - params.bbox_context));
    y2 = min(size(im_scaled, 1), round((bbox.y2-1)*scale+1 + params.bbox_context));

    % GC_BGD = 0, GC_FGD = 1, GC_PR_BGD = 2, GC_PR_FGD = 3
    class_gt_masks{i}(:) = 0;
    class_gt_masks{i}(y1:y2,x1:x2) = 3;
    if ~any(class_gt_masks{i} == 0) % Bbox covers whole image
      class_gt_masks{i}(1,:) = 0;
      class_gt_masks{i}(:,1) = 0;
      class_gt_masks{i}(end,:) = 0;
      class_gt_masks{i}(:,end) = 0;
    end
    assert(any(class_gt_masks{i}(:) == 3));
    assert(any(class_gt_masks{i}(:) <= 2));
    bbox_length = x2 - x1 + 1;
    bbox_height = y2 - y1 + 1;
    bbox_area = bbox_length * bbox_height;
    class_min_fg_areas(i) = params.bbox_min_fg_area * bbox_area;
    class_max_fg_areas(i) = params.bbox_max_fg_area * bbox_area;
    class_min_fg_lengths(i) = params.bbox_min_fg_length * bbox_length;
    class_min_fg_heights(i) = params.bbox_min_fg_height * bbox_height;
  end

  elapsed = toc(stime);
  fprintf('Loading time class %d: %g sec\n', test_class, elapsed);

  stime = tic;
  % Do actual coseg
  tmaps = myCoseg(class_ims, class_gt_masks, class_min_fg_areas, class_max_fg_areas, class_min_fg_lengths, class_min_fg_heights, options);
  new_class_cosegs = cellfun(@(x)(x==3 | x==1) , tmaps, 'uniformoutput', false);
  coseg_elapsed = toc(stime);
  fprintf('Coseg time class %d: %g sec/im.\n', test_class, coseg_elapsed/numel(class_ims));

  for i = 1:numel(im_inds)
    ind_str = sprintf('%06d', im_inds(i));
    save_fname = fullfile(coseg_save_dir, ind_str);
    seg = new_class_cosegs{i};
    parsave(save_fname, struct('segmentation', seg));
  end
  class_cosegs{class_ind} = new_class_cosegs;
end


fprintf('Aggregate and save\n');
segmentations = cell(1, numel(images));
for i = 1:numel(classes)
  test_class = classes(i); 
  im_inds = find([images.class] == test_class);
  segmentations(im_inds) = class_cosegs{i};
end

save(params.coseg_save_fname, 'segmentations', '-v7.3');
fprintf('saved to %s\n', save_fname);

