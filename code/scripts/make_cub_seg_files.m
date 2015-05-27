function make_cub_seg_files(config)

seg_base_dir = config.cub_seg_dir;
OUT_DIR = fullfile(config.root, 'processed', 'data');
done_fname = fullfile(OUT_DIR, 'cubseg.done');
if exist(done_fname, 'file')
  fprintf('Already done!\n');
  return;
end

if ~exist(OUT_DIR, 'dir')
  mkdir(OUT_DIR);
end

%in_fnames = {'cub_images.mat', 'cub_images_test.mat', 'cub_images_train.mat', 'cub_images_train_flip.mat'};
%out_fnames = {'cub_segmentations.mat', 'cub_segmentations_test.mat', 'cub_segmentations_train.mat', 'cub_segmentations_train_flip.mat'};
in_fnames = {'cub_images_train.mat'};
out_fnames = {'cub_segmentations_train.mat'};

for f_ind = 1:numel(in_fnames)
  in_fname = in_fnames{f_ind};
  out_fname = out_fnames{f_ind};
  image_data = load(fullfile(OUT_DIR, in_fname));
  images = image_data.images;
  segmentations = cell(size(images));
  na = zeros(size(images));
  for i = 1:numel(images)
    if mod(i, 256) == 0
      fprintf('%d/%d\n', i, numel(images));
    end
    [pardir, base, ext] = fileparts(images(i).rel_path);
    seg_path = fullfile(seg_base_dir, pardir, [base '.png']);
    seg_im = double(imread(seg_path));
    uniq = unique(seg_im(:));
    num_annotators = numel(uniq) - 1;
    na(i) = num_annotators;
    assert(all(ismember(uniq, 0:51:255)));
    seg_im = seg_im ./ max(seg_im(:));
    bin_seg = seg_im >= .4;
    if images(i).flip
      bin_seg = fliplr(bin_seg);
    end
    segmentations{i} = bin_seg;
  end
  save(fullfile(OUT_DIR, out_fname), 'segmentations', '-v7.3');
end
fclose(fopen(done_fname, 'w'));
