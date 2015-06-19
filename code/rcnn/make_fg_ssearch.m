function make_fg_ssearch(config)


image_sets = {'train', 'test'};
target_im_width = 500;
fast_mode = true;

for i = 1:numel(image_sets)

  in_fname = fullfile('imdb', 'cache', sprintf('imdb_%s_%s.mat', config.domain, image_sets{i}));
  out_fname = fullfile('data', 'selective_search_data', sprintf('%s_%s.mat', config.domain, image_sets{i}));

  if exist(out_fname, 'file')
    fprintf('ssearch %s %s already done\n', config.domain, image_sets{i});
    continue;
  end
  fprintf('Selective search %s %s', config.domain, image_sets{i});
  in_data = load(in_fname);
  imdb = in_data.imdb;
  % boxes
  % images
  boxes = cell(1, numel(imdb.image_ids));
  images = imdb.image_ids;
  parfor j = 1:numel(images)
    fprintf('Im %d/%d\n', j, numel(images));
    tic;
    im = imread(fullfile(imdb.image_dir, [images{j} imdb.extension]));
    boxes{j} = selective_search_boxes(im, fast_mode, target_im_width);
    %boxes{i} = selective_search_boxes(im, fast_mode, size(im, 2));
    toc;
  end
  [pardir, ~, ~] = fileparts(out_fname);
  if ~exist(pardir, 'dir')
    mkdir(pardir);
  end
  save(out_fname, 'boxes', 'images');
end
