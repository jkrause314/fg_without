function imdb = imdb_from_fg_domain(config, image_set)
% Modified to handle parts
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

%imdb.name = 'voc_train_2007'
%imdb.image_dir = '/work4/rbg/VOC2007/VOCdevkit/VOC2007/JPEGImages/'
%imdb.extension = '.jpg'
%imdb.image_ids = {'000001', ... }
%imdb.sizes = [numimages x 2]
%imdb.classes = {'aeroplane', ... }
%imdb.num_classes
%imdb.class_to_id
%imdb.class_ids
%imdb.eval_func = pointer to the function that evaluates detections
%imdb.roidb_func = pointer to the function that returns regions of interest

name = sprintf('%s_%s', config.domain, image_set);
cache_file = sprintf('./imdb/cache/imdb_%s.mat', name);
try
  load(cache_file);
catch
  imdb.name = name;
  imdb.image_set = image_set;
  imdb.image_dir = config.im_base;

  if strcmp(image_set, 'train')
    imdb.im_data_fname = config.train_image_fname;
  elseif strcmp(image_set, 'test')
    imdb.im_data_fname = config.test_image_fname;
  else
    error('bad image_set for cub');
  end
  part_data = load(config.tight_part_fname);
  train_regions = part_data.regions;
  num_parts = size(train_regions, 1);

  im_data = load(imdb.im_data_fname);
  images = im_data.images;
  num_ims = numel(images);
  image_ids = cell(1, num_ims);
  sizes = zeros(num_ims, 2);
  for i = 1:numel(images)
    [pardir, base, ext] = fileparts(images(i).rel_path);
    image_ids{i} = fullfile(pardir, base);
    assert(strcmp(ext, '.jpg'))
    sizes(i,:) = [images(i).height images(i).width];
  end
  imdb.image_ids = image_ids;
  imdb.sizes = sizes;
  imdb.extension = '.jpg';
  imdb.classes = {config.domain};
  for i = 1:num_parts
    imdb.classes = [imdb.classes sprintf('part_%d', i)];
  end
  imdb.num_classes = numel(imdb.classes);
  imdb.class_ids = 1:imdb.num_classes;
  imdb.class_to_id = containers.Map(imdb.classes, imdb.class_ids);
  imdb.details = [];
  imdb.eval_func = @(varargin)imdb_eval_fg_domain(config, varargin{:});
  imdb.roidb_func = @(varargin)roidb_from_fg_domain(config, varargin{:});
  imdb.image_at = @(i)fullfile(imdb.image_dir, [imdb.image_ids{i} imdb.extension]);

  fprintf('Saving imdb to cache...');
  save(cache_file, 'imdb', '-v7.3');
  fprintf('done\n');
end
