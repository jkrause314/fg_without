function features = extract_caffe(im, options)
% use_gpu, gpu_num, layer, mean_fname, use_center, use_corners, use_whole, use_flips


if ~isfield(options, 'use_gpu')
  options.use_gpu = true;
end
if ~isfield(options, 'layer')
  options.layer = 'fc6';
end
if ~isfield(options, 'gpu_num')
  options.gpu_num = 0;
end
if ~isfield(options, 'mean_fname')
  options.mean_fname = '';
end
if ~isfield(options, 'model_def_file')
  options.model_def_file = '';
end
if ~isfield(options, 'model_file')
  options.model_file = '';
end
if ~isfield(options, 'use_center')
  options.use_center = true;
end
if ~isfield(options, 'use_corners')
  options.use_corners = true;
end
if ~isfield(options, 'use_whole')
  options.use_whole = false;
end
if ~isfield(options, 'use_flips')
  options.use_flips = true;
end


persistent initialized;
if isempty(initialized) || ~initialized
  % Get the right number of images to use
  num_ims = (1 + options.use_flips) * (options.use_center + options.use_whole + 4 * options.use_corners);
  copyfile(options.model_def_file, options.temp_model_def_loc);
  system(sprintf('sed -i ''s#input_dim: 10#input_dim: %d#g'' %s', num_ims, options.temp_model_def_loc));

  if options.use_gpu
    fprintf('set device to %d\n', options.gpu_num);
    caffe('set_device', options.gpu_num);
  end
  matcaffe_init(options.use_gpu, options.temp_model_def_loc, options.model_file);
  initialized = true;
end

input_images = prepare_image(im, options);
num_ims = size(input_images, 4);
features = caffe('get_layer_features', {input_images}, options.layer);
features = reshape(features, [], num_ims);
end



function images = prepare_image(im, options)

persistent IMAGE_MEAN
if isempty(IMAGE_MEAN)
  d = load(options.mean_fname);
  IMAGE_MEAN = d.image_mean;
end
IMAGE_DIM = 256;

% Hack to get input dims correct and a warning to anyone who changes the net.
if strcmp(options.net, 'vgg') || strcmp(options.net, 'vgg-ft')
  CROPPED_DIM = 224;
else
  CROPPED_DIM = 227;
end

% resize to fixed input size
im = single(im);
im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
% permute from RGB to BGR (IMAGE_MEAN is already BGR)
im = im(:,:,[3 2 1]) - IMAGE_MEAN;

num_ims = (1 + options.use_flips) * (options.use_center + options.use_whole + 4 * options.use_corners);
% oversample (4 corners, center, and their x-axis flips)
images = zeros(CROPPED_DIM, CROPPED_DIM, 3, num_ims, 'single');
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
center = floor(indices(2) / 2)+1;
curr = 1;
if options.use_corners
  for i = indices
    for j = indices
      images(:, :, :, curr) = ...
          permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]);
      curr = curr + 1;
    end
  end
end
if options.use_center
  images(:,:,:,curr) = ...
      permute(im(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:), [2 1 3]);
  curr = curr + 1;
end
if options.use_whole
  images(:,:,:,curr) = ...
      permute(imresize(im, [CROPPED_DIM, CROPPED_DIM]), [2 1 3]);
  curr = curr + 1;
end
if options.use_flips
  images(:,:,:,curr:end) = images(end:-1:1,:,:,1:curr-1);
end
end
