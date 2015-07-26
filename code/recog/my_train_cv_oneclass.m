function [model, cv_models, cv_inds] = my_train_cv(labels, features, ll_options, block_size)
% Block size means that blocks of N elements will be in the same CV folds

if nargin < 4
  block_size = 1;
end

last_seed = rand('seed');
rand('seed', 0);

assert(isfield(ll_options, 'cv_folds'));
cv_folds = ll_options.cv_folds;
ll_options = rmfield(ll_options, 'cv_folds');

ll_str = options_to_ll_str(ll_options);

labels = double(labels(:));
features = single(features);
num_classes = numel(unique(labels));
num_examples = numel(labels);

assert(numel(labels)==size(features,2)); % Number of examples

% Make things in the same block end up in the same folds
if block_size == 1
  cv_inds = mod(randperm(numel(labels)), cv_folds) + 1;
  cv_preds = zeros(size(labels));
  cv_dec_vals = zeros(num_classes, num_examples);
else
  assert(mod(numel(labels), block_size) == 0);
  perm = randperm(numel(labels)/block_size);
  perm = reshape(repmat(perm, [block_size, 1]), 1, []);
  cv_inds = mod(perm, cv_folds) + 1;
end
cv_preds = zeros(size(labels));
cv_dec_vals = zeros(num_classes, num_examples);

% Do each fold
for i = 1:cv_folds
  test_mask = cv_inds == i;
  train_mask = ~test_mask;
  model = train(labels(train_mask), features(:, train_mask), ll_str, 'col');
  cv_models{i} = model;
  % Test set
  w = model.w;
  if numel(model.Label) == 2
    w = [w; -w];
  end
  if model.bias > 0
    bias = w(:, end);
    w = w(:, 1:end-1);
  end
  dec_vals = w * features(:, test_mask);
  if model.bias > 0
    dec_vals = bsxfun(@plus, dec_vals, bias);
  end
  [~, label_order] = sort(model.Label);
  cv_dec_vals(:, test_mask) = dec_vals(label_order, :);
  [~, pred_inds] = max(dec_vals, [], 1);
  cv_preds(test_mask) = model.Label(pred_inds);
end
cv_acc = mean(cv_preds(:) == labels(:));

%fprintf('CV Accuracy: %g\n', cv_acc*100);

if nargout >= 3
  model = train(labels, features, ll_str, 'col');
end

rand('seed', last_seed);
