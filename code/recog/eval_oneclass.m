function eval_oneclass(config)

if exist(config.test_result_fname, 'file')
  fprintf('Already done, loading results...\n')
  res_data = load(config.test_result_fname);
  fprintf('Accuracy: %g\n', res_data.test_acc);
  return;
end

% Calculate test decision values
num_parts = config.final_part_pool_size + 1;
test_dec_vals = cell(1, num_parts);
for i = 1:num_parts
  fprintf('Test dec vals %d/%d\n', i, num_parts);
  % Get the best train model base on non-ft, load the ft model
  c_svm_data = load(fullfile(config.svm_out_dir, sprintf('svm.vgg.part.%d.mat', i)));
  [~, maxi] = max(c_svm_data.cv_accs);
  svm_data = load(fullfile(config.svm_out_dir, sprintf('svm.vgg-ft.part.%d.mat', i)));
  model = svm_data.models{maxi};

  % Get test features
  test_data = load(fullfile(config.cnn_feat_dir, sprintf('feats.test.part.%d.net.vgg-ft.mat', i)));

  assert(all(cellfun(@(x)size(x,2), test_data.all_feats)==2));
  test_feats = single([test_data.all_feats{:}]);
  num_test = numel(test_data.all_labels);
  test_labels = single(reshape([test_data.all_labels{:}], [], 1));

  % Decision values
  w = model.w;
  if numel(model.Label) == 2
    w = [w; -w];
  end
  if model.bias > 0
    bias = w(:, end);
    w = w(:, 1:end-1);
  end
  curr_dec_vals = w * test_feats;
  if model.bias > 0
    curr_dec_vals = bsxfun(@plus, curr_dec_vals, bias);
  end
  [~, label_order] = sort(model.Label);
  curr_dec_vals = curr_dec_vals(label_order, :);
  test_dec_vals{i} = curr_dec_vals;

end

% Load DCoP
dcop_data = load(config.dcop_fname);
[~, maxi] = max(dcop_data.cv_accs);
model = dcop_data.models{maxi};

% Apply the DCoP
oneclass_w = model.w;
if numel(model.Label) == 2
  oneclass_w = [oneclass_w; -oneclass_w];
end
assert(model.bias<=0);
dec_vals = zeros(size(test_dec_vals{1}));
oneclass_ind = find(model.Label == 1);
for i = 1:num_parts
  dec_vals = dec_vals + test_dec_vals{i} * oneclass_w(oneclass_ind, i);
end

test_mult_factor = size(dec_vals, 2) / num_test;
grouped_decs = zeros(size(dec_vals, 1), num_test);
for i = 1:num_test
  i1 = test_mult_factor*(i-1)+1;
  i2 = test_mult_factor*(i-1)+2;
  grouped_decs(:, i) = sum(dec_vals(:, i1:i2), 2);
end
[~, test_preds] = max(grouped_decs, [], 1);
grouped_labels = test_labels(1:test_mult_factor:end);
test_acc = mean(test_preds(:) == grouped_labels(:));

% Print and save results
fprintf('Test accuracy: %g\n', test_acc);

save(config.test_result_fname, 'test_acc', 'test_preds');
