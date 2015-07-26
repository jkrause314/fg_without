function train_oneclass(config)

fprintf('Discriminative combination of parts\n');
done_fname = fullfile(config.svm_out_dir, 'dcop.done');
if exist(done_fname, 'file')
  fprintf('Already done\n');
  return;
end

save_fname = fullfile(config.svm_out_dir, 'dcop.mat');

num_parts = config.final_part_pool_size + 1;

cv_dec_vals = cell(1, num_parts);
all_models = cell(1, num_parts);
for i = 1:num_parts
  fprintf('Load CV dec vals %d/%d\n', i, num_parts);
  svm_fname = fullfile(config.svm_out_dir, sprintf('svm.vgg.part.%d.mat', i));
  svm_data = load(svm_fname);
  [~, maxi] = max(svm_data.cv_accs);
  cv_dec_vals{i} = svm_data.cv_dec_vals{maxi};
  all_models{i} = svm_data.models{maxi};
end
fprintf('done.\n');

solver = config.dcop_solver;
c_vals = config.dcop_cs;
cv_accs = zeros(size(c_vals));
models = cell(1, numel(c_vals));

fprintf('Make one-class features\n');
label_data = load(fullfile(config.cnn_feat_dir, 'feats.train.part.1.net.vgg.mat'), 'all_labels');
all_labels = label_data.all_labels;
num_train = numel(all_labels);
assert(all(cellfun(@numel, all_labels) == 2));
train_labels = [all_labels{:}];
assert(numel(train_labels) == size(cv_dec_vals{1}, 2));
train_mult_factor = numel(train_labels) / num_train;
num_classes = numel(unique(train_labels));

oneclass_features = zeros(num_parts, (num_classes-1)*num_train*train_mult_factor, 'single');
for i = 1:num_parts
  fprintf('Make feats %d/%d\n', i, num_parts);
  assert(size(cv_dec_vals{i}, 2) == num_train*train_mult_factor);
  for j = 1:num_train*train_mult_factor
    im_vals = cv_dec_vals{i}(:, j);
    diffs = im_vals(train_labels(j)) - [im_vals(1:train_labels(j)-1); im_vals(train_labels(j)+1:end)];
    oneclass_features(i,(num_classes-1)*(j-1)+1:(num_classes-1)*j) = diffs;
  end
end
oneclass_labels = ones(1, size(oneclass_features, 2));
fprintf('done.\n');

parfor c_ind = 1:numel(c_vals)
  fprintf('Train c val %d/%d\n', c_ind, numel(c_vals));
  ll_options = [];
  ll_options.cv_folds = config.dcop_folds;
  ll_options.c = c_vals(c_ind);
  ll_options.bias = false;
  ll_options.solver = solver;
  tic;
  block_size = (num_classes - 1) * train_mult_factor;
  [model, cv_models, cv_inds] = my_train_cv_oneclass(oneclass_labels, oneclass_features, ll_options, block_size);
  toc;
  models{c_ind} = model;

  % Compute CV acc
  cv_inds_use = cv_inds(1:(num_classes-1):end);
  dec_vals = zeros(size(cv_dec_vals{1}));
  for i = 1:numel(cv_models)
    inds = find(cv_inds_use == i);
    oneclass_w = cv_models{i}.w;
    if numel(cv_models{i}.Label) == 2
      oneclass_w = [oneclass_w; -oneclass_w];
    end
    oneclass_ind = find(cv_models{i}.Label == 1);
    curr_dec_vals = zeros(num_classes, numel(inds));
    for j = 1:num_parts
      curr_dec_vals = curr_dec_vals + cv_dec_vals{j}(:, inds) * oneclass_w(oneclass_ind, j);
    end
    dec_vals(:, inds) = curr_dec_vals;
  end
  grouped_decs = zeros(size(dec_vals, 1), num_train);
  for i = 1:size(grouped_decs, 2);
    i1 = train_mult_factor*(i-1)+1;
    i2 = train_mult_factor*i;
    decs = dec_vals(:, i1:i2);
    grouped_decs(:, i) = sum(decs, 2);
  end
  [~, cv_preds] = max(grouped_decs, [], 1);
  cv_accs(c_ind) = mean(cv_preds(:) == reshape(train_labels(1:train_mult_factor:end), [], 1));
  fprintf('C: %g, cv acc: %g\n', ll_options.c, cv_accs(c_ind));
end

[~, best_c_ind] = max(cv_accs);
fprintf('\n\nBest:\n');
fprintf('C: %g, cv acc: %g, test acc: %g\n', c_vals(best_c_ind), cv_accs(best_c_ind));

save(save_fname, 'cv_accs', 'models');
