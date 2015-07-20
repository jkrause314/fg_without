function train_part_svms(config, net, cv_folds)
% Train svms

fprintf('Train part svms %s %s\n', config.domain, net);
done_fname = fullfile(config.svm_out_dir, sprintf('%s.done', net));
if exist(done_fname, 'file')
  fprintf('Already done\n');
  return;
end

num_parts = config.final_part_pool_size;

parfor part_ind = 1:num_parts+1
  fprintf('SVM %s, part %d/%d\n', net, part_ind, num_parts);
  save_fname = fullfile(config.svm_out_dir, sprintf('svm.%s.part.%d.mat', net, part_ind));
  if exist(save_fname, 'file')
    fprintf('Already done.\n');
    continue;
  end

  feat_fname = fullfile(config.cnn_feat_dir, sprintf('feats.train.part.%d.net.%s.mat', part_ind, net));
  assert(logical(exist(feat_fname, 'file')));

  fprintf('Loading...\n');
  train_feat_data = load(feat_fname);
  train_feats = single([train_feat_data.all_feats{:}]);
  train_labels = single(reshape([train_feat_data.all_labels{:}], [], 1));
  num_train = numel(train_feat_data.all_feats);
  mult_factor = size(train_feats, 2) / num_train;

  c_vals = config.svm_cs;
  cv_accs = zeros(size(c_vals));

  cv_dec_vals = cell(size(c_vals));
  models = cell(1, numel(c_vals));

  for c_ind = 1:numel(c_vals)
    fprintf('Train c val %d/%d\n', c_ind, numel(c_vals));
    ll_options = [];
    ll_options.cv_folds = cv_folds;
    ll_options.c = c_vals(c_ind);
    ll_options.bias = true;
    ll_options

    tic;
    [cv_acc, cv_preds, model, curr_cv_dec_vals] = my_train_cv(...
        train_labels, train_feats, ll_options, mult_factor);
    toc;

    cv_accs(c_ind) = cv_acc;
    models{c_ind} = model;
    cv_dec_vals{c_ind} = curr_cv_dec_vals;

    fprintf('C: %g, cv acc: %g\n', ll_options.c, cv_accs(c_ind));
  end
  [pardir, ~, ~] = fileparts(save_fname);
  if ~exist(pardir, 'dir')
    mkdir(pardir);
  end
  %save(save_fname, 'models', 'cv_accs', 'cv_dec_vals', 'll_options', 'mult_factor', 'net', '-v7.3');
  myparsave(save_fname, models, cv_accs, cv_dec_vals, ll_options, mult_factor, net);
end
fclose(fopen(done_fname, 'w'));
end

function myparsave(fname, models, cv_accs, cv_dec_vals, ll_options, mult_factor, net)
  save(fname, 'models', 'cv_accs', 'cv_dec_vals', 'll_options', 'mult_factor', 'net', '-v7.3');
end
