function make_msts(config)


save_fname = config.mst_save_fname;
if exist(save_fname, 'file')
  fprintf('Already did msts.\n');
  return;
end

[out_dir, ~, ~] = fileparts(save_fname);
if ~exist(out_dir, 'dir')
  mkdir(out_dir);
end

fprintf('Computing msts...\n');

% Load features
fprintf('Loading features...\n');
feat_fname = config.cnn_bbox_fname;
feat_data = load(feat_fname);
features = cat(2, feat_data.all_feats{:});
features = bsxfun(@rdivide, features, sqrt(sum(features.^2, 1))); % L2-normalize

fprintf('Compute distances\n');
dists = mypdist(features');

num_msts = config.num_msts;

dists = sparse(double((dists + dists')/2)); 
msts = cell(1, num_msts);

for i = 1:num_msts
  fprintf('mst %d/%d\n', i, num_msts);
  tic;
  tree = mst(dists);
  toc;
  msts{i} = tree;
  dists(find(tree)) = 0;
end
save(save_fname, 'msts', '-v7.3');
