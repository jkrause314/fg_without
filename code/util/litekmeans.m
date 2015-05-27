function [center, label] = litekmeans(X, k, max_iters, progress_threshold, do_spherical, verbose)
% X is num_dims x num_data_points

if isa(X, 'single')
  X = double(X); % Matlab doesn't support sparse multiplies with single.
end

if ~exist('progress_threshold', 'var')
  progress_threshold = .001;
end
if ~exist('max_iters', 'var')
  max_iters = 100;
end
if ~exist('do_spherical', 'var')
  do_spherical = false;
end
if ~exist('verbose', 'var')
  verbose = false;
end


if verbose
  fprintf('k-means got %d samples, want %d clusters\n', size(X, 2), k);
end
n = size(X,2);
last = 0;
label = ceil(k*rand(1,n));  % random initialization
iter = 1;
block_size = 100000;
any_empty = true;
while any_empty || ...
  (any(label ~= last) && ...
    nnz(label ~= last) > progress_threshold * numel(label) && ...
    iter <= max_iters)
  if verbose
    fprintf('kmeans, iter %d\n', iter);
  end
  s=tic;
  E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
  % Split clusters if any are empty
  cluster_counts = full(sum(E, 1));
  empty_clusters = find(cluster_counts==0);
  any_empty = ~isempty(empty_clusters);
  if ~isempty(empty_clusters)
    if verbose
      fprintf('%d empty clusters, splitting...\n', numel(empty_clusters));
    end
    for i = 1:numel(empty_clusters)
      empty_cluster = empty_clusters(i);
      % Split the max cluster
      [max_count, maxi] = max(cluster_counts);
      max_inds = find(E(:, maxi) == 1);
      assert(max_count == numel(max_inds));
      is_new_class = rand(1, max_count) > .5;
      E(max_inds(is_new_class), maxi) = 0;
      E(max_inds(is_new_class), empty_cluster) = 1;
      cluster_counts = full(sum(E, 1));
      assert(sum(cluster_counts) == n);
    end
  end

  center = X*(E*spdiags(1./sum(E,1)',0,k,k));
  if do_spherical
    center = bsxfun(@rdivide, center, sqrt(sum(center.^2, 1)));
  end

  last = label;
  for i = 1:block_size:n
    i1 = i;
    i2 = min(i + block_size - 1, n);
    % assign samples to the nearest centers
    [val, block_label] = max(bsxfun(@minus,center'*X(:, i1:i2),0.5*sum(center.^2,1)'), [], 1);
    label(i1:i2) = block_label;
  end
  if verbose
    fprintf('Num changed: %d of %d\n', nnz(label ~= last), numel(label));
  end
  iter = iter + 1;
  if verbose
    toc(s);
  end
end

