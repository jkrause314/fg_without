function [label, energy, index] = kmedoids(X,k, max_iters, progress_threshold)
% X: d x n data matrix
% k: number of cluster
% Written by Mo Chen (sth4nth@gmail.com)


if isa(X, 'single')
  X = double(X); % Matlab doesn't support sparse multiplies with single.
end
if ~exist('progress_threshold', 'var')
  progress_threshold = .001;
end
if ~exist('max_iters', 'var')
  max_iters = 100;
end


v = dot(X,X,1);
D = bsxfun(@plus,v,v')-2*(X'*X);
n = size(X,2);
sample = randperm(n);
sample = sample(1:k);
[~, label] = min(D(sample,:),[],1);
last = 0;
iter = 1;
while any(label ~= last) && ...
    nnz(label ~= last) > progress_threshold * numel(label) && ...
    iter <= max_iters
  fprintf('kmedoids, iter %d\n', iter);
  tic;
  [~, index] = min(D*sparse(1:n,label,1,n,k,n),[],1);
  last = label;
  [val, label] = min(D(index,:),[],1);
  fprintf('Num changed: %d of %d\n', nnz(label ~= last), numel(label));
  iter = iter + 1;
  toc;
end
energy = sum(val);
