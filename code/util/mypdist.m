function D = mypdist(X)
% Only does euclidean distance
% Rows are instances
X_rsum = sum(X.^2, 2);
D = sqrt(max(0, bsxfun(@plus, bsxfun(@plus, -2*X*X', X_rsum), X_rsum')));
