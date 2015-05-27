function D = mypdist2(X, Y)
% Only does euclidean distance
% Rows are instances
X_rsum = sum(X.^2, 2);
Y_rsum = sum(Y.^2, 2);
D = sqrt(max(0, bsxfun(@plus, bsxfun(@plus, -2*X*Y', X_rsum), Y_rsum')));
