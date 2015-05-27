function [fx fy] = do_point_warp(x, y, wp, imsize1, imsize2)
% Warp from im1 to im2

x = x(:);
y = y(:);

if isempty(wp)
  fx = (x - 1) * imsize2(2) / imsize1(2) + 1;
  fy = (y - 1) * imsize2(1) / imsize1(1) + 1;
  return;
end

cx = wp.cx;
cy = wp.cy;
X3b = wp.X3b;
n_good = wp.n_good;
M = length(x);
fx_aff=cx(n_good+1:n_good+3)'*[ones(1,M); x'; y'];
%d2=dist2(X3b,[x y]);
d2=mypdist2(X3b,[x y]).^2;
fx_wrp=cx(1:n_good)'*(d2.*log(d2+eps));
fx=fx_aff+fx_wrp;
fy_aff=cy(n_good+1:n_good+3)'*[ones(1,M); x'; y'];
fy_wrp=cy(1:n_good)'*(d2.*log(d2+eps));
fy=fy_aff+fy_wrp;
if isfield(wp, 'flipped') && wp.flipped
  fx = imsize2(2) - fx + 1;
end
