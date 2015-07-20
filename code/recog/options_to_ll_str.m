function [ll_str, ll_str_nocv] = options_to_ll_str(options)

ll_str = '-q';
if options.bias
  ll_str = [ll_str ' -B 1'];
else
  ll_str = [ll_str ' -B -1'];
end

if isfield(options, 'c')
  ll_str = sprintf('%s -c %g', ll_str, options.c);
end

if isfield(options, 'solver')
  ll_str = sprintf('%s -s %d', ll_str, options.solver);
end

if isfield(options, 'weights')
  assert(isfield(options, 'classes'));
  for i = 1:numel(options.weights)
    ll_str = sprintf('%s -w%d %g', ll_str, options.classes(i), options.weights(i));
  end
end

ll_str_nocv = ll_str;
if isfield(options, 'cv_folds') && ~isempty(options.cv_folds)
  ll_str = sprintf('%s -v %d', ll_str, options.cv_folds);
end

