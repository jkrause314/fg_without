clear all;

domain = 'cub';
config = set_config(domain)

make_cub_anno_files(config);
run_coseg(config);

% Do these if you want to evaluate cub segmentation.
eval_seg = false;
if eval_seg
  make_cub_seg_files(config);
  check_cub_coseg(config);
end

% STOPPED RIGHT HERE.
extract_feats_bbox_domain(config);
make_msts(config);

run_matching(config);
part_propagate(config);
tighten_parts(config);
