clear all;

domain = 'car';
config = set_config(domain)

make_car_anno_files(config);
run_coseg(config);

extract_feats_bbox_domain(config);
make_msts(config);

run_matching(config);
part_propagate(config);
tighten_parts(config);

%my_run_rcnn(config);

vgg_finetune(config);
