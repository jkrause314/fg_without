function run_recognition(config)
% Do the actual recognition part of the method:
% -Train SVMs on fine-tuned and non-fine-tuned features
% -Train the discriminative combination of parts
% -Evaluate on the test set.

% Extract non-fine-tuned features around the parts (train only)
extract_part_feats(config, 'train', 'vgg');
% Extract fine-tuned features around the parts (train and test)
extract_part_feats(config, 'train', 'vgg-ft');
extract_part_feats(config, 'test', 'vgg-ft');

% TODO
%% Train cross-validated SVMs on the non-fine-tuned features
%train_part_svms(config, 'vgg', config.svm_folds)
%
%% Train SVMs on the fine-tuned features
%train_part_svms(config, 'vgg-ft', config.svm_folds)

% Learn the discriminative combination of parts
% Apply the DCoP using the fine-tuned features
