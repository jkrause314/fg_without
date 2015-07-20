function [rec,prec,ap] = myevaldet(pred_boxes, gt_boxes)
% Assumes pred_boxes and gt_boxes are in the same order

minoverlap = .5;

% load ground truth objects
tic;
npos=0;
gt = repmat(struct('BB', [], 'det', []), size(gt_boxes));

for i=1:numel(gt)
  gt(i).BB = reshape(gt_boxes{i}, 4, []);
  gt(i).det = false;
  npos = npos + 1;
end

num_pred_boxes = sum(cellfun(@(x)size(x,1), pred_boxes));
all_pred_boxes = zeros(num_pred_boxes, size(pred_boxes{1}, 2));
ids = zeros(num_pred_boxes, 1);
ind = 1;
for i = 1:numel(pred_boxes)
  all_pred_boxes(ind:ind+size(pred_boxes{i}, 1)-1,:) = pred_boxes{i};
  ids(ind:ind+size(pred_boxes{i}, 1)-1) = i;
  ind = ind + size(pred_boxes{i}, 1);
end
BB = all_pred_boxes(:, 1:4)';
confidence = all_pred_boxes(:,end);

% sort detections by decreasing confidence
[sc,si]=sort(-confidence);
ids=ids(si);
BB=BB(:,si);

% assign detections to ground truth objects
nd=length(confidence);
tp=zeros(nd,1);
fp=zeros(nd,1);
tic;
for d=1:nd
  % display progress
  if toc>1
    fprintf('pr: compute: %d/%d\n',d,nd);
    drawnow;
    tic;
  end
 
  % find ground truth image
  i = ids(d);

  % assign detection to ground truth object if any
  bb=BB(:,d);
  ovmax=-inf;
  for j=1:size(gt(i).BB,2)
    bbgt=gt(i).BB(:,j);
    bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
    iw=bi(3)-bi(1)+1;
    ih=bi(4)-bi(2)+1;
    if iw>0 & ih>0        
      % compute overlap as area of intersection / area of union
      ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
         (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
         iw*ih;
      ov=iw*ih/ua;
      if ov>ovmax
        ovmax=ov;
        jmax=j;
      end
    end
  end
  % assign detection as true positive/don't care/false positive
  if ovmax>=minoverlap
    if ~gt(i).det(jmax)
      tp(d)=1;      % true positive
  gt(i).det(jmax)=true;
    else
      fp(d)=1;      % false positive (multiple detection)
    end
  else
    fp(d)=1;          % false positive
  end
end

% compute precision/recall
fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);

% compute average precision

ap=0;
for t=0:0.1:1
  p=max(prec(rec>=t));
  if isempty(p)
    p=0;
  end
  ap=ap+p/11;
end
