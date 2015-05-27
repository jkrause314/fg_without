function warp_params = compute_warp_params(im, other_ims, options)

if ~exist('options', 'var')
  options = [];
end
if ~isfield(options, 'try_flip')
  options.try_flip = false;
end

assert(iscell(other_ims));

num_ims = numel(other_ims);
warp_params = cell(1, num_ims);
parfor i = 1:num_ims
  %fprintf('sc compute %d/%d\n', i, num_ims);
  i1 = im;
  i2 = other_ims{i};

  flips = [false];
  if options.try_flip
    flips = [false true];
  end
  flip_sc_costs = zeros(size(flips));
  flip_aff_costs = zeros(size(flips));
  flip_warp_params = cell(size(flips));
  for flip_ind = 1:numel(flips)
    i2_used = i2;
    if flips(flip_ind)
      i2_used = flipdim(i2, 2);
    end
    s1 = ~all(i1==0,3);
    s2 = ~all(i2_used==0,3);

%    if nnz(s1) < 350 || nnz(s2) < 350
%      continue;
%    end

    %%%%%%%%%%%%%%%%%%%%%%%%% START SHAPE CONTEXT
    affine_start_flag=1;
    polarity_flag=1;
    nsamp=100;
    eps_dum=0.25;
    ndum_frac=0.25;        
    mean_dist_global=[];
    ori_weight=0.1;
    nbins_theta=12;
    nbins_r=5;
    r_inner=1/8;
    r_outer=2;
    tan_eps=1.0;
    n_iter=6;
    beta_init=1;
    r=1; % annealing rate
    w=4;
    sf=2.5;

    V1=double(s1);
    V2=double(s2);
    [N1,N2]=size(V1);

    %%%
    %%% edge detection
    %%%
    [x2,y2,t2]=bdry_extract_3(V2);
    nsamp2=length(x2);
    if nsamp2>=nsamp
      [x2,y2,t2]=get_samples_1(x2,y2,t2,nsamp);
    else
      fprintf('shape #2 doesn''t have enough samples, not doing alignment');
      continue;
    end
    Y=[x2 y2];

    % get boundary points
    [x1,y1,t1]=bdry_extract_3(V1);

    nsamp1=length(x1);
    if nsamp1>=nsamp
      [x1,y1,t1]=get_samples_1(x1,y1,t1,nsamp);
    else
      fprintf('shape #1 doesn''t have enough samples, not doing alignment');
      continue;
    end
    X=[x1 y1];

    %%%
    %%% compute correspondences
    %%%
    Xk=X;
    tk=t1;
    k=1;
    s=1;
    ndum=round(ndum_frac*nsamp);
    out_vec_1=zeros(1,nsamp);
    out_vec_2=zeros(1,nsamp);
    while s
      [BH1,mean_dist_1]=sc_compute(Xk',zeros(1,nsamp),mean_dist_global,nbins_theta,nbins_r,r_inner,r_outer,out_vec_1);
      [BH2,mean_dist_2]=sc_compute(Y',zeros(1,nsamp),mean_dist_global,nbins_theta,nbins_r,r_inner,r_outer,out_vec_2);

      if affine_start_flag
        if k==1
          % use huge regularization to get affine behavior
          lambda_o=1000;
        else
          lambda_o=beta_init*r^(k-2);	 
        end
      else
        lambda_o=beta_init*r^(k-1);
      end
      beta_k=(mean_dist_2^2)*lambda_o;
      costmat_shape=hist_cost_2(BH1,BH2);
      theta_diff=repmat(tk,1,nsamp)-repmat(t2',nsamp,1);
      if polarity_flag
        % use edge polarity
        costmat_theta=0.5*(1-cos(theta_diff));
      else
        % ignore edge polarity
        costmat_theta=0.5*(1-cos(2*theta_diff));
      end      
      costmat=(1-ori_weight)*costmat_shape+ori_weight*costmat_theta;
      nptsd=nsamp+ndum;
      costmat2=eps_dum*ones(nptsd,nptsd);
      costmat2(1:nsamp,1:nsamp)=costmat;
      
      %cvec=hungarian(costmat2);

      %cvec=munkres(costmat2);
      %cvec(cvec) = 1:numel(cvec);
  
      %cvec=lapjv(costmat2);
      %cvec(cvec) = 1:numel(cvec);

      % This is much faster
      cvec=assignmentoptimal(costmat2)';
      cvec(cvec) = 1:numel(cvec);

      % update outlier indicator vectors
      [a,cvec2]=sort(cvec);
      out_vec_1=cvec2(1:nsamp)>nsamp;
      out_vec_2=cvec(1:nsamp)>nsamp;
   
      X2=NaN*ones(nptsd,2);
      X2(1:nsamp,:)=Xk;
      X2=X2(cvec,:);
      X2b=NaN*ones(nptsd,2);
      X2b(1:nsamp,:)=X;
      X2b=X2b(cvec,:);
      Y2=NaN*ones(nptsd,2);
      Y2(1:nsamp,:)=Y;
   
      % extract coordinates of non-dummy correspondences and use them
      % to estimate transformation
      ind_good=find(~isnan(X2b(1:nsamp,1)));
      n_good=length(ind_good);
      X3b=X2b(ind_good,:);
      Y3=Y2(ind_good,:);
   
      [cx,cy,E]=bookstein(X3b,Y3,beta_k);

      % calculate affine cost
      A=[cx(n_good+2:n_good+3,:) cy(n_good+2:n_good+3,:)];
      s=svd(A);
      aff_cost=log(s(1)/s(2));
     
      % calculate shape context cost
      [a1,b1]=min(costmat,[],1);
      [a2,b2]=min(costmat,[],2);
      sc_cost=max(mean(a1),mean(a2));
     
      % warp each coordinate
      fx_aff=cx(n_good+1:n_good+3)'*[ones(1,nsamp); X'];
      d2=max(dist2(X3b,X),0);
      U=d2.*log(d2+eps);
      fx_wrp=cx(1:n_good)'*U;
      fx=fx_aff+fx_wrp;
      fy_aff=cy(n_good+1:n_good+3)'*[ones(1,nsamp); X'];
      fy_wrp=cy(1:n_good)'*U;
      fy=fy_aff+fy_wrp;

      Z=[fx; fy]';

      % apply the warp to the tangent vectors to get the new angles
      Xtan=X+tan_eps*[cos(t1) sin(t1)];
      fx_aff=cx(n_good+1:n_good+3)'*[ones(1,nsamp); Xtan'];
      d2=max(dist2(X3b,Xtan),0);
      U=d2.*log(d2+eps);
      fx_wrp=cx(1:n_good)'*U;
      fx=fx_aff+fx_wrp;
      fy_aff=cy(n_good+1:n_good+3)'*[ones(1,nsamp); Xtan'];
      fy_wrp=cy(1:n_good)'*U;
      fy=fy_aff+fy_wrp;
      
      Ztan=[fx; fy]';
      tk=atan2(Ztan(:,2)-Z(:,2),Ztan(:,1)-Z(:,1));

      % update Xk for the next iteration
      Xk=Z;
      if k==n_iter
        s=0;
      else
        k=k+1;
      end
    end
    % Save cx,cy,X3b
    flip_sc_costs(flip_ind) = sc_cost;
    flip_aff_costs(flip_ind) = aff_cost;
    flip_warp_params{flip_ind} = struct('cx', cx, 'cy', cy, 'X3b', X3b, 'n_good', n_good, 'flipped', flips(flip_ind));
  end
  %if numel(flips) == 2 && flip_aff_costs(2)<flip_aff_costs(1) && flip_sc_costs(2)<flip_sc_costs(1)
  if numel(flips) == 2 && flip_aff_costs(2)<flip_aff_costs(1)% && flip_sc_costs(2)<flip_sc_costs(1)
    warp_params{i} = flip_warp_params{2};
  else
    warp_params{i} = flip_warp_params{1};
  end
  %warp_params{i} = struct('cx', cx, 'cy', cy, 'X3b', X3b, 'n_good', n_good);
end
