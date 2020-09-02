function [embedding_rows, embedding_cols, gammas_sr3, emd_inds,uk, vk ] = ...
            gamma_traversal_single(x,x_orig,mask,gamma_init,knn,params)
        
[n_rows,n_cols,~] = size(x);

%% SOME SR3 PARAMETERS
maxit = 100;
if length(knn) < ndims(x)
knn = knn * ones(1,ndims(x));
end

SR3.params.tolF = 1e-6;
SR3.params.pcg_stop = false; %BUG? true is equivalent to maxit = 1!
%Right now pcg_stop = true stops the algorithm after the first PCG iteration.
% The intention was to stop when pcg hits some convergence setting.

SR3.params.maxiter = maxit; %the time spent for a single SR3_tensor (scale) calls
SR3.params.verbose = false;
SR3.params.epsilon = 1e-8;
SR3.params.store_updates = false; %STORE THE WHOLE GAMMA PATH - not recommended
% we generate gammas over the simplex using ratio_between_modes(i).*magnitudes
convexparams.Nratios = 10; % Number of ratios to traverse
convexparams.Nmagnitudes = 10; % magnitudes to take
convexparams.min_mag = -3; % minimum magnitudes
convexparams.max_mag = 2; % maximum magnitude.  You want this to be proportional to nu.
%params for regular SR3
% SR3.min_gamma = 1e-4;
SR3.nu = 1e-6;


if params.oracleWeights
   % calculate initial row and column graphs
    [phi] = tensor_graph(x_orig,knn);
else
    % calculate initial row and column graphs
    [phi] = tensor_graph(x,knn);
end

% using oracle graph
[L,A] = tensor_incidence(phi,false);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SR3.missing_data = mask(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SR3.params.epsilon = 1;
%%
% convexparams.max_gamma = maxgamma(L,A,phi,x,SR3.nu,SR3.params.epsilon);
% convexparams.max_gamma = min(convexparams.max_gamma) * ones(size(convexparams.max_gamma));
[convexparams.max_gamma, convexparams.min_gamma] = maxgamma2(x,phi,SR3,gamma_init);
convexparams.min_mag = floor(log2(min(convexparams.min_gamma))) - floor(log2(max(convexparams.max_gamma)));
%convexparams.min_mag = min(convexparams.min_gamma);
%convexparams.max_mag = 1;


%%
tic
[SR3,gammas,ratios,magnitudes] = SR3_simplex2(x, phi,convexparams,SR3);
toc
%% The scales are stored ratio-wise in SR3{}
% for i=1:convexparams.Nratios
% SR3{i} (where i is the index of the scale) has fields
% SR3{i}.ratio -> the ratio that was used to generate this sequence
% SR3{i}.magnitudes -> the magnitudes that are multiplied by that ratio
% SR3{i}.gammas -> the ACTUAL gammas that were used from taking the
% ratio.*magnitudes

% The j-th entry of the following fields corresonds to the gamma of
% SR3{i}.ratio.*SR3{i}.magnitudes{j} = SR3{i}.gammas{j}

% SR3{i}.U ->a cell of U tensors.see below
% SR3{i}.V -> cell of V tensors. see below
% SR3{i}.F -> cell of objective function matrices. see below
% SR3{i}.SR3 -> A cell of SR3-tensor output structs for each scale.
% SR3{i}.iter ->the number of iterations required for each SR3_tensor to
% "converge"

%% AN individual SR3{i}.SR3{j} run is stored in SR3
% here i corresponds to the ratio and j corresponds to the magnitude
% For j=1:convexparams.Nmagnitudes
% Params are kept in SR3

% SR3.output.U is the resulting tensor from the final optimization step
% You will need to call double() on it to make it a matrix.

% SR3.output.V is a cell of the shrunken difference tensors at the last output.
% Again you must call double() to make it a matrix for imagesc and other
% applicaitons.


% SR3.output.F will store your objective function results for each
% iteration (i.e. call to PCG)
% Column 1 is the total loss
% column 2 is the loss due to l2 error between X and U
% column 3 is the loss due to the sum of snowflake of V
% column 4 is the loss due to l2 error between U's differences and the
% elements of V
% the remaining columns are the total mode-wise losses.
%%
uk = [SR3.U];
vk = [SR3.V];
gammas_sr3 =[SR3.gammas];
gammas_sr3 = [gammas_sr3(:,1:2:end), gammas_sr3(:,2:2:end)];
gammas_sr3 =reshape(gammas_sr3,[],2);
%%
nP_c = zeros(length(vk),1);
nP_r = zeros(length(vk),1); % Fusion Clustering Using Folded-Concave Penalties
cc_cols = zeros(length(vk),size(x,2));
cc_rows = zeros(length(vk),size(x,1));

for i =1:length(vk)
    
%     mc=(bsxfun(@times,~vecnorm(double(vk{1,i}{1,2}),2,1),double(phi{1, 1})));
%     Lc = mc*mc';
%     
%     mr=(bsxfun(@times,~vecnorm(double(vk{1,i}{1,1}),2,1),double(phi{2, 1})));
%     Lr = mr*mr';
    
    mc=(bsxfun(@times,~vecnorm(double(vk{1,i}{1,2}),2,2),double(phi{2, 1})));
    Lc = mc'*mc;
    
    mr=(bsxfun(@times,~vecnorm(double(vk{1,i}{1,1}),2,2),double(phi{1, 1})));
    Lr = mr'*mr;
    
    G_r = graph(Lr);
    cc_rows(i,:) = conncomp(G_r);
    G_c = graph(Lc);
    cc_cols(i,:) = conncomp(G_c);
end

nP_c = max(cc_cols,[],2);
nP_r = max(cc_rows,[],2);

%%

% only run over unique clustering assignments for metric calculation 
[~,row_inds,~] = unique(cc_rows,'rows');
[~,col_inds,~] = unique(cc_cols,'rows');

emd_inds = union(row_inds,col_inds);

row_dist = zeros(n_rows);
col_dist = zeros(n_cols);
alpha = -0.5;
for i = emd_inds'
    gamma_c = gammas_sr3(i,2);
    gamma_r = gammas_sr3(i,1);  
    
%     if gamma_r==0 || gamma_c ==0
%         continue
%     end
    
%    if (nP_c(i) > 1 && nP_c(i) < n_cols && nP_c(i-1) ~= nP_c(i)) || ...
 %           (nP_r(i) > 1 && nP_r(i) < n_rows && nP_r(i-1) ~= nP_r(i))
    if ((nP_c(i) > 1 && nP_c(i) < n_cols) || (nP_r(i) > 1 && nP_r(i) < n_rows))     
        %x_smooth = calculate_averaging_matrix(x, [nP_r(i), nP_c(i)], {cc_rows(i,:), cc_cols(i,:)});
        %SR3.missing_data
        
        % use U to initialize x_tilde
        x_smooth = double(uk{i});
        % fill x_tilde with orig vals in non-missing entries
        x_smooth(mask(:)) = x(mask(:));
                
        figure;
        subplot(131);imagesc(x_smooth);
        %axis image;
        title(sprintf('n_r =%d, n_c=%d',nP_r(i) ,nP_c(i) ));
        colorbar
        subplot(132);imagesc( double(uk{i}));
        %axis image;
        title(sprintf('n_r =%d, n_c=%d',nP_r(i) ,nP_c(i) ));
        colorbar
        subplot(133);imagesc(abs(x_smooth-x));
        colorbar

        drawnow
        row_dist  = row_dist + ...
            (gamma_c+gamma_r).^(alpha) * squareform(pdist(x_smooth,'euclidean'));
        col_dist =  col_dist + ...;
            (gamma_c+gamma_r).^(alpha) * squareform(pdist(x_smooth','euclidean'));
        
    end
end
%%
eps     = median(row_dist(:));
aff_mat_row = exp(-row_dist.^2 / eps.^2);

eps     = median(col_dist(:));
aff_mat_col = exp(-col_dist.^2 / eps.^2);

% [ vecs, vals ] = CalcEigs( aff_mat_row, 4 );
% embedding_rows = vecs*vals;
% [ vecs, vals ] = CalcEigs( aff_mat_col, 4 );
% embedding_cols = vecs*vals;

[ vecs, vals ] = CalcEigs( aff_mat_row, params.nEigs );
embedding_rows = vecs*vals;
[ vecs, vals ] = CalcEigs( aff_mat_col, params.nEigs );
embedding_cols = vecs*vals;

%%
figure;
subplot(221);imagesc(aff_mat_row);axis image
subplot(222);imagesc(aff_mat_col);axis image
subplot(223)
scatter3(embedding_rows(:,1),embedding_rows(:,2),embedding_rows(:,3),50,1:n_rows,'filled')
title('Embedding rows')
subplot(224)
scatter3(embedding_cols(:,1),embedding_cols(:,2),embedding_cols(:,3),50,1:n_cols,'filled')
title('Embedding cols')

saveas(gcf,sprintf('sr3_inpaint_%s_missing_%d_embed.png',params.dataset,params.p))

return
%%
figure;
subplot(121);scatter(gammas(:,1),gammas(:,2),50,nP_r,'filled')
subplot(122);scatter(gammas(:,1),gammas(:,2),50,nP_c,'filled')
colormap jet
clsuters=[gammas, nP_r, nP_c];
%%
return
%%
figure
clear nuk
t = double(uk{end});
t = t(1:end-1,:);
dists = pdist(t');
tree = linkage(t','average');
leafOrder = optimalleaforder(tree,dists);

for i=1:1:numel(uk)
    t = double(uk{i});
    t = t(1:end-1,:);
    t = t(:,leafOrder);
    nuk{i} = t;
end
filename = './lung_fixed.gif';
f = cell2imgif(nuk,filename, false, 0.1, 1,true,false);

return;
%%
figure;pause(3)
for i=1:100
    
    subplot(121);imagesc(double(vk{1, i}{1,2}  ));colorbar;
    title(sprintf('gamma_r=%1.2f, gamma_c = %1.2f',gammas(i,1),gammas(i,2)))
    subplot(122);imagesc(double(vk{1, i}{1,1}  ));colorbar;
    pause(0.5);
end