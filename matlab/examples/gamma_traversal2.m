%% A DEMO OF HOW A SIMPLEX GAMMA TRAVERSAL IS COMPUTED
% OUTPUT DOCUMENTATION IN THE BOTTOM
close all; clear;
%try
%load lung500
%catch
load('/Users/mishne/Dropbox/Yale/data/coorg/lung100.mat')
%end
matrix = lung100;
gamma_vec = 2.^[-6:0.25:2];
lim_lower = -6;
lim_upper = 6;
kNN = 10;
row_labels = zeros(1,56);
row_labels(1:20) = 1;    % carcinoid
row_labels(21:33) = 2; % colon
row_labels(34:10) = 3;  % normal
row_labels(51:56) = 4; % small cell

%[matrix, gamma_vec, kNN, col_labels, row_labels] = load_data('checker');
x = matrix';
%x = x - mean(x(:));
%% SOME SR3 PARAMETERS
maxit = 100;
knn = [5,5];
clear SR3
SR3.params.tolF = 1e-6;
SR3.params.pcg_stop = false; %BUG? true is equivalent to maxit = 1!
%Right now pcg_stop = true stops the algorithm after the first PCG iteration.
% The intention was to stop when pcg hits some convergence setting.

SR3.params.maxiter = maxit; %the time spent for a single SR3_tensor (scale) calls
SR3.params.verbose = true;
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
[phi] = tensor_graph(x,knn);
[L,A] = tensor_incidence(phi,false);

SR3.params.epsilon = 1;

gammas = maxgamma(L,A,phi,x,SR3.nu,SR3.params.epsilon);

[SR3,gammas,ratios,magnitudes] = SR3_simplex2(x, phi,convexparams,SR3);

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
nP_r = zeros(length(vk),1);%Fusion Clustering Using Folded-Concave Penalties

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
    cc_rows = conncomp(G_r);
    G_c = graph(Lc);
    cc_cols = conncomp(G_c);
    nP_c(i) = max(cc_cols);
    nP_r(i) = max(cc_rows);
end
%%
[n_rows,n_cols] = size(x);
row_dist = zeros(n_rows);
col_dist = zeros(n_cols);
alpha = -0.5;
for i = 2:length(nP_c)
    gamma_c = SR3.output.gammas{i}(2);  
    gamma_r = SR3.output.gammas{i}(1);  
    if (nP_c(i) > 1 && nP_c(i) < n_cols && nP_c(i-1) ~= nP_c(i)) || ...
            (nP_r(i) > 1 && nP_r(i) < n_rows && nP_r(i-1) ~= nP_r(i))
        row_dist  = row_dist + ...
            (gamma_c*gamma_r).^(alpha) * squareform(pdist(x,'euclidean'));
        col_dist =  col_dist + ...;
            (gamma_c*gamma_r).^(alpha) * squareform(pdist(x','euclidean'));
        
    end
end
%%
eps     = median(row_dist(:));
aff_mat_row = exp(-row_dist.^2 / eps.^2);

eps     = median(col_dist(:));
aff_mat_col = exp(-col_dist.^2 / eps.^2);
1
[ vecs, vals ] = CalcEigs( aff_mat_row, 4 );
      proxfun: @flakeprox
embedding_rows = vecs*vals;
[ vecs, vals ] = CalcEigs( aff_mat_col, 4 );
embedding_cols = vecs*vals;
%%
figure;scatter3(embedding_rows(:,1),embedding_rows(:,2),embedding_rows(:,3),50,row_labels,'filled')
figure;scatter3(embedding_cols(:,1),embedding_cols(:,2),embedding_cols(:,3),50,1:n_cols,'filled')

%%
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


%%
figure;pause(3)
for i=1:100
    
    subplot(121);imagesc(double(vk{1, i}{1,2}  ));colorbar;
    title(sprintf('gamma_r=%1.2f, gamma_c = %1.2f',gammas(i,1),gammas(i,2)))
    subplot(122);imagesc(double(vk{1, i}{1,1}  ));colorbar;
    pause(0.5);
end