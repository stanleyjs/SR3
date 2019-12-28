%% A DEMO OF HOW A SIMPLEX GAMMA TRAVERSAL IS COMPUTED
% OUTPUT DOCUMENTATION IN THE BOTTOM
close all; clear all;
try
load lung100
catch
load('/Users/mishne/Dropbox/Yale/data/coorg/lung100.mat')
end
matrix = lung100' / 6;
gamma_vec = 2.^[-6:0.25:2];
lim_lower = -6;
lim_upper = 6;
kNN = 10;
row_labels = zeros(1,56);
row_labels(1:20) = 1;    % carcinoid
row_labels(21:33) = 2; % colon
row_labels(34:10) = 3;  % normal
row_labels(51:56) = 4; % small cell
x = matrix;

%% SOME SR3 PARAMETERS
maxit = 100;
knn = [10,10];
clear SR3
SR3.params.tolF = 1e-4;
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
convexparams.min_mag = 1; % minimum magnitudes
convexparams.max_mag = 1e8; % maximum magnitude.  You want this to be proportional to nu.
%params for regular SR3
% SR3.min_gamma = 1e-4;

SR3.nu = 1e-6;
[phi] = tensor_graph(x,knn);

[SR3,gammas,ratios,magnitudes] = SR3_simplex(x, phi,convexparams,SR3);

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
% teh remaining columns are the total mode-wise losses.
