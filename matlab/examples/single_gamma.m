%% A DEMO OF HOW A SINGLE GAMMA MIGHT BE COMPUTED
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
maxit = 10;
knn = [10,10];
clear SR3
SR3.params.tolF = 1e-15;
SR3.params.pcg_stop = false; %BUG?
%Right now pcg_stop=true stops the algorithm after the first PCG iteration.
% The intention was to stop when pcg hits some convergence setting.
SR3.params.maxiter = maxit;
SR3.params.verbose = true;
SR3.params.epsilon = 1e-8;
SR3.params.store_updates = false; %STORE THE WHOLE GAMMA PATH
rho_log = {@log, @logprox}; 
rho_flake = {@snowflake, @flakeprox};
rho_l2 = {@vecnorm, @l2_prox};
SR3.solver.f = 'legendre';
% IDENTITY
SR3.gamma = [0 0];
SR3.nu = 1e-6;


[phi] = tensor_graph(x,knn);
[SR3] = SR3_tensor(x, phi,SR3);

%% Single MODE
SR3.gamma = [1 0];
[SR3] = SR3_tensor(x, phi,SR3);

%% Two MODE
SR3.gamma = [0.3 0.7];
[SR3] = SR3_tensor(x, phi,SR3);


%% AN individual output of SR3 is stored in the resulting SR3.output

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

