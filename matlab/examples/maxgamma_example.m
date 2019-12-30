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

%% TO COMPUTE A MAXGAMMA YOU NEED A GRAPH LAPLACIAN, AN INCIDENCE MATRIX, AN X, AN EPSILON, AND A NU.
knn = [10,10];
nu = 1e-6;
epsilon = 1e-8;
[phi] = tensor_graph(x,knn);
[L,A] = tensor_incidence(phi,false);

gammas = maxgamma(L,A,phi,x,nu,epsilon);
%%
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
SR3.solver.f = 'pcg_preconditioned';
% IDENTITY
SR3.gamma = gammas;
SR3.nu = 1e-6;
SR3.epsilon = 1e-8;

[phi] = tensor_graph(x,knn);
[SR3] = SR3_tensor(x, phi,SR3);
