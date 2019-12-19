clear SR3;
N = 300;
M = 10;
sigma = 1;
mu = 10;
maxiter=1000;
Pr = 2;
Pc = 4;
[x,phi] = blockdiag_example('N',N, 'M', M,'Pr',Pr,'Pc',Pc,'sigma', sigma,'mu',mu);
x = x-mean(mean(x));
SR3.params.maxiter = maxiter;
SR3.params.verbose = true;
SR3.params.minstep = 2;
SR3.params.epsilon = 1e-12;
SR3.params.store_updates = false;
SR3.params.paths.rann = './RANN';
SR3.params.paths.tensor_toolbox = './tensor_toolbox';
SR3.solver.f = 'pcg_preconditioned';
%params for regular SR3
SR3.gamma=[0 1];
% SR3.min_gamma = 1e-4;

SR3.nu = 1e-6;
SR3.solver.params.tol = 1e-6;

grph.paths.rann = SR3.params.paths.rann;
grph.paths.tensor_toolbox = SR3.params.paths.tensor_toolbox;

%[phi] = tensor_graph(x,[5 5], grph);
% phit{1} = zeros(size(phit{1}));
% phit{2} = phi;
[SR3] = SR3_tensor(x, phi,SR3);