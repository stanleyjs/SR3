clear SR3;
N = 300;
M = 10;
sigma = 1;
mu = 10;
maxiter=100;
nu = 1;
gamma = 10000;
Pr = 2;
Pc = 4;
[x,phi] = blockdiag_example('N',N, 'M', M,'Pr',Pr,'Pc',Pc,'sigma', sigma,'mu',mu);
x = x-mean(mean(x));
SR3.params.maxiter = maxiter;
SR3.params.verbose = true;
SR3.params.minstep = 2;
SR3.params.epsilon = 1e-6;
SR3.params.store_updates = false;
SR3.params.paths.rann = './RANN';
SR3.params.paths.tensor_toolbox = './tensor_toolbox';
SR3.solver.f = 'legendre';
SR3.gamma=[100 0.25]; %%full column fusion!

SR3.nu = 1e-6;
SR3.solver.params.tol = 1e-6;

grph.paths.rann = SR3.params.paths.rann;
grph.paths.tensor_toolbox = SR3.params.paths.tensor_toolbox;


[SR3] = SR3_tensor(x, phi,SR3);
[L,A,lmin,lmax] = tensor_incidence(phi, true);