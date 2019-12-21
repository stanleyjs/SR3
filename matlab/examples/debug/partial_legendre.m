clear all;
N = 300;
M = 10;
sigma = 1;
mu = 10;
maxiter=1000;
Pr = 2;
Pc = 4;
[X,phi] = blockdiag_example('N',N, 'M', M,'Pr',Pr,'Pc',Pc,'sigma', sigma,'mu',mu);
X = X-mean(mean(X));
[L,A] = tensor_incidence(phi,false);
[U,S,~] = svds(L,50,'smallest');
lmax = svds(L,10,'largest');
lmax = lmax(1);
x = reshape(X,[],1);
%%

%%
order = 30; %add to parameters ultimately.
lower = 0.1;%10^(floor(log10(lmin))-2);
upper = lmax;%10^ceil(log10(lmax)+2);
[coeffs,error] = fit_legendre(@(x) 1./(1e-6+x),order, lower, upper);
error
mat = ((2*L)./lmax)-speye(size(L,1));
solve = @(b,b0) matvecleg(mat, b, order, coeffs);
%%
xproj = U'*x;
xorth = x-(U*xproj);
Htaps = 1./(1e-6+diag(S));
Htaps_xproj = xproj.*Htaps;

xorth_approx = solve(xorth,0);
y = (U*Htaps_xproj)+xorth_approx;

%%
[U_full,S_full,~] = svds(L,3000);

%%
v = cell(2,1);
sumV = 0;
outdists = [];
for c = 1:2
    uij = A{c}*x;
    v{c} = uij;
    stride = size(phi{c},1);
    otherdim = max(size(uij))/stride;

    uij = reshape(tensor(uij), [stride,otherdim]);
    temp = vecnorm(double(uij)');
    temp = flakeprox(temp,1,1e-8)./temp;

    v{c}(1:end) = v{c}(1:end).*repmat(temp,1,otherdim)';
    sumV = sumV+ A{c}'*v{c};
end
V = v;

%%
y = pcg((1e-6*speye(size(L))+L), 1e-6*x+sumV, 1e-8,10000);
%%
Htaps_full = 1./(1e-6+diag(S_full));
xproj_full = U_full'*(1e-6*x+sumV);
yhat_full = Htaps_full.*xproj_full;
y_full = U_full*yhat_full;
y_full = reshape(y_full,size(X));
%%
xnu = 1e-6+sumV;
xproj = U'*xnu;
xorth = xnu-(U*xproj);
Htaps = 1./(1e-6+diag(S));
Htaps_xproj = xproj.*Htaps;

xorth_approx = solve(xorth,0);
y = (U*Htaps_xproj)+xorth_approx;
%%
xnu = 1e-6+sumV;

[psi,lambda] = prodgraph_eigs(phi,100);
[lambda,ix] = sort(lambda);
psi = psi(:,ix);
xproj = psi'*xnu;
xorth = xnu-(psi*xproj);
Htapsnu = 1./(1e-6+((lambda)));
Htaps_xproj = xproj.*Htapsnu;

xorth_approx = solve(xorth,0);
y = (psi*Htaps_xproj)+xorth_approx;

imagesc(reshape(y,size(X)))