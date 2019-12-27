function [coeffs,error] = fit_legendre(f,order,lower, upper)
% construct the Legendre discretization of the interval [lower,upper]
    [xs,wts] = lgwt(order,-1,1);
    wts = flipud(wts);
    xs = flipud(xs);
    ts=(lower.*(1-xs)+upper.*(1+xs))./2;      
    fs = f(ts);

    [u] = legepols(xs',order);
    v = u';
    for i = 1:order
        d=1;
        d=d*(2*i-1)/2;
        for j = 1:order
            u(i,j)=v(j,i)*wts(j)*d;
        end
    end
    coeffs = (u*fs);
%     coeffs(abs(coeffs)<1e-9) = 0;
    error = 10^floor(log10(abs(coeffs(end))));
end