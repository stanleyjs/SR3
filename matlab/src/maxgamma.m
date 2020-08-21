function gamma = maxgamma(L,A,phi,x,nu,epsilon)

    mat = (L+nu*speye(size(L,1)));
    xvec =reshape(x,[],1);
    ustar = pcg(mat,nu*xvec,1e-8,500);
    
    gamma = zeros(1,ndims(x));
    
    for i = 1:ndims(x)
        uij = A{i}*ustar;
        stride = size(phi{i},1);
        otherdim = max(size(uij))/stride;
        uij = reshape(tensor(uij), [stride,otherdim]);
        temp = vecnorm(double(uij)');
        gamma(i) = (2*epsilon)/nu .* max(temp);
    end

end
%%

%% single way max gamma
% mat = (A{1}'*A{1}+nu*speye(size(L,1)));
% xvec =reshape(x,[],1);
% ustar = pcg(mat,nu*xvec,1e-8,500);
%     
% ustar = reshape(ustar,size(x));
% max((2*epsilon)/nu .*vecnorm((phi{1} * ustar)'))
% max((2*epsilon)/nu .*vecnorm((phi{2} * ustar')'))
% 
% 
% mat = (A{2}'*A{2}+nu*speye(size(L,1)));
% xvec =reshape(x,[],1);
% ustar = pcg(mat,nu*xvec,1e-8,500);
%     
% ustar = reshape(ustar,size(x));
% max((2*epsilon)/nu .*vecnorm((phi{1} * ustar)'))
% max((2*epsilon)/nu .*vecnorm((phi{2} * ustar')'))