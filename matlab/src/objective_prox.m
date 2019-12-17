function [loss,term1,term2,term3,mode_loss] = objective_prox(x, U, V, A, nu, gam, objfun,epsilon)
    % objective to minimize
    u = tenmat(U,1:ndims(U));
    term1 = 1/2 * sum(double(x-u).^2); % distance between U and X
    term2 = 0;
    term3 = 0;
    mode_loss = {};
    for l = 1:numel(A)

        %the individual components of the objective for each mode.
        order = [1:numel(A)];
        order(2) = l;
        order(l) = 2;
        vl = double(permute(V{l},order));
        dists = vecnorm(vl);
        if strcmp(objfun, 'log')
            tmp_rho= sum(log(abs(dists)+epsilon));
        elseif strcmp(objfun,'l2')
            tmp_rho = dists;
        else
            tmp_rho = sum(snowflake(abs(dists),1));
        end
        prox_shrinkage = gam(l) * tmp_rho;
        if isnan(prox_shrinkage)
            prox_shrinkage = 0;
        end
        term2 = term2+prox_shrinkage;
        Au =  A{l}*double(u);
        v = tenmat(V{l},1:ndims(V{l}));
        prox_error = (1/(2*nu))*(norm(double(v)-Au)^2);
        if isnan(prox_error)
            prox_error = 0;
        end
        term3 = term3+prox_error;
        mode_loss{l} = prox_error+prox_shrinkage;
    end
    loss = term1+term2+term3;
end