function [loss,term1,term2] = objective_basic(x, U, phi, gam, objfun, epsilon)
    % objective to minimize
    u = tenmat(U,1:ndims(U));
    term1 = 1/2 * sum(double(x-u).^2); % distance between U and X
    term2 = 0;
    for l = 1:ndims(U)
        %the individual components of the objective for each mode.
        order = [1:ndims(U)];
        order(2) = l;
        order(l) = 2;
        dU = ttm(U,phi{l},l);
        vl = double(permute(dU,order));
        dists = vecnorm(vl);
        if strcmp(objfun, 'log')
            tmp_rho= sum(log(abs(dists)+epsilon));
        else
            tmp_rho = sum(snowflake(abs(dists),epsilon));
        end
        gam_part = gam(l) * tmp_rho;
        term2 = term2+gam_part;
    end
    loss = term1 + term2;
end

