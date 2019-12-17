function [rho,possible_solutions] = logprox(theta, gam, epsilon)
    if nargin<=2
        epsilon = eps;
    end
    sz = size(theta);
    theta = reshape(theta,1,[]);
    E = size(theta,2); %number of edges
    candidates = zeros(3,E);
    sgn = sign(theta);
    theta = abs(theta);
    %stick them in a vector, zero is also a possible solution candidates(1,:)
    candidates(2,:) = quadratic_solution(theta, 1,gam,epsilon);
    candidates(3,:) = quadratic_solution(theta,-1,gam,epsilon);
    
    %evaluate
    objective = 0.5*(candidates-theta).^2  + ...
                 gam.*log(epsilon+candidates);

    minim = objective;
    %Select zero if either quadratic solution is complex.
    %minim(2:3, complex_vals) = NaN; %min treats nan as infinity.
    [~,ix] = min(minim); %column min
%     ix
    ix = sub2ind(size(candidates), ix,1:E); %use linear indexing
    rho = candidates(ix);
    rho = sgn.*rho;
end

function sol = quadratic_solution(d,pm,gam,epsilon)

    discriminant = (d - epsilon).^2 - 4.*(gam - d.*epsilon);
    sol = zeros(size(d));
    mask = discriminant>=0;
    sol(mask) = 0.5 .* ( (d(mask) - epsilon) + ...
                            pm .* sqrt( discriminant(mask) ));
end