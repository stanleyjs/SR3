function [rho] = flakeprox(theta, gam, epsilon)
    if nargin<=2
        epsilon = eps;
    end
    theta = reshape(theta,1,[]);
    sgn = sign(theta);
    theta = abs(theta);
    E = size(theta,2); %number of edges
    rho = zeros(1,E);
    p = epsilon;
    q = -1*theta;
    r = gam/2 - theta.*epsilon;
    candidates = zeros(4,E);
    
    candidates(2:end,:) = cubicrealroots(p,q,r,E);
    candidates = candidates.^2;
    [~,ix] = min(0.5*(candidates-theta).^2 + gam.*snowflake(candidates,epsilon));
    ix = sub2ind(size(candidates), ix,1:E); %use linear indexing
    rho = candidates(ix);
    rho = sgn.*rho;
end
function solutions = cubicrealroots(p,q,r,N)
%   solve for real roots of N cubic polynomials of real coefficients written as
%   0 = x^3 + p x^2 + q x^2 + r  
%   using a factorization of 
%   y^3 + ay + b = 0
%   where 
%   x = y-p/3
    solutions = zeros(3,N);
    a = (1/3).*(3.*q-p.^2);
    b = (1/27).*(2.*p.^3-9.*p.*q+27.*r);
    %each pair [a(i) b(i)] are the coefficients of the i-th polynomial in y
    inner_determinant = (b.^2)./4+(a.^3)./27;
    to_root = @(pm) ((-b./2) + pm.*(inner_determinant).^(1/2));
    A = to_root(1);
    A = sign(A).*abs(A.^(1/3));
    B = to_root(-1);
    B = sign(B).*abs(B.^(1/3));
    y_1 = A+B;
%     y_tmp1 = -0.5.*y_1;
%     y_tmp2 = (1i*sqrt(3))/2*(A-B);
%     y_2 = y_tmp1 + y_tmp2;
%     y_3 = y_tmp1 - y_tmp2;
    single_root = inner_determinant>0;
    solutions(1,single_root)= y_1(single_root);
    
    
    two_roots = inner_determinant==0;
    two_roots_bpos = two_roots & (b>0);
    two_roots_bneg = two_roots & (b<0);
    two_roots_b0 = two_roots & (b==0);
    assert(all(two_roots_bpos +two_roots_bneg +two_roots_b0 == two_roots))
    if any(two_roots_bpos)
        tmp_a = a(two_roots_bpos);
        solutions(:,two_roots_bpos) = [-2*sqrt(-tmp_a./3); sqrt(-tmp_a./3); sqrt(-tmp_a./3)];
    end
    if any(two_roots_bneg)
        tmp_a = a(two_roots_bneg);
        solutions(:,two_roots_bneg) = [2*sqrt(-tmp_a./3); -sqrt(-tmp_a./3); -sqrt(-tmp_a./3)];
    end
    if any(two_roots_b0)
        solutions(:,two_roots_b0) = [0 0 0];
    end
    
    
    three_roots = inner_determinant<0;
    if any(three_roots)
        tmp_a = a(three_roots);
        tmp_b = b(three_roots);
        phi = acos(-1.*sign(tmp_b).*sqrt(((tmp_b.^2)/4)./((-tmp_a.^3)./27)));
        solutions(1,three_roots) = 2.*sqrt(-tmp_a./3).*cos(phi./3+(2.*0.*pi)./3);
        solutions(2,three_roots) = 2.*sqrt(-tmp_a./3).*cos(phi./3+(2.*1.*pi)./3);
        solutions(3,three_roots) = 2.*sqrt(-tmp_a./3).*cos(phi./3+(2.*2.*pi)./3);
    end
    
    %solutions(2:3,~single_root) = [ y_2(~single_root); y_3(~single_root)];
    %elseif inner_determinant == 0
    %     if b == 0 
    %         candidates = 0;
    %     else
    %         candidates = sign(b)*[2*sqrt(-a/3), -sqrt(-a/3)];
    % 
    %     end
    % elseif inner_determinant<0
    %     phi = acos(sign(b)*sqrt((b^2/4)/(-a^3/27)));
    %     candidates = 2.*sqrt(-a/3).*cos(phi./3+(2*[0,1,2]*pi)./3);
    % end
    solutions = solutions - p/3;
end