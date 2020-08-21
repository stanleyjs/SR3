function [gammas, ratios, magnitudes] = convexratios2(dims, max_gamma ,n_ratios, n_scales,minscale,maxscale)
    output = [];
    ratios = [];
    
    if dims > 2
        error('only works for 2d')
    end
    
    max_gamma(max_gamma>1) = ceil(max_gamma(max_gamma>1) );
    b = max_gamma(2);
    m = -(b/max_gamma(1));
    L = norm(max_gamma);
    
    ratio1 = linspace(1e-15,L-1e-15,n_ratios);
    for i = 1:n_ratios
        rem = ((1+1e-15)-ratio1(i)); % the remaining ratio after variable 1
        output = [output; (L-ratio1(i))*(max_gamma(1)/L) ratio1(i)*(max_gamma(2)/L)];
    end
%     for i = 1:size(output,1)
%         ratios = [ratios; perms(output(i,:))];
%     end
    tmpratios = unique(round(output,15),'rows');
    %tmpmagnitudes = linspace(minscale,maxscale, n_scales);
    tmpmagnitudes = 2.^linspace(minscale,0, n_scales);
    ratios = [];
    gammas = [];
    magnitudes = [];
    for i = 1:size(tmpratios,1)
        toappend = tmpratios(i,:).*tmpmagnitudes';
        ratios = [ratios; repmat(tmpratios(i,:),size(toappend,1),1)];
        magnitudes = [magnitudes; tmpmagnitudes'];
        gammas = [gammas; toappend];
    end
end