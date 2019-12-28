function [gammas, ratios, magnitudes] = convexratios(dims,n_ratios, n_scales,minscale,maxscale)
    output = [];
    ratios = [];
    ratio1 = linspace(0,1,n_ratios);
    for i = 1:n_ratios
        rem = ((1+1e-15)-ratio1(i)); % the remaining ratio after variable 1
        gate = ratio1<rem;
        if dims == 3
            rem2 = rem-(ratio1(gate)); % 
            output = [output; repmat(ratio1(i),numel(rem2),1),ratio1(gate)',rem2'];
        else
            output = [output; ratio1(i) rem];
        end
    end
%     for i = 1:size(output,1)
%         ratios = [ratios; perms(output(i,:))];
%     end
    tmpratios = unique(round(output,15),'rows');
    tmpmagnitudes = linspace(minscale,maxscale, n_scales);
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