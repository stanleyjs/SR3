function [multiscale_Y, multiscale_distances, multiscale_tensor] = multiscale_data(Y, gamma, metricfun, missing_data_mask, X)
    if nargin==4
        if missing_data_mask == true
            warning('No X supplied with mask.   No fill will be used');
        end
        missing_data_mask = false;
    end
    if nargin < 4
        missing_data_mask = false;
    end
    if nargin < 3
        warning('No metricfun supplied.  Using @(x) sqrt(prod(x))');
        metricfun = @(x) sqrt(prod(x));
    end
    numscales = numel(Y);
    dims=size(Y{1});
    order = length(dims);
    for ell = 1:order
        dl = dims(ell);
        multiscale_distances{ell} = zeros(dl);
        multiscale_Y{ell} = zeros(dl, prod(dims)/dl);
        if nargout == 3
            multiscale_tensor{ell} = tenzeros([dl, prod(dims)/dl,numscales]);
        end
    end
    for j = 1:numscales
        Yj = Y{j};
        scale = metricfun(gamma{j}) ;

        if missing_data_mask
            embedding = tenmat(Yj,1:ndims(Y));
            embedding = remap_tensor(~missing_data_mask.*embedding.data(1:end),embedding);
            Yj = tenmat(X,1:ndims(Y)) + tenmat(embedding,1:ndims(Y));
            Yj = tensor(Yj);
        end
        for ell = 1:order
            matrix = tenmat(Yj,ell);
            matrix = double(matrix);
            multiscale_Y{ell} = multiscale_Y{ell} + scale.*matrix;
            multiscale_distances{ell} = multiscale_distances{ell} + scale.*squareform(pdist(matrix));
            if nargout == 3
                multiscale_tensor{ell}(:,:,j) = scale.*matrix;
            end
        end
    end
end