function distortion = metric_similarity(dX,dY)

expansion = max(dY(:)./dX(:));
distortion = max(dX(:)./dY(:));

distortion = expansion * distortion;

return
%%%%%%%%
