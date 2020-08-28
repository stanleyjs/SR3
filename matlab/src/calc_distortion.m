function distortion = calc_distortion(embedding,orig_metric)

if size(orig_metric,1) ~= size(orig_metric,2)
    orig_metric = squareform(pdist(orig_metric));
end

d_embed = squareform(pdist(embedding));
distortion = metric_similarity(orig_metric,d_embed);