%%
dataset = 'lung500';
%%
n_rand = 10;
pvals      = [10:10:90];
%%
if ~exist('AR_DM','var')
AR_DM           = zeros(length(pvals),n_rand);
RI_DM           = zeros(length(pvals),n_rand);
AR_SR3          = zeros(length(pvals),n_rand);
RI_SR3          = zeros(length(pvals),n_rand);

distortion_rows_orig   = zeros(length(pvals),n_rand);
distortion_cols_orig   = zeros(length(pvals),n_rand);
distortion_DM_rows     = zeros(length(pvals),n_rand);
distortion_DM_cols     = zeros(length(pvals),n_rand);
distortion_SR3_rows     = zeros(length(pvals),n_rand);
distortion_SR3_cols     = zeros(length(pvals),n_rand);

distortion_DM_oDM_rows = zeros(length(pvals),n_rand);
distortion_DM_oDM_cols = zeros(length(pvals),n_rand);
distortion_SR3_oDM_rows  = zeros(length(pvals),n_rand);
distortion_SR3_oDM_cols  = zeros(length(pvals),n_rand);
end
%%
poolobj = gcp('nocreate'); % If no pool, do not create new one.
if isempty(poolobj)
    parpool(10);
end
rng(20200826);
%%
for i = 1:n_rand
    rng(i)
    for pi = 1:length(pvals)
        %% load data
        ppp = pvals(pi)/100;
        params = [];
        params.doPerm = false;
        [X,X_orig,mask,row_labels,col_labels,gamma_vec,kNN,origPts] = ...
            missing_data_matrix(dataset,params,ppp);
        %row_labels(row_labels == 0) =10;
        %mask(isnan(X(:))) = false;
        %X(isnan(X(:))) = mean(X(~isnan(X)));

        X(~mask) = mean(X(mask));
        [n_clusters]   = length(unique(row_labels));
        params.nEigs   = n_clusters+1;
        
        if ~isempty(origPts)
            dX = squareform(pdist(origPts.rows'));
            dY = squareform(pdist(origPts.cols'));
            DM_embedding_rows = calcDM(X_orig,params);
            distortion_rows_orig(pi,i) = calc_distortion(DM_embedding_rows,dX);
            DM_embedding_cols = calcDM(X_orig',params);
            distortion_cols_orig(pi,i) = calc_distortion(DM_embedding_cols,dY);
        end
        %%
        DM_embedding_rows_missing = calcDM(X.*mask,params);
        
        idx_kmeans = kmeans(DM_embedding_rows_missing,n_clusters,'Replicates',10);
        [AR_DM(pi,i),RI_DM(pi,i)] = RandIndex(idx_kmeans,row_labels);
        
        if ~isempty(origPts)
            distortion_DM_rows(pi,i)  = calc_distortion(DM_embedding_rows_missing,dX);
            DM_embedding_cols_missing = calcDM((X)',params);
            distortion_DM_cols(pi,i)  = calc_distortion(DM_embedding_cols_missing,dY);
            
            distortion_DM_oDM_rows(pi,i) = calc_distortion(DM_embedding_rows_missing,DM_embedding_rows);
            distortion_DM_oDM_cols(pi,i) = calc_distortion(DM_embedding_cols_missing,DM_embedding_cols);
        end
        
        %%%%%%%%%%
        params.p = pvals(pi);
        params.dataset = dataset;
        params.oracleWeights = true;
        [embedding_rows, embedding_cols, gammas_sr3, emd_inds, U{i}, V{i}] = ...
            gamma_traversal_single(X,X_orig,mask,gamma_vec(1),kNN,params);
        
        idx_kmeans = kmeans(embedding_rows,n_clusters,'Replicates',10);
        [AR_SR3(pi,i),RI_SR3(pi,i)] = RandIndex(idx_kmeans,row_labels);
        
        if ~isempty(origPts)
            distortion_SR3_rows(pi,i) = calc_distortion(embedding_rows,dX);
            distortion_SR3_cols(pi,i) = calc_distortion(embedding_cols,dY);
            
            distortion_SR3_oDM_rows(pi,i) = calc_distortion(DM_embedding_rows_missing,DM_embedding_rows);
            distortion_SR3_oDM_cols(pi,i) = calc_distortion(DM_embedding_cols_missing,DM_embedding_cols);
        end
        
        save(sprintf('sr3_inpaint_random_%s_eval.mat',dataset),...
            'AR_DM','AR_SR3','pvals','gammas_sr3',...
            'distortion_rows_orig' ,'distortion_cols_orig'   ,...
            'distortion_DM_rows' ,'distortion_DM_cols'    ,...
            'distortion_SR3_rows'    ,'distortion_SR3_cols'   ,...
            'distortion_DM_oDM_rows','distortion_DM_oDM_cols',...
            'distortion_SR3_oDM_rows','distortion_SR3_oDM_cols');
        
    end
    save(sprintf('sr3_inpaint_random_%s_eval.mat',dataset),...
    'AR_DM','AR_SR3','pvals','gammas_sr3',...
    'distortion_rows_orig' ,'distortion_cols_orig'   ,...
    'distortion_DM_rows' ,'distortion_DM_cols'    ,...
    'distortion_SR3_rows'    ,'distortion_SR3_cols'   ,...
    'distortion_DM_oDM_rows','distortion_DM_oDM_cols',...
    'distortion_SR3_oDM_rows','distortion_SR3_oDM_cols');

end


return
%%
% figure
% plot(pvals/10,mean(AR_X,2),'LineWidth',2);hold on;
% plot(pvals/10,mean(AR_DM,2),'LineWidth',2);
% plot(pvals/10,mean(AR_NLPCA,2),'LineWidth',2);
% mean_Ar = squeeze(mean(AR_FRPCAG,1));
% min_Ar = (min(mean_Ar,[],1));
% max_Ar = (max(mean_Ar,[],1));
% plot(pvals/10,min_Ar,'LineWidth',2);
% plot(pvals/10,max_Ar,'LineWidth',2);
% grid on
% ylim([0 1])
% %%
% figure;errorbar(pvals/10,mean(AR_X,2),std(AR_X,0,2),'LineWidth',2)
% hold on
% errorbar(pvals/10,mean(AR_DM,2),std(AR_DM,0,2),'LineWidth',2)
% errorbar(pvals/10,mean(AR_NLPCA,2),std(AR_NLPCA,0,2),'LineWidth',2)
% plot(pvals/10,min_Ar,'LineWidth',2);
% plot(pvals/10,max_Ar,'LineWidth',2);
% grid on
% ylim([0 1])
% 
% %%
% figure
% semilogy([pvals(1)/10   pvals(end)/10], [orig_row_distort orig_row_distort],'Color','k','LineWidth',2);hold on
% semilogy(pvals/10,mean(distortion_X_rows,2),'b','LineWidth',2);hold on;
% semilogy(pvals/10,mean(distortion_DM_rows,2),'LineWidth',2);
% semilogy(pvals/10,mean(distortion_NLPCA_rows,2),'LineWidth',2);
% orig_row_distort = mean(distortion_rows_orig(:));
% grid on
% mean_distortion_FRPCAG_rows = squeeze(mean(distortion_FRPCAG_rows,1));
% mean_distortion_FRPCAG_cols = squeeze(mean(distortion_FRPCAG_cols,1));
% std_distortion_FRPCAG_rows = squeeze(std(distortion_FRPCAG_rows,0,1));
% std_distortion_FRPCAG_cols = squeeze(std(distortion_FRPCAG_cols,0,1));
% 
% semilogy(pvals/10,mean_distortion_FRPCAG_rows([1,7],:),'--','LineWidth',2);
% legend('DM','Co-manifold','DM-missing','NLPCA','FRPCAG \gamma=1','FRPCAG \gamma=100')
% % min_dist = (min(mean_distortion_FRPCAG_rows,[],1));
% % max_dist = (max(mean_distortion_FRPCAG_rows,[],1));
% % plot(pvals/10,min_dist,'LineWidth',2);
% % plot(pvals/10,max_dist,'LineWidth',2)
% %%
% 
% figure
% semilogy(pvals/10,mean(distortion_X_cols,2),'LineWidth',2);hold on;
% semilogy(pvals/10,mean(distortion_DM_cols,2),'LineWidth',2);
% semilogy(pvals/10,mean(distortion_NLPCA_cols,2),'LineWidth',2);
% orig_col_distort = mean(distortion_cols_orig(:));
% line([pvals(1)/10   pvals(end)/10], [orig_col_distort orig_col_distort],'Color','k','LineWidth',2);
% grid on
% min_dist = (min(mean_distortion_FRPCAG_cols,[],1));
% max_dist = (max(mean_distortion_FRPCAG_cols,[],1));
% semilogy(pvals/10,mean_distortion_FRPCAG_cols,'--')
% 
% %plot(pvals/10,min_dist,'LineWidth',2);
% %plot(pvals/10,max_dist,'LineWidth',2);
% 
% %%
% figure
% semilogy(pvals/10,mean(distortion_X_oDM_rows,2),'LineWidth',2);hold on;
% semilogy(pvals/10,mean(distortion_DM_oDM_rows,2),'LineWidth',2);
% 
% figure
% semilogy(pvals/10,mean(distortion_X_oDM_cols,2),'LineWidth',2);hold on;
% semilogy(pvals/10,mean(distortion_DM_oDM_cols,2),'LineWidth',2);