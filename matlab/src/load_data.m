function [matrix, gamma_vec, kNN, col_labels, row_labels, origPts] = load_data(dataset)
%%
if nargin < 1
    dataset = 'checker';%'mmpi';
end

col_labels = [];
row_labels = [];
origPts    = [];
switch dataset
    %%
    case 'checker'
        matrix = rgb2gray(im2double(imread('checker.png')));
        matrix(:,51:55) = [];
        gamma_vec = 2.^[-4:2:-2];
        lim_lower = 0;
        lim_upper = 1.5;
        kNN = 16;
        matrix = matrix + 0.001*rand(size(matrix));
         %%
    case 'harmonic'
        n_rows = 200;
        n_cols = 150;
        
        [x, y] = meshgrid((1:n_cols)*2*pi/n_cols, (1:n_rows)*2*pi/n_rows);
        pf = 0.5*(1+sin((x+y+2*x.*y)/2));
        matrix = pf;
        matrix = matrix + 0.5*rand(size(matrix));
        figure;imagesc(matrix,[0 1.5]);
        gamma_vec = 2.^[-6:1:-2];
        lim_lower = 0;
        lim_upper = 1.5;
        kNN = 16;
        %%
    case 'mmpi'
        if ismac
            load('/Users/mishne/Dropbox/research/tree_transform/data/MMPI2_Depolarized.mat')
        elseif isunix
            load('/data/Gal/coorg/MMPI2_Depolarized.mat')
        end
        pr = randperm(size(matrix,2),500);
        matrix = matrix(:,pr);
        original_people_scores = original_people_scores(:,pr);
        scores = scores(:,pr);
        gamma_vec = 2.^[-2:1:0];
        lim_lower = -1;
        lim_upper = 1;
        kNN = 16;
    case 'SN'
        %%
        load('/data/Gal/coorg/SN_Quantized.mat')
        gamma_vec = 2.^[-1:1:2];
        [n_r,n_c] = size(matrix);
        lim_lower = 0;
        lim_upper = 1;
        kNN = 16;
        col_labels = points_dat.class;
        [col_labels,sortI] = sort(col_labels);
        matrix = matrix(:,sortI);
    case 'lung100'
        %%
        if ismac
            load('/Users/mishne/Dropbox/Yale/data/coorg/lung100.mat')
        elseif isunix
            load('/data/Gal/coorg/lung100.mat')
        end
        matrix = lung100' / 6;
        gamma_vec = 2.^[-4:1:2];
        lim_lower = -6;
        lim_upper = 6;
        kNN = 10;
        row_labels = zeros(1,56);
        row_labels(1:20) = 1;    % carcinoid
        row_labels(21:33) = 2; % colon
        row_labels(34:50) = 3;  % normal
        row_labels(51:56) = 4; % small cell
    case 'lung500'
        if ismac
            load('lung500.mat')
        elseif isunix
            load('/data/Gal/coorg/lung500.mat')
        end
        matrix = lung500' / 6;
        gamma_vec = 2.^[-4:1:2];
        lim_lower = -6;
        lim_upper = 6;
        kNN = 10;
        row_labels = zeros(1,56);
        row_labels(1:20) = 1;    % carcinoid
        row_labels(21:33) = 2; % colon
        row_labels(34:50) = 3;  % normal
        row_labels(51:56) = 4; % small cell
    case 'sym_dif'
        %%
        r = 1:500;
        c = 1:500;
        [R,C] = meshgrid(r,c);
        matrix = 1./(0.3 + 0.2*( R-C));
        gamma_vec = 2.^[-2:1:0];
        lim_lower = -2;
        lim_upper = 2;
        kNN = 16;
        col_labels = [];
    case 'potential'
        t = 0:0.05:pi*3;
        x = cos(t);
        y = sin(t);
        z = 0.75*t;%-0.8;
        
%         t = 0:0.1:pi*6;
%         x = cos(t);
%         y = sin(t);
%         z = 0.75*(t-1);
        Xpoints = [x;y;z];
        
        nY = 300;
        pr = rand(2,nY);
        Y = [sort(pr(1,:) * 2) - 1 ;pr(2,:)+2;zeros(1,nY)];
        
        matrix = pdist2(Xpoints',Y');
        gamma_vec = 2.^[-4:1:0];
        lim_lower = -2;
        lim_upper = 2;
        kNN = 16;
        col_labels = Y(1,:);
        row_labels = t;
        origPts.rows = Xpoints;
        origPts.cols = Y;
    case 'potential3'
        t = 0:0.1:pi*6;
        x = 0.5+0.5*cos(t);
        y = 0.5*sin(t);
        z = -9+t;
        Xpoints = [x;y;z];
        
        nY = 300;
        pr = rand(2,nY);
        Y = [sort(pr(1,:)) ;2*ones(1,nY);4*(pr(2,:))-2];
        
        matrix = 1./pdist2(Xpoints',Y');
        gamma_vec = 2.^[-4:1:0];
        lim_lower = -2;
        lim_upper = 2;
        kNN = 16;
        col_labels = Y(1,:);
        origPts.rows = Xpoints;
        origPts.cols = Y;
    case 'potential2'
                %%
        mu1 = [0 0 0;-1 1 0.5;1 1 1];
        sigma1 = cat(3,[0.2 0 0;0 0.15 0;0 0 0.1].^2,[0.15 0 0;0 0.2 0;0 0 0.1].^2,[0.2 0 0;0 0.1 0;0 0 0.15].^2);
        p1 = [1/3 1/3 1/3];
        obj = gmdistribution(mu1,sigma1,p1);
        [Xpoints, idx] = random(obj,200);
        [~,sorted] = sort(idx);
        Xpoints = Xpoints(sorted,:);
        Xpoints = Xpoints';
        
        nY = 300;
        pr = rand(2,nY);
        Y = [sort(pr(1,:) * 2) - 1 ;pr(2,:)+2;0.2*ones(1,nY)];
        
        matrix = pdist2(Xpoints',Y');
        gamma_vec = 2.^[-4:1:0];
        lim_lower = -2;
        lim_upper = 2;
        kNN = 16;
        col_labels = Y(1,:);
        row_labels = idx(sorted);
        origPts.rows = Xpoints;
        origPts.cols = Y;
    case 'potential2_fix'
                %%
        mu1 = [0 0 0;-1 1 0.5;1 1 1];
        sigma1 = cat(3,[0.2 0 0;0 0.15 0;0 0 0.1].^2,[0.15 0 0;0 0.2 0;0 0 0.1].^2,[0.2 0 0;0 0.1 0;0 0 0.15].^2);
        p1 = [1/3 1/3 1/3];
        obj = gmdistribution(mu1,sigma1,p1);
        [Xpoints, idx] = random(obj,200);
        [~,sorted] = sort(idx);
        Xpoints = Xpoints(sorted,:);
        Xpoints = Xpoints';
        
        nY = 300;
        p = rand(2,nY);
        Y = [sort(p(1,:) * 2) - 1 ;p(2,:)+2;0.2*ones(1,nY)];
        
        matrix = 1./pdist2(Xpoints',Y');
        gamma_vec = 2.^[-4:1:0];
        lim_lower = -2;
        lim_upper = 2;
        kNN = 16;
        col_labels = Y(1,:);
        row_labels = idx(sorted);
    case 'TCGA'
        load('/data/Gal/coorg/brca547.mat')
        matrix = TCGA500'/4;
        gamma_vec = 2.^[-4:1:2];
        kNN = 10;
        [labels,~,row_labels]=unique(points_dat.PAM50);
        col_labels = 1:500;
    case 'mnist1'
        load('/data/Gal/coorg/mnist.mat')
        matrix = double(mnist034);
        matrix(:,sum(matrix)==0) = [];
        matrix = matrix (1:3:end,:)/255;
        gamma_vec = 2.^[-4:1:2];
        kNN = 10;
        row_labels = labels034(1:3:end);
        col_labels = 1:size(matrix,2);
    case 'mnist2'
        load('/data/Gal/coorg/mnist.mat')
        matrix = double(mnist358);
        matrix(:,sum(matrix)==0) = [];
        matrix = matrix (1:3:end,:)/255;
        
        matrix=bsxfun(@minus,matrix,mean(matrix));
        matrix=bsxfun(@times,matrix,1./std(matrix));
        
        gamma_vec = 2.^[-4:1:2];
        kNN = 10;
        row_labels = labels358(1:3:end);
        col_labels = 1:size(matrix,2);
    case 'mouse'
        load('/data/Gal/coorg/mouse_expression.mat');
        row_labels = class_id;
        mean_cols = nanmean(matrix);
        std_cols = nanstd(matrix);
        matrix = bsxfun(@minus,matrix,mean_cols);
        matrix = bsxfun(@rdivide,matrix,std_cols);
       col_labels = 1:size(matrix,2);
       kNN = 7;
       gamma_vec = 2.^[-4:1:2];
    case 'voice'
       load('/data/Gal/coorg/LSVT.mat');
        mean_cols = nanmean(matrix);
        std_cols = nanstd(matrix);
        matrix = bsxfun(@minus,matrix,mean_cols);
        matrix = bsxfun(@rdivide,matrix,std_cols);
       col_labels = 1:size(matrix,2);
       kNN = 7;
       gamma_vec = 2.^[-4:1:2];
end
% figure;scatter3(Xpoints(1,:),Xpoints(2,:),Xpoints(3,:),30,(1:189)/189,'filled');hold on; scatter3(Y(1,:),Y(2,:),Y(3,:),30,(Y(1,:)+1)/2,'filled');colormap jet
% figure;scatter3(Xpoints(1,:),Xpoints(2,:),Xpoints(3,:),30,idx(sorted)/3,'filled');hold on; scatter3(Y(1,:),Y(2,:),Y(3,:),30,(Y(1,:)+1)/2,'filled');colormap jet
%%
% figure;
% scatter3(Xpoints(1,idx(sorted)==1),Xpoints(2,idx(sorted)==1),Xpoints(3,idx(sorted)==1),30,'b','filled');
% hold on; 
% scatter3(Xpoints(1,idx(sorted)==2),Xpoints(2,idx(sorted)==2),Xpoints(3,idx(sorted)==2),30,'g','filled');
% scatter3(Xpoints(1,idx(sorted)==3),Xpoints(2,idx(sorted)==3),Xpoints(3,idx(sorted)==3),30,'r','filled');
% scatter3(Y(1,:),Y(2,:),Y(3,:),30,(Y(1,:)+1)/2,'filled');colormap jet
