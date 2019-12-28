function [matrix, gamma_vec, kNN, col_labels, row_labels] = load_data(dataset)
%%
if nargin < 1
    dataset = 'checker';%'mmpi';
end

col_labels = [];
row_labels = [];
switch dataset
    %%
    case 'checker'
        matrix = rgb2gray(im2double(imread('checker.png')));
        matrix(:,51:55) = [];
        gamma_vec = 2.^[-4:2:-2];
        lim_lower = 0;
        lim_upper = 1.5;
        kNN = 16;
        matrix = matrix + 0.1*rand(size(matrix));
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

        load('lung100.mat')

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
            load('/Users/mishne/Dropbox/Yale/data/coorg/lung500.mat')
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
