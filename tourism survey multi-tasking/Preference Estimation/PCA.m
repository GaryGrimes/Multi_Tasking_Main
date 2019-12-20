
%% load raw data
load('fre_raw.mat','fre_sel')
%% transfer data into normalized binary choice matrix
data = zeros(17,size(fre_sel,1));
for i=1:size(fre_sel,1)
    data(fre_sel(i,fre_sel(i,:)>0),i) = 1;
end
% replace 0s with NaN, as missing data
data(data==0) = NaN;
data = data';
%% 
% calculate the covariance
dim1 = data(:,1);
dim2 = data(:,2);
cov12 = sum((dim1-mean(dim1)).*(dim2-mean(dim2)))/(size(data,1)-1);  % cov of dim 1 & 2
cov11 = std(dim1)^2;
Cov = cov(data);  % no covariance or correlation coefficients
Corr_coe = corrcoef(data);
%%  PCA and factor analysis of the satisfaction questinare.  Ref: https://ww2.mathworks.cn/help/stats/pca.html#bttvehk-2
% [COEFF,SCORE,latent,tsquare] = pca(data','Rows','complete');  
% Each column of score corresponds to one principal component. 
% The vector, latent, stores the variances of the principal components.

[coeff1,score1,latent,tsquared,explained,mu1] = pca(data,...
'algorithm','als');  % PCA using Alternating LS for the presence of missing data

% Reconstruct the observed data.
t = score1*coeff1' + repmat(mu1,size(data,1),1);

% explained ratio (total variance explained by principle components
[m, n] = size(data); 
result1 = cell(n+1, 4); 

result1(1,:) = {'Eigen', 'Diff', 'Contri', 'Cum_cont'};
result1(2:end,1) = num2cell(latent);
result1(2:end-1,2) = num2cell(-diff(latent));  %  principal component variances
result1(2:end,3:4) = num2cell([explained, cumsum(explained)]);
disp(result1)


[L1,T1] = rotatefactors(coeff1(:,1:3));

result2 = cell(n+2, 4); 

result2(1,:) = {'Purp', 'Factor1', 'Factor2', 'Factor3'};
result2(2:end-1,1) = num2cell([1:n]');result2(end,1) = {'Accu_Contri'};
result2(2:end-1,2:4) = num2cell(L1);
result2(end,2:end) = num2cell(explained(1:3));
disp('Results of factor analysis1: ');disp(result2)

%% a second PCA analysis
[coeff2,score2,latent2,tsquared2,explained2,mu2] = pca(data,'algorithm','als');
[L2,T2] = rotatefactors(coeff2(:,1:3));

% [L3,psi,T] = factoran(data,3);  % Error using svd: Input to SVD must not contain NaN or Inf.

result3 = cell(n+2, 4); 
result3(1,:) = {'Purp', 'Factor1', 'Factor2', 'Factor3'};
result3(2:end-1,1) = num2cell([1:n]');result3(end,1) = {'Accu_Contri'};
result3(2:end-1,2:4) = num2cell(L2);
result3(end,2:end) = num2cell(explained2(1:3));
disp('Results of factor analysis2: ');disp(result3)






