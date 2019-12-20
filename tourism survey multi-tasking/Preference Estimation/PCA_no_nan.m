 clear; clc;
%% load raw data
load('fre_raw.mat','fre_sel')
%% transfer data into normalized binary choice matrix
data = zeros(17,size(fre_sel,1));
for i=1:size(fre_sel,1)
    data(fre_sel(i,fre_sel(i,:)>0),i) = 1;
end
data = data';
%% 
% calculate the covariance
dim1 = data(:,1);
dim2 = data(:,2);
cov12 = sum((dim1-mean(dim1)).*(dim2-mean(dim2)))/(size(data,1)-1);  % cov of dim 1 & 2
cov11 = std(dim1)^2;
Cov = cov(data);
Corr_coe = corrcoef(data);

%% calculate eigenvalues
[vec,eigenval] = eig(Cov);
newval=diag(eigenval);
[y,i]=sort(newval);   % y is same as 'latent'

newi=[];
rate=y/sum(y);
sumrate=0;
for k=length(y):-1:1
    sumrate=sumrate+rate(k);
    newi(length(y)+1-k)=i(k);
    if sumrate>0.85 
        break;
    end  
end                %记下累积贡献率大85%的特征值的序号放入newi中

% fprintf('主成分数：%g\n\n',length(newi));
% fprintf('主成分载荷：\n')
% for p=1:length(newi)
%     for q=1:length(y)
%         result(q,p)=sqrt(newval(newi(p)))*vec(q,newi(p));
%     end
% end                    %计算载荷
% disp(result)

%% PCA

[COEFF,SCORE,latent,tsquare, explained] = pca(data);  % principle component analysis. coeff: component coeffecients

disp('First max 5 eigenvalues of principle components')
disp(latent(1:5))

explained2 = 100*latent/sum(latent); % calculate contributions

[m, n] = size(data); 
result1 = cell(n+1, 4); % 定义一个n+1行、4列的cell
result1(1,:) = {'Eigen', 'Diff', 'Contri', 'Cum_cont'};
result1(2:end,1) = num2cell(latent);
result1(2:end-1,2) = num2cell(-diff(latent));
result1(2:end,3:4) = num2cell([explained2, cumsum(explained2)]);

disp(result1)
%% factor analysis
[L,T] = rotatefactors(COEFF(:,1:3));

[Loadings2,specVar2,T2,stats2,F2]=factoran(data,9,'rotate','none');

result3 = cell(n+2, 4); 
result3(1,:) = {'Purp', 'Factor1', 'Factor2', 'Factor3'};
result3(2:end-1,1) = num2cell([1:n]');result3(end,1) = {'Accu_Contri'};
result3(2:end-1,2:4) = num2cell(L);
result3(end,2:end) = num2cell(explained(1:3));
disp('Results of factor analysis using 0 to fill: (factor load)');disp(result3)


%%


result3 = cell(n+2, 4); 
result3(1,:) = {'Purp', 'Factor1', 'Factor2', 'Factor3'};
result3(2:end-1,1) = num2cell([1:n]');result3(end,1) = {'Accu_Contri'};
result3(2:end-1,2:4) = num2cell(L);
result3(end,2:end) = num2cell(explained(1:3));
disp('Results of factor analysis using 0 to fill: (factor load)');disp(result3)


