%% k-means using hamming distance
clear; clc;
%% load raw data
load('fre_raw.mat','fre_sel')
%% transfer data into normalized binary choice matrix
data = zeros(17,size(fre_sel,1));
for i=1:size(fre_sel,1)
    data(fre_sel(i,fre_sel(i,:)>0),i) = 1;
end
data = data';
data_binary = logical(data);
%% 3 clusters
fprintf('kmeans++ using 3 clusters: \n');fprintf('\n');
[idx,C,sumd,D] = kmeans(data_binary,3,'Distance','hamming', 'Display','final','Replicates',15);
fprintf('\n')
disp('Centroids: ')
for i = 1:size(C,1)
    if isempty(find(C(i,:),1))
        disp([0,0,0])
    else
        disp(find(C(i,:)))
    end
end
fprintf('within-cluster sums of point-to-centroid distances: c1: %.2f, c2: %.2f, c3: %.2f.\n',sumd')
fprintf('cluster sizes: c1: %d, c2: %d, c3: %d.\n',[sum(idx==1),sum(idx==2),sum(idx==3)]);fprintf('\n')

%% 4 clusters
fprintf('\n');fprintf('kmeans++ using 4 clusters: \n')
[idx4,C4,sumd4,D4] = kmeans(data_binary,4,'Distance','hamming','Display','final','Replicates',15);
fprintf('\n')
disp('Centroids: ')
for i = 1:size(C4,1)
    if isempty(find(C4(i,:),1))
        disp([0,0,0,0])
    else
        disp(find(C4(i,:)))
    end
end
fprintf('within-cluster sums of point-to-centroid distances: c1: %.2f, c2: %.2f, c3: %.2f, c4: %.2f.\n',sumd4')
fprintf('cluster sizes: c1: %d, c2: %d, c3: %d, c4: %d.\n',[sum(idx4==1),sum(idx4==2),sum(idx4==3),sum(idx4==4)]);fprintf('\n')
%% 5 clusters
fprintf('\n');fprintf('kmeans++ using 5 clusters: \n')
[idx5,C5,sumd5,D5] = kmeans(data_binary,5,'Distance','hamming','Display','final','Replicates',15);
fprintf('\n')
disp('Centroids: ')
for i = 1:size(C5,1)
    if isempty(find(C5(i,:),1))
        disp([0,0,0,0,0])
    else
        disp(find(C5(i,:)))
    end
end
fprintf('within-cluster sums of point-to-centroid distances: c1: %.2f, c2: %.2f, c3: %.2f, c4: %.2f, c5: %.2f.\n',sumd5')
fprintf('cluster sizes: c1: %d, c2: %d, c3: %d, c4: %d, c5: %d.\n',...
    [sum(idx5==1),sum(idx5==2),sum(idx5==3),sum(idx5==4),sum(idx5==5)])
fprintf('\n');

%% reformat 'data' to unique purpose sets
[m,~] = size(data);
data_reformated = zeros(m,3);
for i =1:m
    row = data(i,:);  % each row
    if any(row)
        row = sort(find(row),'ascend');
        for j =1:length(row)
            data_reformated(i,j) = row(j);
        end
    end
end

%% discard records that include both 1 and 8.
% row_idx = ones(1,m);
% for i =1:m
%     row = data(i,:);  % each row
%     if row(1)==1 && row(6)==1
%         row_idx(i) = 0;
%     end
% end
% fprintf('# of people who didn''t choose 1 and 8: %d.\n', sum(row_idx))
% % data2 contains records that don't include both 1 and 8
% data2 = data(logical(row_idx),:);

%% 4-23: discard columns 1 and 8
% replace columns with zeros
data_zeros18 = data;
data_zeros18(:,[1,8]) = 0;

% delete columns
% data_zeros18 = data;
% data_zeros18(:,[1,8]) = [];

% Above two approaches have same results
%% reformat 'data' to unique purpose sets
data2 = data_zeros18;
[m,n] = size(data2);
data2_reformated = zeros(m,3);
for i =1:m
    row = data2(i,:);  % each row
    if any(row)
        row = sort(find(row),'ascend');
        for j =1:length(row)
            data2_reformated(i,j) = row(j);
        end
    end
end

fre_ex18 = unique(data2_reformated,'rows');
nn = zeros(1,size(fre_ex18,1));
for i =1:size(data2_reformated,1)
    [~,j] = ismember(data2_reformated(i,:),fre_ex18,'rows');
    nn(j) = nn(j)+1;
end
fre_ex18 = [fre_ex18,nn'];
%% perform kmeans++ using 3 clusters with data containing both 1 and 8 excluded
data2_binary = logical(data2);
fprintf('kmeans++ using 3 clusters with data containing both 1 and 8 excluded: \n');fprintf('\n');
[idx2,C2,sumd2,D2] = kmeans(data2_binary,3,'Distance','hamming', 'Display','final','Replicates',15);
fprintf('\n')
disp('Centroids: ')
for i = 1:size(C2,1)
    if isempty(find(C2(i,:),1))
        disp([0,0,0])
    else
        disp(find(C2(i,:)))
    end
end
fprintf('within-cluster sums of point-to-centroid distances: c1: %.2f, c2: %.2f, c3: %.2f.\n',sumd2')
fprintf('cluster sizes: c1: %d, c2: %d, c3: %d.\n',[sum(idx2==1),sum(idx2==2),sum(idx2==3)]);fprintf('\n')
data2_reformated = [data2_reformated,idx2];  % purpose frequecy result with cluster index



%% save frequency, idx, C and D
fre_sel_raw = [data_reformated,idx];
% save('res_kmeans_raw.mat','fre_sel_raw','C','D')


%% k-means on person destination choices
load('person_destination.mat')
% destination choices are stored in variable 'pd'
numbering = pd(:,1); dest_choice = pd(:,2:end);

% attraction choices are extracted from 'pd' and stored as
% 'attraction_choice'

attraction_choice = pd(:,2:38);


% check destination choice 为空的人
pempty = [];
for i = 1:size(dest_choice,1)
    if sum(dest_choice(i,:)) == 0
        pempty = [pempty;i,numbering(i)];
    end
end
%

[X,idx,C] = kmeans_n(dest_choice,3,100);

destinations = values;

%% Elbow method for deciding appropriate # of clusters
Cost = zeros(10,1);
for k = 1:10
    [~,~,C,Sumd] = kmeans_n(dest_choice,k,150);
    disp('CLuster centroids: ')
    for j = 1:size(C,1)
        fprintf('Cluster %d: ',j)
        if isempty(find(C(j,:),1))
            disp('None')
        else
            res = find(C(j,:));
            for i =1:length(res)
                if i>1
                    fprintf(', ')
                end
                fprintf(destinations{res(i)});
            end
            fprintf('\n')
        end
    end
    Cost(k) = sum(Sumd);
end
figure('Name','Elbow curve')
plot(1:10,Cost)
title('Elbow curve')
xlabel('K: no. of clusters')
ylabel('Cost function J')

%% Elbow method for deciding appropriate # of clusters --> attractions only
Cost = zeros(10,1);
Index = zeros(length(numbering),10);
for k = 1:10
    [~,idx,C,Sumd] = kmeans_n(attraction_choice,k,250);
    Index(:,k) = idx;
    % print
    disp('CLuster centroids: ')
    for j = 1:size(C,1)
        fprintf('Cluster %d: ',j)
        if isempty(find(C(j,:),1))
            disp('None')
        else
            res = find(C(j,:));
            for i =1:length(res)
                if i>1
                    fprintf(', ')
                end
                fprintf(destinations{res(i)});
            end
            fprintf('\n')
        end
    end
    Cost(k) = sum(Sumd);
end
figure('Name','Elbow curve')
plot(1:10,Cost)
title('Elbow curve')
xlabel('K: no. of clusters')
ylabel('Cost function J')
%% functions
function [X,idx,C,J] = kmeans_n(X,n,iterations)
% n clusters

fprintf("\nKmeans using %d clusters: \n", n);
fprintf('\n');

[idx,C,J,~] = kmeans(X,n,'Distance','hamming','Display','off','Replicates',iterations);
fprintf('\n')
disp('Centroids: ')
for i = 1:size(C,1)
    if isempty(find(C(i,:),1))
        disp([0,0,0])
    else
        disp(find(C(i,:)))
    end
end
fprintf('within-cluster sums of point-to-centroid distances: ')
for i = 1:size(J)
    fprintf('c%d: %.2f ',i,J(i))
end

fprintf('\ncluster sizes: ')
for i = 1:size(C,1)
    fprintf('c%d: %d ',i,sum(idx==i))
end
fprintf('\n')
X = [X,idx];  % clustering result with cluster index
end




