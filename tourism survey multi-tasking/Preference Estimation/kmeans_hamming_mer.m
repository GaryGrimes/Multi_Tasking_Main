%% k-means using hamming distance
% raw_data: dimension reduced (low frequency purposes merged)
 clear; clc;
%% load raw data
load('fre_after.mat')
load('fre_after_newly merged.mat')

%% transfer data into normalized binary choice matrix
% data = zeros(max(fre_after(:,3)),size(fre_after,1));
% for i=1:size(fre_after,1)
%     data(fre_after(i,fre_after(i,:)>0),i) = 1;
% end
% data = data';
% data_binary = logical(data);

data = zeros(max(max(newly_merged_purposes)),size(fre_after,1));
for i=1:size(newly_merged_purposes,1)
    data(newly_merged_purposes(i,newly_merged_purposes(i,:)>0),i) = 1;
end
data = data';
data_binary = logical(data);
%% 3 clusters
fprintf('kmeans++ using 3 clusters: \n');fprintf('\n');
[idx,C,sumd,D] = kmeans(data_binary,3,'Distance','hamming', 'Display','final','Replicates',100);
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

data_cluster = [data_reformated,idx];
%% 4 clusters
fprintf('\n');fprintf('kmeans++ using 4 clusters: \n')
[idx4,C4,sumd4,D4] = kmeans(data_binary,4,'Distance','hamming','Display','final','Replicates',15);
fprintf('\n')
disp('Centroids: ')
for i = 1:size(C4,1)
    if isempty(find(C4(i,:),1))
        disp([0,0,0])
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
        disp([0,0,0])
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

% %% discard records that include both 1 and 6 （after merging. (red leaves) 8 became 6
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
% %% reformat 'data' to unique purpose sets
% [m,~] = size(data2);
% data2_reformated = zeros(m,3);
% for i =1:m
%     row = data2(i,:);  % each row
%     if any(row)
%         row = sort(find(row),'ascend');
%         for j =1:length(row)
%             data2_reformated(i,j) = row(j);
%         end
%     end
% end
% 
% %% perform kmeans++ using 3 clusters excluding data containing both 1 and 8 
% data2_binary = logical(data2);
% fprintf('kmeans++ using 3 clusters with data containing both 1 and 6 (8 before merging) excluded: \n');fprintf('\n');
% [idx2,C2,sumd2,D2] = kmeans(data2_binary,3,'Distance','hamming', 'Display','final','Replicates',15);
% fprintf('\n')
% disp('Centroids: ')
% for i = 1:size(C2,1)
%     if isempty(find(C2(i,:),1))
%         disp([0,0,0])
%     else
%         disp(find(C2(i,:)))
%     end
% end
% fprintf('within-cluster sums of point-to-centroid distances: c1: %.2f, c2: %.2f, c3: %.2f.\n',sumd2')
% fprintf('cluster sizes: c1: %d, c2: %d, c3: %d.\n',[sum(idx2==1),sum(idx2==2),sum(idx2==3)]);fprintf('\n')
% data2_reformated = [data2_reformated,idx2];  % purpose frequecy result with cluster index

%% 4-23 discard columns 1 and 6
% replace columns with zeros
data_zeros16 = data;
data_zeros16(:,[1,6]) = 0;

% delete columns
% data_zeros18 = data;
% data_zeros18(:,[1,6]) = [];

% Above two approaches have same results
%% reformat 'data' to unique purpose sets
data2 = data_zeros16;
[m,~] = size(data2);
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

% frequency of purposes (unique as sets)
fre_ex16 = unique(data2_reformated,'rows');
nn = zeros(1,size(fre_ex16,1));
for i =1:size(data2_reformated,1)
    [~,j] = ismember(data2_reformated(i,:),fre_ex16,'rows');
    nn(j) = nn(j)+1;
end
fre_ex16 = [fre_ex16,nn'];

%% perform kmeans++ using 3 clusters not considering both 1 and 8 
data2_binary = logical(data2);
fprintf('kmeans++ using 3 clusters with data containing both 1 and 6 (8 before merging) excluded: \n');fprintf('\n');
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

%% perform kmeans++ using 4 clusters not considering both 1 and 8 
data2_binary = logical(data2);
fprintf('kmeans++ using 4 clusters not considering 1 and 6: \n');fprintf('\n');
[idx3,C3,sumd3,D3] = kmeans(data2_binary,4,'Distance','hamming', 'Display','final','Replicates',15);
fprintf('\n')
disp('Centroids: ')
for i = 1:size(C3,1)
    if isempty(find(C3(i,:),1))
        disp([0,0,0])
    else
        disp(find(C3(i,:)))
    end
end
fprintf('within-cluster sums of point-to-centroid distances: c1: %.2f, c2: %.2f, c3: %.2f, c4: %.2f, .\n',sumd3')
fprintf('cluster sizes: c1: %d, c2: %d, c3: %d, c4: %d.\n',[sum(idx3==1),sum(idx3==2),sum(idx3==3),sum(idx3==4)]);fprintf('\n')
data3_reformated = [data2_reformated,idx3];  % purpose frequecy result with cluster index

%% save frequency, idx, C and D
fre_after_merged = [data_reformated,idx];
% save('res_kmeans_mer.mat','fre_after_merged','C','D')

