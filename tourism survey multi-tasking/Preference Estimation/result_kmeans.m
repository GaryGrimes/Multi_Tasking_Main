%% k-means statistics, raw and after merging
 clear; clc;
%% load raw data
load('res_kmeans_raw.mat')
C1 = C; D1 = D;

load('res_kmeans_mer.mat')
C2 = C; D2 = D;
fprintf('Centroids of the 3 clusters (raw_data): \n')
for i = 1:size(C1,1)
    line = find(C1(i,:));
    disp(line)
end

fprintf('\nCentroids of the 3 clusters (raw_data): \n')
for i = 1:size(C2,1)
    line = find(C2(i,:));
    disp(line)
end
