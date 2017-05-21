%--------------------------PCA-------------------------------------------
load face_images.mat
%reshaping the matrix to 4096*910 
images = reshape(face_images, [4096,910]);
%find means
mean_images = mean(images,2);
%subtract means
mean_shift = images(:,:)-mean_images(:,:);
%find covariance matrix
cov_mat = cov(mean_shift');
% find eigenvectors and eigenvalues
[Evec, Eval] = eig(cov_mat);
%arranging eigenvectors and eigenvalues
for i=1:1:4096
    newEval(i,i) = Eval(4097-i,4097-i);
    newEvec(:,i) = Evec(:,4097-i);
end
%finding percentage of variance and cumulative variance of PCs
tr = trace(newEval);
sum = 0;
for i=1:1:4096
    per_var(1,i) = 100*newEval(i,i)/tr;
    sum = sum + newEval(i,i)/tr;
    cumulative_var(1,i) = 100*sum;
end

%defining the feature vector by selecting no of features
feature_vector = newEvec(:,1:25);
%calculating principal components
principal_components = feature_vector'*mean_shift;
%compression/top 4 PCs
comp_images = feature_vector*principal_components;
comp_images = comp_images(:,:) + mean_images(:,:);
comp_images = reshape(comp_images,[64,64,910]);

feature_vecsize = size(feature_vector);
comp_ratio = feature_vecsize(1,2)/4096;

%-----------------------------------K-Means--------------------------------------
%initialise random cluster centers
cluster_centers = randi([-10,10],13,2);
%intitialisations
mean_clusters = zeros(13,2);
new_clusters = zeros(13,2);
points_count = zeros(13,1);
dis = zeros(910,13);
stop_iterations = 0;
iteration = 0;
while(stop_iterations==0)
for i=1:1:910
    for j=1:1:13
        %finding distance of each data point from each cluster centers
        dis(i,j)=sqrt((principal_components(1,i)-cluster_centers(j,1))^2 + (principal_components(2,i)-cluster_centers(j,2))^2);
    end
end
%arranging distance matrix in ascending order and finding indices
[asc_dis,index_dis] = sort(dis,2,'ascend');
for i=1:1:910
    %adding means for each cluster and number of data points of each
    %cluster
    mean_clusters(index_dis(i,1),1) = mean_clusters(index_dis(i,1),1)+ principal_components(1,i);
    mean_clusters(index_dis(i,1),2) = mean_clusters(index_dis(i,1),2)+ principal_components(2,i);
    points_count(index_dis(i,1),1)= points_count(index_dis(i,1),1)+1; 
end
for i=1:1:13
    %calculating means of each cluster
    if(points_count(i,1)~=0)
    mean_clusters(i,1)= mean_clusters(i,1)/ points_count(i,1);
    mean_clusters(i,2)= mean_clusters(i,2)/ points_count(i,1);
    end
end
%copying values
new_clusters = mean_clusters;
if(prod(prod(new_clusters==cluster_centers))==1)
    %comparing new and current values
    stop_iterations = 1;
else
   cluster_centers = mean_clusters;
   iteration = iteration + 1;
   mean_clusters = zeros(13,2);
   points_count = zeros(13,1); 
   dis = zeros(910,13);
end
end
%----------------------------------Face recognition------------------------
%wrt weights
load unknown_faces.mat
%reshaping unknown faces matrix
uface = reshape(unknown_faces,[4096,65]);
%finding means
mean_uface = mean(uface,2);
%subtracting means
uface_shift = uface(:,:)-mean_uface(:,:);
%finding weights (projecting mean shifted images on top eigen vectors aka projection matrix)
weights = feature_vector'*uface_shift;
L2= [];
classify_knn=[];
for i=1:1:65
    for j=1:1:910
        %comparing each unknown with known image using L2 norm
        L2 = [L2 norm(weights(:,i)-principal_components(:,j))];
    end
    %sorting L2 in ascending order
    [l2_dis l2_index] = sort(L2,'ascend');
    %dividing by 70 for classification (70 images per class)
    %calculating quotient 
    classify_q = fix(l2_index(1,1)/70);
    %calculating remainder
    classify_r = mod(l2_index(1,1),70);
    %incase where it is a multiple of 70
    if classify_r == 0
        %incase if it is 70
        if classify_q == 1
            classify_knn = [classify_knn classify_q];
        else
            classify_knn = [classify_knn classify_q-1];
        end
    %not a multiple of 70    
    else
        classify_knn = [classify_knn classify_q+1];
    end
    L2=[];
end


%wrt original images
L2= [];
classify_og=[];
for i=1:1:65
    for j=1:1:910
        %comparing each unknown with known image using L2 norm
        L2 = [L2 norm(uface(:,i)-images(:,j))];
    end
    %sorting L2 in ascending order
    [l2_dis l2_index] = sort(L2,'ascend');
    %dividing by 70 for classification (70 images per class)
    %calculating quotient 
    classify_q = fix(l2_index(1,1)/70);
    %calculating remainder
    classify_r = mod(l2_index(1,1),70);
    %incase where it is a multiple of 70
    if classify_r == 0
        %incase if it is 70
        if classify_q == 1
            classify_og = [classify_og classify_q];
        else
            classify_og = [classify_og classify_q-1];
        end
    %not a multiple of 70    
    else
        classify_og = [classify_og classify_q+1];
    end
    L2=[];
end


figure;
for i=1:1:910
    plot(principal_components(1,i),principal_components(2,i),'o');
    hold on
    %text(principal_components(1,i),principal_components(2,i),num2str(i));
end
for i=1:1:13
    scatter(cluster_centers(i,1),cluster_centers(i,2),'filled','k');
    hold on
end
xlabel('Principal Component 1');
ylabel('Principal Component 2');
figure;
plot(per_var,'LineWidth',1.25,'Color',[1 0 0]);
xlabel('Principal Component');
ylabel('Percentage of Variance');
figure;
plot(cumulative_var, 'LineWidth',1.25,'Color',[0 0 1]);
xlabel('Principal Component');
ylabel('Percentage of Variance');
