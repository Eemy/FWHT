function [labels,overwrites] = labelCenters(distanceMethod,spike_class,centers,trainFeatures,trainSetIdx)
%Given a set of centers, label them. First, training data is averaged
%according their labels and the centers are simply labeled with the closest
%one. 

    numClusters = max(spike_class);
    numFeats = size(centers,2);

    % Get average of each labeled class in feature space using TRAINING set
    fs_avg = zeros(numClusters,numFeats);
    clusterCount = zeros(1,numClusters);
    for i=1:size(trainFeatures,1)
        class = spike_class(trainSetIdx(i));
        fs_avg(class,:) = fs_avg(class,:) + trainFeatures(i,:);
        clusterCount(class) = clusterCount(class) + 1; %sum
    end
    for i=1:numClusters
        fs_avg(i,:) = fs_avg(i,:) ./ clusterCount(i); %divide
    end
    
    % Label the found centers with closest average 
    NCLUST = size(centers,1);
    labels = zeros(1,NCLUST);
    overwrites = 0;
    for i = 1:size(fs_avg,1)
        
        %Find closest labeled average
        minIdx = 0;
        minDist = 0;
        for j=1:NCLUST
            switch distanceMethod
                case "Manhattan"
                    dist = ComputeDistance.Manhattan(centers(j,:),fs_avg(i,:));
                case "Euclidean"
                    dist = ComputeDistance.Euclidean(centers(j,:),fs_avg(i,:));
            end
            
            if j==1
                minDist = dist;
                minIdx = j;
            end
            if dist < minDist
                minDist = dist;
                minIdx = j;
            end
        end
        
        if(labels(minIdx) ~= 0)
            overwrites = overwrites + 1;
        end
        labels(minIdx) = i;
    end
    
    % Excess clusters will count up 4...5...
    if NCLUST > numClusters
        maxLabel = max(labels);
        for i=1:length(labels)
            if labels(i) == 0
                if maxLabel < NCLUST
                    maxLabel = maxLabel + 1;
                end
                labels(i) = maxLabel;
            end
        end
    % For less than 3 clusters
    elseif NCLUST == 2
        present = ones(1,numClusters);
        for i=1:length(labels)
            if labels(i) == 0
                continue;
            end
            present(labels(i)) = 0;
        end
        for i=1:length(labels)
            if (NCLUST < labels(i)) || (labels(i) == 0)
                notPresent = find(present);
                labels(i) = notPresent(1);
            end
        end
    elseif NCLUST == 1
        labels = 1;
    end

    %This is a useless block of code LOL, it will never actually do
    %anything because of how I handled all of my cases
    if overwrites > 0
        for i=1:length(labels)
            if labels(i) == 0
                labels(i) = 1;
            end
        end
    end

end