function [fScore,labelCounts,labels] = evalFScore_v2(distanceMethod,spike_class,centers,score,trainSetIdx,fSpace,classificationSetIdx,verbose)
% The main difference from v2 provides from the original method is that it
% determines the average of each labeled class using the training data,
% instead of the unseen data

    numClusters = max(spike_class);
    numFeats = size(centers,2);

    % Get average of each labeled class in feature space using TRAINING set
    fs_avg = zeros(numClusters,numFeats);
    clusterCount = zeros(1,numClusters);
    for i=1:size(score,1)
        class = spike_class(trainSetIdx(i));
        fs_avg(class,:) = fs_avg(class,:) + score(i,:);
        clusterCount(class) = clusterCount(class) + 1; %sum
    end
    for i=1:numClusters
        fs_avg(i,:) = fs_avg(i,:) ./ clusterCount(i); %divide
    end
    
    % Label the found centers with closest average 
    %  (excess clusters will count up 4...5...)
    labels = zeros(1,size(centers,1));
    for i = 1:size(fs_avg,1)
        minIdx = 0;
        minDist = 0;
        for j=1:length(labels)
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
        labels(minIdx) = i;
    end
    
    maxLabel = max(labels);
    for i=1:length(labels)
        if labels(i) == 0
            maxLabel = maxLabel + 1;
            labels(i) = maxLabel;
        end
    end
    
    % Then finally classify data by nearest distance clustering
    classAfterTrain = zeros(1,size(fSpace,1));
    labelCounts = zeros(1,size(centers,1));
    for i=1:length(classificationSetIdx)

        minDist = 0;
        minIdx = 0;
        for j=1:size(centers,1) %same size as labels
            dist = 0;

            %Compute distance to center
            switch distanceMethod
                case "Manhattan"
                    dist = ComputeDistance.Manhattan(centers(j,:),fSpace(i,:));
                case "Euclidean"
                    dist = ComputeDistance.Euclidean(centers(j,:),fSpace(i,:));
            end

            %Classify with closest cluster center
            if j==1
                minDist = dist;
                minIdx = j;
                %classAfterTrain(i) = labels(j);
            end
            if dist < minDist
                minDist = dist;
                minIdx = j;
                %classAfterTrain(i) = labels(j);
            end
        end
        
        %Count the number of spikes classified to each found cluster
        labelCounts(minIdx) = labelCounts(minIdx) + 1;
        classAfterTrain(i) = labels(minIdx);
        %labelCounts(classAfterTrain(i)) = labelCounts(classAfterTrain(i))+1;
    end
    
    % Evaluate accuracy
    fp = 0;
    tp = 0;

    for i=1:length(classificationSetIdx)
        answer = spike_class(classificationSetIdx(i));
        if classAfterTrain(i) ~= answer
            fp = fp+1;
        else
            tp = tp+1;
        end
    end

    recall = tp/(tp+fp);
    fScore = 2*recall/(1+recall);
    
end