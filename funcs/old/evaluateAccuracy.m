function [trainingAccuracy,classificationAccuracy] = evaluateAccuracy(distanceMethod,spike_class,centers,clAssign,trainingSetIdx,classificationSetIdx,fSpace,score,verbose)
    %
    %Check Training Accuracy
    % 
    alignmentMatrix = zeros(3);
    for i=1:length(trainingSetIdx)
        idx = trainingSetIdx(i);
        fsa_row = clAssign(i);
        if fsa_row == 0
            fsa_row = 1;
        end
        answer_col = spike_class(idx);
        alignmentMatrix(fsa_row,answer_col) = alignmentMatrix(fsa_row,answer_col)+1;
    end
    error = sum(alignmentMatrix);
    for i = 1:length(error)
        error(i)=error(i)-max(alignmentMatrix(:,i));
    end

    if verbose
        disp("Training Set Accuracy");
        disp(alignmentMatrix);
        disp(error);
    end
    
    %trainingAccuracy = 0;
    trainingAccuracy = (1-sum(error)/length(trainingSetIdx))*100;
    
    numClusters = length(centers);
    numComponents = size(fSpace,2);
    %Test Centers from Training Output
    classAfterTrain = zeros(1,length(classificationSetIdx));
    for i=1:length(classificationSetIdx)

        minDist = 0;
        for j=1:numClusters
            dist = 0;

            %Compute distance to center
            switch distanceMethod
                case "Manhattan"
                    dist = ComputeDistance.Manhattan(score(centers(j),:),fSpace(i,:));
                case "Euclidean"
                    dist = ComputeDistance.Euclidean(score(centers(j),:),fSpace(i,:));
            end

            %Classify with closest cluster center
            if j==1
                minDist = dist;
                classAfterTrain(i) = j;
            end
            if dist < minDist
                minDist = dist;
                classAfterTrain(i) = j;
            end

        end
    end

    alignmentMatrix = zeros(3);
    for i=1:length(classificationSetIdx)
        idx = classificationSetIdx(i);
        class_row = classAfterTrain(i);
        answer_col = spike_class(idx);
        alignmentMatrix(class_row,answer_col) = alignmentMatrix(class_row,answer_col)+1;
    end

    error = sum(alignmentMatrix);
    for i = 1:length(error)
        error(i)=error(i)-max(alignmentMatrix(:,i));
    end

    if verbose
        disp("Classification Set Accuracy");
        disp(alignmentMatrix);
        disp(error)
    end

    classificationAccuracy = (1-sum(error)/length(classificationSetIdx))*100;
end