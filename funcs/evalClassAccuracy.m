function classificationAccuracy = evalClassAccuracy(distanceMethod,spike_class,centers,fSpace,classificationSetIdx,verbose)
    %numClusters = 3;
    numClusters = size(centers,1);
    classAfterTrain = zeros(1,size(fSpace,1));

    for i=1:length(classificationSetIdx)

        minDist = 0;
        for j=1:numClusters
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
                classAfterTrain(i) = j;
            end
            if dist < minDist
                minDist = dist;
                classAfterTrain(i) = j;
            end

        end
    end

    alignmentMatrix = zeros(numClusters);
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