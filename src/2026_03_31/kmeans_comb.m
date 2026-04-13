clear
addpath("../../funcs/","../../modules")
path = "../../../SpikeTraining_RISCV/Raw_Data/";

%% Set up parameters for run
% baseName = ["C_Easy1_noise005","C_Easy1_noise01","C_Easy1_noise015","C_Easy1_noise02", ...
%             "C_Easy1_noise025","C_Easy1_noise03","C_Easy1_noise035","C_Easy1_noise04", ...
%             "C_Easy2_noise005","C_Easy2_noise01","C_Easy2_noise015","C_Easy2_noise02", ...
%             "C_Difficult1_noise005","C_Difficult1_noise01","C_Difficult1_noise015","C_Difficult1_noise02", ...
%             "C_Difficult2_noise005","C_Difficult2_noise01","C_Difficult2_noise015","C_Difficult2_noise02"
%             ];

% baseName = ["C_Easy1_noise005","C_Easy1_noise01","C_Easy1_noise015","C_Easy1_noise02", ...
%              "C_Easy1_noise025","C_Easy1_noise03","C_Easy1_noise035","C_Easy1_noise04"];

% baseName = ["C_Easy2_noise005","C_Easy2_noise01","C_Easy2_noise015","C_Easy2_noise02"];

% baseName = ["C_Difficult1_noise005","C_Difficult1_noise01","C_Difficult1_noise015","C_Difficult1_noise02"];

% baseName = ["C_Difficult2_noise005","C_Difficult2_noise01","C_Difficult2_noise015","C_Difficult2_noise02"];

baseName = ["C_Easy1_noise04","C_Difficult1_noise02"];

numIters = 1;
minComponents = 2;
maxComponents = 2;
distanceMethod = "Manhattan";
feMethod = "FWHT";
coeffOrder = "hadamard"; %WHT = 'sequency', HT = 'hadamard'
showPlot = false;
filesInvolved = "all";

%% Load data
% classCounts = [20,20,20 ; ...
%                 30,30,30 ; ...
%                 40,40,40 ; ...
%                 50,50,50 ; ...
%                 100,100,100; ...
%                 200,200,200; ...
%                 300,300,300; ...
%                 400,400,400; ...
%                 700,700,700];
classCounts = [700,700,700]; % one row for each data file
% classCounts = [10,10,10 ; ...
%                 20,20,20 ; ...
%                 30,30,30 ; ...
%                 40,40,40 ; ...
%                 50,50,50];
numGroups = 3;
startOffset = int32(12);
vectorSize = int32(64);
noise = int32(0);

%Create Dataset objects for each file
numSpikes = 0; trainCount = 0; classCount = 0;
numFiles = numel(baseName);
dataCell = cell(1,numFiles);
for fileIdx = 1:numFiles
    dataCell{fileIdx} = Dataset(path,baseName(fileIdx),vectorSize,startOffset,noise);
    numSpikes = numSpikes+dataCell{fileIdx}.numSpikes;
end

ntrainingSize = size(classCounts,1);
trainingPortion = zeros(1,ntrainingSize);
for i=1:numFiles
    for j=1:ntrainingSize
        trainingPortion(i,j) = sum(classCounts(j,:))/dataCell{i}.numSpikes*100;
    end
end

%% Store results
allFScoreUnseen = zeros(ntrainingSize,numIters,numFiles);

%% Perform Feature Extraction and Clustering
for numComponents = minComponents:maxComponents
prefix = strcat(feMethod+numComponents,"_allcomb_",filesInvolved);

for fileIdx = 1:numFiles
    for trainVal = 1:ntrainingSize
        for runNum = 1:numIters
            dataCell{fileIdx}.splitDataCounts(classCounts(trainVal,:));
            [trainingFeatures,classificationFeatures] = FeatureExtract.getFeatureVecs(feMethod,dataCell{fileIdx}.spikeVecs,coeffOrder,dataCell{fileIdx}.trainingIdx,dataCell{fileIdx}.classIdx);
            
            bestCoeffs = [];
            bestCenters = [];
            bestF1 = 0;
            feature_combinations = nchoosek(1:vectorSize, numComponents); % Generate combinations of features
            for selIdx = 1:size(feature_combinations, 1)                %% Coefficient Selection: sliding window
                %Store continuous coeffs
                coeffSel = feature_combinations(selIdx);
                trainingFeatures_temp = trainingFeatures(:,coeffSel);
                classificationFeatures_temp = classificationFeatures(:,coeffSel);
                
                %% Perform Clustering (K-Means) -- maybe FSA in the future
                %opts = statset('Display','final');
                %[clAssign_train,centers] = kmeans(trainingFeatures_temp,numGroups,'Distance','cityblock','Replicates',5,'Options',opts);
                [clAssign_train,centers] = kmeans(trainingFeatures_temp,numGroups,'Distance','cityblock','Replicates',5);
                [~,clAssign_unseen] = pdist2(centers,classificationFeatures_temp,'cityblock','Smallest',1);
                
                %% Obtain Confusion Matrix
                trueLabels = dataCell{fileIdx}.spike_class(dataCell{fileIdx}.classIdx);
                alignMat = confusionmat(trueLabels,clAssign_unseen');
                truePositiveP = max(alignMat, [], 1);
                [truePositiveR, Reorder] = max(alignMat, [], 2);
                valPrecision = truePositiveP ./ sum(alignMat, 1);
                valPrecision = [valPrecision(1, Reorder(1, 1)), ...
                                valPrecision(1, Reorder(2, 1)), ...
                                valPrecision(1, Reorder(3, 1))];
                valRecall = truePositiveR ./ sum(alignMat, 2);
                valRecall = valRecall';
                
                valF1 = 2 .* valPrecision .* valRecall ./ (valPrecision + valRecall);
                valPrecision = mean(valPrecision);
                valRecall = mean(valRecall);
                valF1 = mean(valF1);

                if valF1 > bestF1
                    bestF1 = valF1;
                    bestCoeffs = coeffSel;
                    bestCenters = centers;
                end
                % [labels,overwrites] = labelCenters(distanceMethod,dataCell{fileIdx}.spike_class,centers,trainingFeatures_temp,dataCell{fileIdx}.trainingIdx);
                % train_realAssign = zeros(1,length(clAssign_train));
                % for i=1:length(clAssign_train)
                %     train_realAssign(i) = labels(clAssign_train(i));
                % end
                % [fScore_train,incorrect] = evalFScore_v3(dataCell{fileIdx}.spike_class,train_realAssign,dataCell{fileIdx}.trainingIdx);            
                % 
                % unseen_realAssign = zeros(1,length(clAssign_unseen));
                % for i=1:length(clAssign_unseen)
                %     unseen_realAssign(i) = labels(clAssign_unseen(i));
                % end
                % [fScore_unseen,incorrect] = evalFScore_v3(dataCell{fileIdx}.spike_class,unseen_realAssign,dataCell{fileIdx}.classIdx);            
           end %selIdx

            %% Visualization
            if showPlot && numComponents == 2
                hues = linspace(0,1,numGroups+1); hues(end) = [];  % remove redundant last hue = 1
                classColor = zeros(numGroups,3);
                for colorIdx=1:numGroups
                    classColor(colorIdx,:) = hsv2rgb([hues(colorIdx),1,1]); 
                end

                plot(trainingFeatures(:,bestCoeffs(1)),trainingFeatures(:,bestCoeffs(2)),'o','MarkerFaceColor',classColor(1,:))
                hold on
                %plot(classificationFeatures(:,1),classificationFeatures(:,2),'og')
                plot(bestCenters(:,1),bestCenters(:,2),'o','MarkerFaceColor',classColor(2,:));
                hold off
            end

            %% Store Results
            allFScoreUnseen(trainVal,runNum,fileIdx) = bestF1;
                
        end %runNum
    end %trainVal
end %fileIdx

avgFScoreUnseen = zeros(numFiles,ntrainingSize);
stdFScoreUnseen = zeros(numFiles,ntrainingSize);
for i=1:numFiles
    avgFScoreUnseen(i,:) = sum(allFScoreUnseen(:,:,i),2)'./numIters.*100;
    stdFScoreUnseen(i,:) = std(allFScoreUnseen(:,:,i),0,2)';
    
    disp(baseName(i));
    results = table(trainingPortion(i,:)', round(avgFScoreUnseen(i,:),2)', round(stdFScoreUnseen(i,:),2)', ...
        'VariableNames', {'Training%','Unseen:F-Score(All)','St-Dev:F-Score(All)'});
    disp(results);
end
save(strcat(prefix,"_kmeans.mat"),"baseName","allFScoreUnseen", ...
   "avgFScoreUnseen","stdFScoreUnseen");
end