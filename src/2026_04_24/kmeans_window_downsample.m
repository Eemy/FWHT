clear
addpath("../../funcs/","../../modules")
path = "../../../SpikeTraining_RISCV/Raw_Data/";

%% Set up parameters for run
baseName = ["C_Easy1_noise005","C_Easy1_noise01","C_Easy1_noise015","C_Easy1_noise02", ...
            "C_Easy1_noise025","C_Easy1_noise03","C_Easy1_noise035","C_Easy1_noise04", ...
            "C_Easy2_noise005","C_Easy2_noise01","C_Easy2_noise015","C_Easy2_noise02", ...
            "C_Difficult1_noise005","C_Difficult1_noise01","C_Difficult1_noise015","C_Difficult1_noise02", ...
            "C_Difficult2_noise005","C_Difficult2_noise01","C_Difficult2_noise015","C_Difficult2_noise02"
            ];

% baseName = ["C_Easy1_noise005","C_Easy1_noise01","C_Easy1_noise015","C_Easy1_noise02", ...
%              "C_Easy1_noise025","C_Easy1_noise03","C_Easy1_noise035","C_Easy1_noise04"];

% baseName = ["C_Easy2_noise005","C_Easy2_noise01","C_Easy2_noise015","C_Easy2_noise02"];

% baseName = ["C_Difficult1_noise005","C_Difficult1_noise01","C_Difficult1_noise015","C_Difficult1_noise02"];

% baseName = ["C_Difficult2_noise005","C_Difficult2_noise01","C_Difficult2_noise015","C_Difficult2_noise02"];

% baseName = ["C_Difficult1_noise02"];

numIters = 1;
minComponents = 2;
maxComponents = 10;

downsample_factor = 4;
downsample_offset = 0;
%numComponents = 2;
distanceMethod = "Euclidean";
feMethod = "HT";
%coeffOrder = "sequency"; %WHT = 'sequency', HT = 'hadamard'
showPlot = false;
saveMat = true;
dir = "euc_ht_random_train06_down4/";

filesInvolved = "all";

%% Load data
classPortions = [0.6];

% classCounts = [10,10,10 ; ...
%                 20,20,20 ; ...
%                 30,30,30 ; ...
%                 40,40,40 ; ...
%                 50,50,50 ; ...
%                 100,100,100; ...
%                 200,200,200; ...
%                 300,300,300; ...
%                 400,400,400; ...
%                 700,700,700];
% classCounts = [700,700,700]; % one row for each data file
numGroups = 3;
startOffset = int32(0);
vectorSize = int32(64);
noise = int32(0);

%Create Dataset objects for each file
numSpikes = 0;
numFiles = numel(baseName);
dataCell = cell(1,numFiles);
for fileIdx = 1:numFiles
    dataCell{fileIdx} = Dataset(path,baseName(fileIdx),vectorSize,startOffset,noise);
    numSpikes = numSpikes+dataCell{fileIdx}.numSpikes;
end

%ntrainingSize = size(classPortions,1);
ntrainingSize = length(classPortions);
trainingPortion = zeros(numFiles,ntrainingSize);
classCounts = zeros(ntrainingSize,numGroups,numFiles);
for i=1:numFiles
    for j=1:ntrainingSize
        classCounts(j,:,i) = floor(dataCell{i}.numSpikes*classPortions(j)/numGroups);
        %trainingPortion(i,j) = sum(classCounts(j,:))/dataCell{i}.numSpikes*100;
        trainingPortion(i,j) = sum(classCounts(j,:,i));
    end
end

%% Perform Feature Extraction and Clustering
for numComponents = minComponents:maxComponents
prefix = strcat(feMethod+numComponents,"_slidingWindow_",filesInvolved);
allFScoreUnseen = zeros(ntrainingSize,numIters,numFiles); %store results

for fileIdx = 1:numFiles
    for trainVal = 1:ntrainingSize
        for runNum = 1:numIters
            dataCell{fileIdx}.splitDataCounts(classCounts(trainVal,:,fileIdx));
            spikeVecs = dataCell{fileIdx}.downsample(downsample_factor,downsample_offset);
            [trainingFeatures,classificationFeatures] = FeatureExtract.getFeatureVecs(feMethod,spikeVecs,numComponents,dataCell{fileIdx}.trainingIdx,dataCell{fileIdx}.classIdx);
            % figure;
            % plot(spikeVecs');

            % % Train/test split (to match Meng script)
            % rng(1, 'twister') % For repeatable result
            % r = dataCell{fileIdx}.numSpikes;
            % splt = 0.6;
            % [trainInd, valInd, testInd] = dividerand(r, splt, 0.8 - splt, 0.2);
            % [trainingFeatures,classificationFeatures] = FeatureExtract.getFeatureVecs(feMethod,dataCell{fileIdx}.spikeVecs,numComponents,trainInd,valInd);

            bestCoeffs = [];
            bestCenters = [];
            bestF1 = 0;
            bestAlignMat = [];
            bestReorder = [];
            vectorSize = size(spikeVecs,2);
            allFScore = zeros(1,vectorSize);
            for selIdx = 1:vectorSize
                %% Coefficient Selection: sliding window
                %Store continuous coeffs
                minCoeff = selIdx;
                maxCoeff = minCoeff+numComponents-1;
                coeffSel = mod(minCoeff:maxCoeff,vectorSize); coeffSel(coeffSel==0) = vectorSize;
                trainingFeatures_temp = trainingFeatures(:,coeffSel);
                classificationFeatures_temp = classificationFeatures(:,coeffSel);
                
                %% Perform Clustering (K-Means) -- maybe FSA in the future
                distArg1 = "sqeuclidean";
                distArg2 = "euclidean";
                switch distanceMethod
                    case "Manhattan"
                        distArg1 = "cityblock";
                        distArg2 = "cityblock";
                    case "Euclidean"
                        distArg1 = "sqeuclidean";
                        distArg = "euclidean";
                end

                %opts = statset('Display','final');
                %[clAssign_train,centers] = kmeans(trainingFeatures_temp,numGroups,'Distance','cityblock','Replicates',5,'Options',opts);
                [clAssign_train,centers] = kmeans(trainingFeatures_temp,numGroups,'Distance',distArg1,'Replicates',5);
                [~,clAssign_unseen] = pdist2(centers,classificationFeatures_temp,distArg2,'Smallest',1);
                
                %% Obtain Confusion Matrix
                trueLabels = dataCell{fileIdx}.spike_class(dataCell{fileIdx}.classIdx);
                %trueLabels = dataCell{fileIdx}.spike_class(valInd);
                alignMat = confusionmat(trueLabels,clAssign_unseen');
                
                % [Reorder,~] = labelCenters(distanceMethod,dataCell{fileIdx}.spike_class,centers,trainingFeatures_temp,dataCell{fileIdx}.trainingIdx);
                % truePositiveP(1) = alignMat(1,Reorder(1));
                % truePositiveP(2) = alignMat(2,Reorder(2));
                % truePositiveP(3) = alignMat(3,Reorder(3));
                % 
                % valPrecision = [truePositiveP(1)/sum(alignMat(:,Reorder(1))); ...
                %                 truePositiveP(2)/sum(alignMat(:,Reorder(2))); ...
                %                 truePositiveP(3)/sum(alignMat(:,Reorder(3)))];
                % valRecall = [truePositiveP(1)/sum(alignMat(1,:)); ...
                %                 truePositiveP(2)/sum(alignMat(2,:)); ...
                %                 truePositiveP(3)/sum(alignMat(3,:))];
                
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
                allFScore(selIdx) = valF1;
                if valF1 > bestF1
                    bestF1 = valF1;
                    bestCoeffs = coeffSel;
                    bestCenters = centers;
                    bestAlignMat = alignMat;
                    bestReorder = Reorder;
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

                figure;
                subplot(2,1,1)
                plot(classificationFeatures(:,bestCoeffs(1)),classificationFeatures(:,bestCoeffs(2)),'o','MarkerFaceColor','k');
                hold on
                % startIdx = 1;
                % hold on
                % for groupIdx = 1:numGroups
                %     endIdx = startIdx+dataCell{fileIdx}.spikeCounts(groupIdx)-1-classCounts(trainVal,groupIdx);
                %     plot(classificationFeatures(startIdx:endIdx,bestCoeffs(1)),classificationFeatures(startIdx:endIdx,bestCoeffs(2)),'o','MarkerFaceColor',classColor(groupIdx,:));
                %     startIdx = endIdx+1;
                % end

                for groupIdx = 1:numGroups
                    plot(bestCenters(groupIdx,1),bestCenters(groupIdx,2),'o','MarkerFaceColor',classColor(bestReorder(groupIdx),:),'MarkerEdgeColor','k','MarkerSize',10);
                end
                title("Unseen Data");
                hold off
                
                subplot(2,1,2)
                plot(trainingFeatures(:,bestCoeffs(1)),trainingFeatures(:,bestCoeffs(2)),'o','MarkerFaceColor','k');
                hold on
                % startIdx = 1;
                % hold on
                % for groupIdx = 1:numGroups
                %     endIdx = startIdx+classCounts(trainVal,groupIdx)-1;
                %     plot(trainingFeatures(startIdx:endIdx,bestCoeffs(1)),trainingFeatures(startIdx:endIdx,bestCoeffs(2)),'o','MarkerFaceColor',classColor(groupIdx,:));
                %     startIdx = endIdx+1;
                % end

                for groupIdx = 1:numGroups
                    plot(bestCenters(groupIdx,1),bestCenters(groupIdx,2),'o','MarkerFaceColor',classColor(bestReorder(groupIdx),:),'MarkerEdgeColor','k','MarkerSize',10);
                end
                title("Training Data");
                hold off

                
            end

            %% Store Results
            allFScoreUnseen(trainVal,runNum,fileIdx) = bestF1;
                
        end %runNum
    end %trainVal
    disp(baseName(fileIdx));
    disp("Best Coefficients: ")
    disp(bestCoeffs);
end %fileIdx

    avgFScoreUnseen = zeros(numFiles,ntrainingSize);
    stdFScoreUnseen = zeros(numFiles,ntrainingSize);
    for i=1:numFiles
        avgFScoreUnseen(i,:) = sum(allFScoreUnseen(:,:,i),2)'./numIters.*100;
        stdFScoreUnseen(i,:) = std(allFScoreUnseen(:,:,i),0,2)'.*100;
        
        disp(baseName(i));
        results = table(trainingPortion(i,:)', round(avgFScoreUnseen(i,:),2)', round(stdFScoreUnseen(i,:),2)', ...
            'VariableNames', {'Train N','Unseen:F-Score(All)','St-Dev:F-Score(All)'});
        disp(results);
    end
    if saveMat
        save(strcat(dir,prefix,"_",distanceMethod,"_kmeans.mat"),"baseName","allFScoreUnseen", ...
           "avgFScoreUnseen","stdFScoreUnseen");
    end
end %numComponents