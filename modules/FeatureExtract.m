classdef FeatureExtract
    properties(Constant)
        % samplingInterval = 0.0417/1000;
        samplingInterval = 0.0417;
    end %end properties

    methods(Static)
        %% UTILITY METHODS
        function [trainingIdx,classIdx] = splitData(trainingPortion,numSpikes)
        %Split Spike Indices into Training and Classification Portions
            trainingIdx = sort(randperm(numSpikes,round((trainingPortion/100)*numSpikes)));
            classIdx = setxor(trainingIdx,1:numSpikes); %remaining of data
        end %end splitData

        function [featureVector_norm,featureMax,featureMin] = normalizeFeatures(featureVector)
        %Normalize each feature between 0 and 1, where min=0, max=1
            featureVector_norm = zeros(size(featureVector));
            featureMax = zeros(1,size(featureVector,2));
            featureMin = zeros(1,size(featureVector,2));
            for i=1:size(featureVector,2)
                featureMax(i) = max(featureVector(:,i));
                featureMin(i) = min(featureVector(:,i));
                
                for j=1:size(featureVector,1)
                    featureVector_norm(j,i) = (featureVector(j,i)-featureMin(i))/(featureMax(i)-featureMin(i));
                end
            end
        end %end normalizeFeatures

        %% FEATURE EXTRACTION METHODS
        function [score,fSpace] = zcf(spikeVecs,trainingSetIdx,classificationSetIdx)
            numComponents = 2; %zc1 and zc2
            numSpikes = size(spikeVecs,1);
            vectorSize = size(spikeVecs,2);

            score = zeros(numSpikes,numComponents);
            for i=1:length(trainingSetIdx)
                spikeIdx = trainingSetIdx(i);
                
                regionNum = 1;
                A = zeros(1,100);
                A_idx = zeros(1,100);

                for j=1:vectorSize
                    if j>1 && (spikeVecs(spikeIdx,j)*spikeVecs(spikeIdx,j-1) < 0)
                        regionNum = regionNum+1;
                        A_idx(regionNum) = j;
                    end
                    A(regionNum) = A(regionNum) + spikeVecs(spikeIdx,j);
                end

                [maxArea,idx_max] = max(A);
                [minArea,idx_min] = min(A);

                if idx_max < idx_min
                    score(i,1) = maxArea;
                    score(i,2) = minArea;
                else
                    score(i,1) = minArea;
                    score(i,2) = maxArea;
                end

                %disp("Max Area: " + maxArea + " ~Min Area: " + minArea);
                %disp("Max Idx: " + A_idx(idx_max) + " ~Min Idx: " + A_idx(idx_min));
                %disp("Num-ZC: " + regionNum)
            end

            fSpace = length(classificationSetIdx);
        end

        function [score,fSpace] = wavelet(spikeVecs,numComponents,trainingSetIdx,classificationSetIdx)
            vectorSize = size(spikeVecs,2);

            %Feature Extraction on Training Data
            score_draft = zeros(length(trainingSetIdx),vectorSize);
            for i=1:length(trainingSetIdx)
                spikeIdx = trainingSetIdx(i);
                [score_draft(i,:),~] = wavedec(spikeVecs(spikeIdx,:),4,"haar");
            end
            
            % KS Normality Deviation Test
            normalityDev = zeros(1,vectorSize);
            for coeff = 1:vectorSize
                coeffDistr = score_draft(:,coeff);
                mu = mean(coeffDistr);
                sigma = std(coeffDistr);

                [f,x] = ecdf(coeffDistr);
                g = cdf("Normal",x,mu,sigma);
                normalityDev(coeff) = max(abs(f-g));
            end
            [~,indices] = maxk(normalityDev,numComponents);
            
            % Store coeffs with largest deviation from normality
            score = zeros(length(trainingSetIdx),numComponents);
            for i = 1:numComponents
                score(:,i) = score_draft(:,indices(i));
            end            
            [score,featureMax,featureMin] = FeatureExtract.normalizeFeatures(score);
            
            %Convert Classification Data into Feature Space
            fSpace = zeros(length(classificationSetIdx),numComponents);
            for i=1:length(classificationSetIdx)
                spikeIdx = classificationSetIdx(i);
                [fSpace_draft,~] = wavedec(spikeVecs(spikeIdx,:),4,"haar");
                for j=1:numComponents
                    fSpace(i,j) = (fSpace_draft(indices(j))-featureMin(j))/(featureMax(j)-featureMin(j));
                end
            end

        end

        function [score,fSpace] = FWHT(spikeVecs,numComponents,trainingSetIdx,classificationSetIdx)
            vectorSize = size(spikeVecs,2);

            %Feature Extraction on Training Data
            score_draft = zeros(length(trainingSetIdx),vectorSize);
            for i=1:length(trainingSetIdx)
                spikeIdx = trainingSetIdx(i);
                score_draft(i,:) = fwht(spikeVecs(spikeIdx,:),vectorSize,'dyadic');
            end
            figure;
            plot(score_draft');

            % % KS Normality Deviation Test
            % normalityDev = zeros(1,vectorSize);
            % for coeff = 1:vectorSize
            %     coeffDistr = score_draft(:,coeff);
            %     mu = mean(coeffDistr);
            %     sigma = std(coeffDistr);
            % 
            %     [f,x] = ecdf(coeffDistr);
            %     g = cdf("Normal",x,mu,sigma);
            %     normalityDev(coeff) = max(abs(f-g));
            % end
            % [~,indices] = maxk(normalityDev,numComponents);
            % 
            % disp(indices)
            % hold on
            % for i=1:length(indices)
            %     xline(indices(i));
            % end
            % hold off
            % 
            % figure;
            % for i=1:length(indices)
            %     subplot(5,2,i)
            %     histogram(score_draft(:,i))
            % end
            % 
            % % Store coeffs with largest deviation from normality
            % score = zeros(length(trainingSetIdx),numComponents);
            % for i = 1:numComponents
            %     score(:,i) = score_draft(:,indices(i));
            % end
            
            % Store continuous coeffs
            minCoeff = 50;
            maxCoeff = minCoeff+numComponents-1;
            score = score_draft(:,minCoeff:maxCoeff);
            [score,featureMax,featureMin] = FeatureExtract.normalizeFeatures(score);
            
            %Convert Classification Data into Feature Space
            fSpace = zeros(length(classificationSetIdx),numComponents);
            for i=1:length(classificationSetIdx)
                spikeIdx = classificationSetIdx(i);
                fSpace_draft = fwht(spikeVecs(spikeIdx,:));
                % for j=1:numComponents
                %     fSpace(i,j) = (fSpace_draft(indices(j))-featureMin(j))/(featureMax(j)-featureMin(j));
                % end
                for j=1:numComponents
                    fSpace(i,j) = (fSpace_draft(minCoeff+j-1)-featureMin(j))/(featureMax(j)-featureMin(j));
                end
            end
        end

        function [score,fSpace] = FSDE(spikeVecs,numComponents,trainingSetIdx,classificationSetIdx)
        %Computes first and second numerical derivatives and extracts
        %select extrema from them. Returns separate, normalized feature matrices for how the
        %data is split. 
            
            %Feature Extraction on Training Data: FSDE
            %spikeVecs = sgolayfilt(spikeVecs,3,9);

            score = zeros(length(trainingSetIdx),numComponents);
            %d_data_dt = num_deriv(1,FeatureExtract.samplingInterval,spikeVecs);
            %d2_data_dt2 = num_deriv(2,FeatureExtract.samplingInterval,spikeVecs);
            d_data_dt = diff(spikeVecs,1,2);
            d2_data_dt2 = diff(d_data_dt,1,2);

            for i = 1:length(trainingSetIdx)
                spikeIdx = trainingSetIdx(i);
                if numComponents == 3
                    [score(i,1),~] = max(d_data_dt(spikeIdx,:));
                    [score(i,2),~] = min(d2_data_dt2(spikeIdx,:));
                    [score(i,3),~] = max(d2_data_dt2(spikeIdx,:));
                elseif numComponents == 2
                    [score(i,1),~] = max(d_data_dt(spikeIdx,:));
                    [score(i,2),~] = min(d2_data_dt2(spikeIdx,:));
                elseif numComponents == 1
                    [score(i,1),~] = min(d2_data_dt2(spikeIdx,:));               
                end
            end
            [score,featureMax,featureMin] = FeatureExtract.normalizeFeatures(score);
            
            %Convert Classification Data into Feature Space
            fSpace = zeros(length(classificationSetIdx),numComponents);
            for i = 1:length(classificationSetIdx)
                spikeIdx = classificationSetIdx(i);
                if numComponents == 3
                    [fSpace(i,1),~] = max(d_data_dt(spikeIdx,:));
                    [fSpace(i,2),~] = min(d2_data_dt2(spikeIdx,:));
                    [fSpace(i,3),~] = max(d2_data_dt2(spikeIdx,:));
                elseif numComponents == 2
                    %[fSpace(i,1),~] = min(d2_data_dt2(spikeIdx,:));
                    %[fSpace(i,2),~] = max(d2_data_dt2(spikeIdx,:));
                    
                    [fSpace(i,1),~] = max(d_data_dt(spikeIdx,:));
                    [fSpace(i,2),~] = min(d2_data_dt2(spikeIdx,:));
                elseif numComponents == 1
                    [fSpace(i,1),~] = min(d2_data_dt2(spikeIdx,:));               
                end

                for j = 1:numComponents
                    fSpace(i,j) = (fSpace(i,j)-featureMin(j))/(featureMax(j)-featureMin(j));
                end
            end
        end %end FSDE

        function [score,fSpace] = PCA(spikeVecs,numComponents,trainingSetIdx,classificationSetIdx)
        %Performs PCA on raw data, or first, or second derivative of the
        %data. Returns number of specified components.
            vectorSize = size(spikeVecs,2);

            %Feature Extraction on Training Data
            trainingSpikeVecs = zeros(length(trainingSetIdx),vectorSize);
            for i=1:length(trainingSetIdx)
                for j=1:vectorSize
                    trainingSpikeVecs(i,j) = spikeVecs(trainingSetIdx(i),j);
                end
            end
            [coeff,score,~,~,~,mu] = pca(trainingSpikeVecs,'NumComponents',numComponents);
            [score,featureMax,featureMin] = FeatureExtract.normalizeFeatures(score);
            
            %Convert Classification Data into Feature Space
            fSpace = zeros(length(classificationSetIdx),numComponents);
            for i=1:length(classificationSetIdx)
                for j=1:numComponents
                    for k=1:vectorSize
                        fSpace(i,j) = fSpace(i,j)+(spikeVecs(classificationSetIdx(i),k)-mu(k))*coeff(k,j);
                    end

                    fSpace(i,j) = (fSpace(i,j)-featureMin(j))/(featureMax(j)-featureMin(j));
                end
            end
        end %end PCA

        function [score,fSpace] = getFeatureVecs(feMethod,spikeVecs,numComponents,trainingSetIdx,classificationSetIdx)
            switch feMethod
                case "FSDE"
                    [score,fSpace] = FeatureExtract.FSDE(spikeVecs,numComponents,trainingSetIdx,classificationSetIdx);
                case "PCA"
                    [score,fSpace] = FeatureExtract.PCA(spikeVecs,numComponents,trainingSetIdx,classificationSetIdx);
                case "PCA_FD"
                    spikeVecs = num_deriv(1,FeatureExtract.samplingInterval,spikeVecs);
                    [score,fSpace] = FeatureExtract.PCA(spikeVecs,numComponents,trainingSetIdx,classificationSetIdx);
                case "wavelet"
                    [score,fSpace] = FeatureExtract.wavelet(spikeVecs,numComponents,trainingSetIdx,classificationSetIdx);
                case "FWHT"
                    [score,fSpace] = FeatureExtract.FWHT(spikeVecs,numComponents,trainingSetIdx,classificationSetIdx);
                case "ZCF"
                    [score,fSpace] = FeatureExtract.zcf(spikeVecs,trainingSetIdx,classificationSetIdx);
            end
        end

    end %end methods
end %end classdef