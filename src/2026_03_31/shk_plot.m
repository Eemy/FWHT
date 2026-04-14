minComponents = 2;
maxComponents = 10;

baseName = ["C_Easy1_noise005","C_Easy1_noise01","C_Easy1_noise015","C_Easy1_noise02", ...
            "C_Easy1_noise025","C_Easy1_noise03","C_Easy1_noise035","C_Easy1_noise04", ...
            "C_Easy2_noise005","C_Easy2_noise01","C_Easy2_noise015","C_Easy2_noise02", ...
            "C_Difficult1_noise005","C_Difficult1_noise01","C_Difficult1_noise015","C_Difficult1_noise02", ...
            "C_Difficult2_noise005","C_Difficult2_noise01","C_Difficult2_noise015","C_Difficult2_noise02"
            ];

numComponents = maxComponents-minComponents+1;
numFiles = length(baseName);
feMethods = ["HT","WHT"];
file = '../../shk_compressed.xlsx';

%FScore_results = zeros(length(feMethods),numComponents,numFiles);

CHT = readmatrix(file,'Sheet','tranposed','Range','B2:J21');
WHT = readmatrix(file,'Sheet','tranposed','Range','B23:J42');

%% Make plots
hues = linspace(0,1,length(trainSizes)+1); hues(end) = [];  % remove redundant last hue = 1
colors = zeros(length(trainSizes),3);
for colorIdx=1:length(trainSizes)
    colors(colorIdx,:) = hsv2rgb([hues(colorIdx),1,0.8]); 
end

for fileIdx = 1:numFiles
    subplot(5,4,fileIdx)
    colororder(colors);

    FScore_results = [CHT(fileIdx,:)*100 ; WHT(fileIdx,:)*100];
    hb = bar(FScore_results);
    set(gca,'xticklabel',feMethods);
    ylabel("F-Score","FontSize",10);
    ylim([55,100]);
    title(baseName(fileIdx));
end
sgtitle("Sliding Consecutive Coefficients");