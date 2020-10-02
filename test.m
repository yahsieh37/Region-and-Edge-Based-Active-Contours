clear;

exps = 'testconstbv4';  % experiment name (folder name to load the segmentation mask)
out_dir = 'testconstbv4';  % folder name save the evaluation results
if ~exist(fullfile('./eval_results', out_dir), 'dir')
   mkdir(fullfile('./eval_results', out_dir))
end

img_path = sprintf('./results/%s/*.jpg', exps);
truth_path = './test_data/gt/*.png';
img_files = dir(img_path);
truth_files = dir(truth_path);
num_img = length(img_files);

start = 1;
stop = num_img;

for i = start:1:stop
    img = imread(sprintf('./results/%s/%s', exps, img_files(i).name));
    %img = logical(rgb2gray(img));
    %img = logical(img);
    img = imbinarize(img,0.5);
    truth = imread(sprintf('./test_data/gt/%s',truth_files(i).name));
    truth = logical(truth);
    
    [score,fp,fn,colorImg]=getScore(img,truth,'ehd');
    scoresheet(i,3)=fn;
    scoresheet(i,2)=fp;
    scoresheet(i,1)=score;
    [h,w,d] = size(colorImg);
    colorImg = insertText(colorImg,[w-135 10],sprintf('FP: %0.4f\nFN: %0.4f\nscore: %0.4f',fp,fn,score));
    imwrite(colorImg,sprintf('./eval_results/%s/%s', out_dir, truth_files(i).name));
end

csvwrite(sprintf('./eval_results/%s/scores.csv', out_dir),scoresheet);
avgScore = mean(scoresheet(:,1));
display(avgScore);

avgFP = mean(scoresheet(:,2));
display(avgFP);
avgFN = mean(scoresheet(:,3));
display(avgFN);