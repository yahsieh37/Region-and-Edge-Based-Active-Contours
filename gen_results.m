clear;
img_path = './test_data/range/*.jpg';  % path to load images
seg_path = './test_data/fcn_resnet50/*.jpg';  % path to load segmentation masks
img_files = dir(img_path);
seg_files = dir(seg_path);
num_img = length(img_files);

exps = 'testconstbv4';  % experiment name (folder name to save the results)
if ~exist(fullfile('./results', exps), 'dir')
   mkdir(fullfile('./results', exps))
end

start = 1;
stop = num_img;

for i = start:1:stop
    img_file = sprintf('./test_data/range/%s',img_files(i).name);
    seg_file = sprintf('./test_data/fcn_resnet50/%s',seg_files(i).name);
    
    seg = REAC_func(img_file,seg_file, 500, 20, -0.01, -0.05, 0.7); % max_its, thresh, a, b, R_bias
    
    imwrite(seg, fullfile('./results', exps, sprintf('%s',seg_files(i).name)));
end
