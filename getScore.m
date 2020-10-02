function [score,fp,fn,colorImg] = getScore(img,truth_img,method)
switch(method)
    case 'ehd'
        [score,fp,fn,colorImg] = SHausdorffDistance(img,truth_img);
    case 'mhd'
        [score,fp,fn,colorImg] = KaulHausdorffDistance(img,truth_img);
    case 'acc'
        [score] = Accuracy(img,truth_img);
        fp=-1;
        fn =-1;
        colorImg = [];
    case 'pre'
        [score] = Precision(img,truth_img);
        fp=-1;
        fn =-1;
        colorImg = zeros(size(img,1),size(img,2),3);
        colorImg(:,:,1) = 0.5.*truth_img;
        colorImg(:,:,2) = 0.5.*truth_img;
        colorImg(:,:,3) = 0.5.*truth_img;
        tp_img = img.*truth_img;
        fp_img = img.*(1-truth_img);
        for i = 1:size(colorImg,1)
            for j = 1:size(colorImg,2)
                if(tp_img(i,j)>0)
                    colorImg(i,j,1)  =0;
                    colorImg(i,j,2)  =0;
                    colorImg(i,j,3)  =1;
                end
                if(fp_img(i,j)>0)
                    colorImg(i,j,1)  =1;
                    colorImg(i,j,2)  =0;
                    colorImg(i,j,3)  =0;
                end
            end
        end
        
end
end
%deprecated
function [score,fp,fn,colorImg] = getScoreWithImage(img,truth_img)
[score,fp,fn,colorImg] = SHausdorffDistance(img,truth_img);
end
%deprecated
function [score,FP_error,FN_error] = buffered_hausdorff(A_img, B_img)
MAX_DISTANCE = 50;
A_dist = bwdist(A_img);
if(max(max(A_dist))==0)
    A_dist(:) = MAX_DISTANCE;
end
A_dist(A_dist>MAX_DISTANCE) = MAX_DISTANCE;
B_dist = bwdist(B_img);
if(max(max(B_dist))==0)
    B_dist(:) = MAX_DISTANCE;
end
B_dist(B_dist>MAX_DISTANCE) = MAX_DISTANCE;
A_area = sum(sum(A_img));
B_area = sum(sum(B_img));
image_size = size(A_img);
if(A_area>0)
    FP_error = sum(sum((A_img.*B_dist)))/A_area;
else
    FP_error = 0;
end
if(B_area>0)
    FN_error = sum(sum((B_img.*A_dist)))/B_area;
else
    FN_error = 0;
end

FP_error;
FN_error;
bmhd = max([FP_error FN_error]);
% remove rounding errors
bmhd = min([bmhd MAX_DISTANCE]);
score = 100 - ((bmhd*100)/MAX_DISTANCE);
end
%deprecated
function [score,FP_error,FN_error] = buffered_modified_hausdorff(A_img,B_img)
MAX_DISTANCE = 150;
A_dist = bwdist(A_img);
if(max(max(A_dist))==0)
    A_dist(:) = MAX_DISTANCE;
end
A_dist(A_dist>MAX_DISTANCE) = MAX_DISTANCE;
B_dist = bwdist(B_img);
if(max(max(B_dist))==0)
    B_dist(:) = MAX_DISTANCE;
end
B_dist(B_dist>MAX_DISTANCE) = MAX_DISTANCE;
A_area = sum(sum(A_img));
B_area = sum(sum(B_img));
image_size = size(A_img);
if(A_area>0)
    FP_error = sum(sum((A_img.*B_dist)))/A_area;
else
    FP_error = 0;
end
if(B_area>0)
    FN_error = sum(sum((B_img.*A_dist)))/B_area;
else
    FN_error = 0;
end
if(A_area==0)
    FN_error = (B_area*MAX_DISTANCE)/(image_size(1)*image_size(2));
end
if(B_area==0)
    FP_error = (A_area*MAX_DISTANCE)/(image_size(1)*image_size(2));
end
FP_error
FN_error
bmhd = max([FP_error FN_error]);
% remove rounding errors
bmhd = min([bmhd MAX_DISTANCE]);
score = 100 - ((bmhd*100)/MAX_DISTANCE);
end

function [score] = Accuracy(A_img,B_img)
tp = A_img.*B_img;
tn = (1-A_img).*(1-B_img);
total_pixels = size(B_img,1)*size(B_img,2);
score = (sum(tp(:))+sum(tn(:)))/total_pixels;
end

function [score] = Precision(A_img,B_img)
tp = A_img.*B_img;
score = sum(tp(:))/sum(A_img(:));
end

function [score,fp,fn,colorImg] = SHausdorffDistance(A_img,B_img)
%% prepare maps
MIN_DISTANCE = 0;
MAX_DISTANCE = 10;
max_penalty = MAX_DISTANCE-MIN_DISTANCE;

A_dist = bwdist(A_img);
A_dist = A_dist - MIN_DISTANCE;
A_dist(A_dist<0) = 0;
A_dist(A_dist>max_penalty) = max_penalty;
A_area = sum(A_img(:));
B_dist = bwdist(B_img);
B_dist = B_dist - MIN_DISTANCE;
B_dist(B_dist<0) = 0;
B_dist(B_dist>max_penalty) = max_penalty;
B_area = sum(B_img(:));
image_size = size(A_img);

%% create color image
% create FN image with grey = ok, blue = FN
layer1 = B_img.*A_dist/max_penalty;
layer1(B_img==0)=-1;
layer1 = myColorMap(layer1,0.5,0.7); %original (layer1,0.6667,0.5)
% create FP image with white = ok, red = FP
layer2 = A_img.*B_dist/max_penalty;
layer2(A_img==0)=-1;
layer2 = myColorMap(layer2,0.8333,1); %original (layer2,0,1)

colorImg = myAddColorMaps(layer1,layer2);
%figure;imshow(A_dist,[min(A_dist(:)) max(A_dist(:))])

%figure;
%g = B_dist/max(B_dist(:))*255;
%mn=min(B_img(:))
%mx=max(B_img(:))
%imshow(g,[]);figure;
%overlayImg = single(zeros(size(B_img,1),size(B_img,2),3));
%overlayImg(:,:,2) = B_img.*255;
%overlayImg(:,:,1) = g;
%overlayImg(:,:,3) = g;
%imshow(overlayImg,[]);

%% calculate fp, fn and score
if(A_area>0)
    FP_error = sum(sum((A_img.*B_dist)))/A_area;
else
    FP_error = 0;
end
if(B_area>0)
    FN_error = sum(sum((B_img.*A_dist)))/B_area;
else
    FN_error = 0;
end
if(A_area==0)
    FN_error = (B_area*max_penalty)/(image_size(1)*image_size(2)/100);
    if(FN_error>max_penalty)
        FN_error = max_penalty;
    end
end
if(B_area==0)
    FP_error = (A_area*max_penalty)/(image_size(1)*image_size(2)/100);
    if(FP_error>max_penalty)
        FP_error = max_penalty;
    end
end

shd = max([FP_error FN_error]);
% remove rounding errors
shd = min([shd max_penalty]);
score = 100 - ((shd*100)/max_penalty);
fp = FP_error/max_penalty;
fn = FN_error/max_penalty;
end

function [score,fp,fn,colorImg] = KaulHausdorffDistance(A_img,B_img)
%% prepare maps
MAX_DISTANCE = size(B_img,2)/5;
max_penalty = MAX_DISTANCE;

A_dist = bwdist(A_img);
A_dist(A_dist>max_penalty) = max_penalty;
A_area = sum(A_img(:));
B_dist = bwdist(B_img);
B_dist(B_dist>max_penalty) = max_penalty;
B_area = sum(B_img(:));
image_size = size(A_img);

%% create color image
% create FN image with grey = ok, blue = FN
layer1 = B_img.*A_dist/max_penalty;
layer1(B_img==0)=-1;
layer1 = myColorMap(layer1,0.5,0.7);
% create FP image with white = ok, red = FP

layer2 = A_img.*B_dist/max_penalty;
layer2(A_img==0)=-1;
layer2 = myColorMap(layer2,0.8333,1);

colorImg = myAddColorMaps(layer1,layer2);
%figure;imshow(A_dist,[min(A_dist(:)) max(A_dist(:))])

%figure;
%g = B_dist/max(B_dist(:))*255;
%mn=min(B_img(:))
%mx=max(B_img(:))
%imshow(g,[]);figure;
%overlayImg = single(zeros(size(B_img,1),size(B_img,2),3));
%overlayImg(:,:,2) = B_img.*255;
%overlayImg(:,:,1) = g;
%overlayImg(:,:,3) = g;
%imshow(overlayImg,[]);

%% calculate fp, fn and score
if(A_area>0)
    FP_error = sum(sum((A_img.*B_dist)))/A_area;
else
    FP_error = 0;
end
if(B_area>0)
    FN_error = sum(sum((B_img.*A_dist)))/B_area;
else
    FN_error = 0;
end
if(A_area==0 && B_area~=0)
    FN_error = max_penalty;
end
if(B_area==0 && A_area~=0)
    FP_error = max_penalty;
end

mhd = max([FP_error FN_error]);
% remove rounding errors
mhd = min([mhd max_penalty]);
score = 100 - ((mhd*100)/max_penalty);
fp = FP_error/max_penalty;
fn = FN_error/max_penalty;
end

function [colorImg] = myColorMap(grayImg, hue,value)
sizeImg = size(grayImg);
colorImg = zeros(sizeImg(1),sizeImg(2),3);
colorImg(:,:,1) = hue;
valuePane = zeros(sizeImg);
valuePane(grayImg>-1) = value;
colorImg(:,:,3) = valuePane;
grayImg(grayImg<0) = 0;
colorImg(:,:,2) = grayImg;
colorImg = hsv2rgb(colorImg);
end

function [addedImg] = myAddColorMaps(l1,l2)
addedImg = l1+l2;
addedImg(addedImg>1) = 1;
end
