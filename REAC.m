clear;

%% Parameters
max_its = 500;  % max iterations of curve evolution
thresh = 20;  % stopping criterion (change of curve) of curve evolution
a = -0.01;  % parameter for region-based regularization
b = 0.01;  % parameter for edge-based regularization
R_bias = 0.7;  % bias between region- and edge-based level sets

%% Image and segmentation mask preparation
img = imread('test_data/range/range_LcmsResult_ImageRng_000150_2.jpg');
img_ori = im2double(img);

seg = imread('test_data/fcn_resnet50/result_LcmsResult_ImageRng_000150_2.jpg');
seg = rgb2gray(seg);
seg = im2double(seg);
seg = imbinarize(seg,0.5);
seg = double(seg);
init_mask = seg;
phi = mask2phi(init_mask);

%% Extract gradient feature to guide the active contour (for edge-based active contour)
% Specific the type of preprocessed image to extract gradient feature from
method_f = "anisodiff";  % Specify the name of preprocessing method used
if method_f == "Gaussian"  % Gaussian smoothing
    sigma = 1;
    feature = imgaussfilt(img, sigma, 'FilterSize', 3);
elseif method_f == "contrast"  % contrast enhancement
    edgeThreshold = 0.4;
    amount = 0.5;
    feature = localcontrast(img, edgeThreshold, amount);
elseif method_f == "anisodiff"  % anisotropic diffusion
    feature = imdiffusefilt(img);
elseif method_f == "diffcontrast"  % anisotropic diffusion + contrast enhancement
    feature = imdiffusefilt(img);
    edgeThreshold = 0.4;
    amount = 0.5;
    feature = localcontrast(feature, edgeThreshold, amount);
end

feature = im2double(feature) .* 255;
%feature = im2double(feature);
[FX,FY] = gradient(feature);
feature = sqrt(FX.^2+FY.^2+eps);

%-- Method1: Increase gradient in prediction area
pred_crack = find(init_mask==1);
feature(pred_crack) = feature(pred_crack)*10;

%-- Method2: Gradient dot with gradient of initial phi
% [FXphi,FYphi] = gradient(phi);
% feature = (FX.*(-FXphi)+FY.*(-FYphi)+eps);
% feature = feature ./max(abs(feature));

%-- Method3: Combine Method1 and Method2
% pred_crack = find(init_mask==1);
% [FXphi,FYphi] = gradient(phi);
% temp = (FX.*(-FXphi)+FY.*(-FYphi)+eps);
% feature(pred_crack) = temp(pred_crack)*10;
% feature = feature ./max(abs(feature));

feature = 1 ./ ( 1 + feature.^2 );

%% plot gradient (for checking only)
% [dimy, dimx, c] = size(img);
% [X,Y] = meshgrid(1:dimx,1:dimy);
% 
% figure
% contour(X,Y,feature)
% hold on
% %quiver(X,Y,FXphi,FYphi)
% colorbar;
% hold off

%% Image preprocessing (for region-based active contour)
method = "Gaussian";  % Specify the name of preprocessing method used
if method == "Gaussian"  % Gaussian smoothing
    sigma = 1;
    img = imgaussfilt(img, sigma, 'FilterSize', 3);
elseif method == "contrast"  % contrast enhancement
    edgeThreshold = 0.4;
    amount = 0.5;
    img = localcontrast(img, edgeThreshold, amount);
elseif method == "anisodiff"  % anisotropic diffusion
    img = imdiffusefilt(img);
elseif method == "diffgaussian"  % anisotropic diffusion + Gaussian smoothing
    img = imdiffusefilt(img);
    sigma = 1;
    img = imgaussfilt(img, sigma, 'FilterSize', 3);
elseif method == "diffcontrast"  % anisotropic diffusion + contrast enhancement
    img = imdiffusefilt(img);
    edgeThreshold = 0.4;
    amount = 0.5;
    img = localcontrast(img, edgeThreshold, amount);
end
img = im2double(img);

%% Curve evolution
its = 0;      stop = 0;
prev_mask = init_mask;        c = 0;

while ((its < max_its) && ~stop)
    idx = find(phi <= 1.2 & phi >= -1.2);  %-- get the curve's narrow band
    
    if ~isempty(idx)
        if(mod(its,50) == 0) 
            showCurveAndPhi(img_ori,phi,its);  
        end

        %-- find interior and exterior mean
        fea = feature(idx);
        upts = find(phi<=0);                 % interior points
        vpts = find(phi>0);                  % exterior points
        u = sum(img(upts))/(length(upts)+eps); % interior mean
        v = sum(img(vpts))/(length(vpts)+eps); % exterior mean
        F = (img(idx)-u).^2-(img(idx)-v).^2;     % force from image information
        %F = F .* (fea*0.5);
        
        %-- energy function
        [curvature,normGrad,FdotGrad] = get_evolution_functions(phi,feature,idx);
        
        %-- gradient descent to minimize energy
        dphidt1 = fea.*curvature.*normGrad;
        dphidt1 = dphidt1./max(abs(dphidt1(:)));

        dphidt2 = FdotGrad;
        dphidt2 = dphidt2./max(abs(dphidt2(:)));

        dphidt3 = fea.*normGrad;
        %dphidt3 = normGrad;
        dphidt3 = dphidt3./max(abs(dphidt3(:)));

        dphidt = R_bias*normGrad.*(F./max(abs(F))) + (1-R_bias)*(dphidt1 + dphidt2) + a*normGrad + b*normGrad.*curvature;
        
        %-- maintain the CFL condition
        %dt = .45/(max(abs(dphidt))+eps);  % delta_x = 1(pixel)?
        dt = .45;

        %-- evolve the curve
        phi(idx) = phi(idx) + dt.*dphidt;

        %-- Keep SDF smooth
        phi = sussman(phi, .5);

        new_mask = phi<=0;
        c = convergence(prev_mask,new_mask,thresh,c);
        if c <= 5
            its = its + 1;
            prev_mask = new_mask;
        else
            stop = 1;
        end  
    else
        break;
    end
end

showCurveAndPhi(img_ori,phi,max_its);
%-- make mask from SDF
seg = phi<=0; %-- Get mask from levelset

%% Functions
function img = im2graydouble(img)
    [dimy, dimx, c] = size(img);
    if (isfloat(img))
        if (c==3) 
            img = rgb2gray(uint8(img)); 
        end
    else
        if (c==3) 
            img = rgb2gray(img); 
        end
        img = im2double(img);
    end
end

function phi = mask2phi(init_a)
    phi=bwdist(init_a)-bwdist(1-init_a)+im2double(init_a)-.5;
end

function showCurveAndPhi(I, phi, i)
    imshow(I,'initialmagnification',800,'displayrange',[0 1]); hold on;
    contour(phi, [0 0], 'g','LineWidth',4);
    contour(phi, [0 0], 'k','LineWidth',2);
    hold off; title([num2str(i) ' Iterations']); drawnow;
end

%-- compute curvature along SDF
function [curvature,normGrad,FdotGrad] = get_evolution_functions(phi,feature,idx)
    [dimy, dimx] = size(phi);        
    [y x] = ind2sub([dimy,dimx],idx);  % get subscripts
    %-- get subscripts of neighbors
    ym1 = y-1; xm1 = x-1; yp1 = y+1; xp1 = x+1;
    %-- bounds checking  
    ym1(ym1<1) = 1; xm1(xm1<1) = 1;              
    yp1(yp1>dimy)=dimy; xp1(xp1>dimx) = dimx;    
    %-- get indexes for 8 neighbors
    idup = sub2ind(size(phi),yp1,x);    
    iddn = sub2ind(size(phi),ym1,x);
    idlt = sub2ind(size(phi),y,xm1);
    idrt = sub2ind(size(phi),y,xp1);
    idul = sub2ind(size(phi),yp1,xm1);
    idur = sub2ind(size(phi),yp1,xp1);
    iddl = sub2ind(size(phi),ym1,xm1);
    iddr = sub2ind(size(phi),ym1,xp1);
    
    %-- get central derivatives of SDF at x,y
    phi_x  = (-phi(idlt)+phi(idrt))/2;
    phi_y  = (-phi(iddn)+phi(idup))/2;
    phi_xx = phi(idlt)-2*phi(idx)+phi(idrt);
    phi_yy = phi(iddn)-2*phi(idx)+phi(idup);
    phi_xy = 0.25*phi(iddl)+0.25*phi(idur)...
             -0.25*phi(iddr)-0.25*phi(idul);
    phi_x2 = phi_x.^2;
    phi_y2 = phi_y.^2;
    
    %-- compute curvature (Kappa)
    curvature = ((phi_x2.*phi_yy + phi_y2.*phi_xx - 2*phi_x.*phi_y.*phi_xy)./...
              (phi_x2 + phi_y2 +eps).^(3/2));        
    %-- compute norm of gradient
    phi_xm = phi(idx)-phi(idlt);
    phi_xp = phi(idrt)-phi(idx);
    phi_ym = phi(idx)-phi(iddn);
    phi_yp = phi(idup)-phi(idx);    
    normGrad = sqrt( (max(phi_xm,0)).^2 + (min(phi_xp,0)).^2 + ...
        (max(phi_ym,0)).^2 + (min(phi_yp,0)).^2 );
    
    %-- compute scalar product between the feature image and the gradient of phi
    F_x = 0.5*feature(idrt)-0.5*feature(idlt);
    F_y = 0.5*feature(idup)-0.5*feature(iddn);   
    FdotGrad = (max(F_x,0)).*(phi_xp) + (min(F_x,0)).*(phi_xm) + ...
        (max(F_y,0)).*(phi_yp) + (min(F_y,0)).*(phi_ym);   
end

%-- level set re-initialization by the sussman method
function D = sussman(D, dt)
    
    % forward/backward differences
    a = D - shiftR(D); % backward
    b = shiftL(D) - D; % forward
    c = D - shiftD(D); % backward
    d = shiftU(D) - D; % forward
    a_p = a;  a_n = a; % a+ and a-
    b_p = b;  b_n = b;
    c_p = c;  c_n = c;
    d_p = d;  d_n = d;
    a_p(a < 0) = 0;
    a_n(a > 0) = 0;
    b_p(b < 0) = 0;
    b_n(b > 0) = 0;
    c_p(c < 0) = 0;
    c_n(c > 0) = 0;
    d_p(d < 0) = 0;
    d_n(d > 0) = 0;
    dD = zeros(size(D));
    D_neg_ind = find(D < 0);
    D_pos_ind = find(D > 0);
    dD(D_pos_ind) = sqrt(max(a_p(D_pos_ind).^2, b_n(D_pos_ind).^2) ...
                       + max(c_p(D_pos_ind).^2, d_n(D_pos_ind).^2)) - 1;
    dD(D_neg_ind) = sqrt(max(a_n(D_neg_ind).^2, b_p(D_neg_ind).^2) ...
                       + max(c_n(D_neg_ind).^2, d_p(D_neg_ind).^2)) - 1;
    D = D - dt .* sussman_sign(D) .* dD;
end

%-- whole matrix derivatives
function shift = shiftD(M)
    shift = shiftR(M')';
end

function shift = shiftL(M)
    shift = [ M(:,2:size(M,2)) M(:,size(M,2)) ];
end

function shift = shiftR(M)
  shift = [ M(:,1) M(:,1:size(M,2)-1) ];
end
function shift = shiftU(M)
    shift = shiftL(M')';
end
  
function S = sussman_sign(D)
    S = D ./ sqrt(D.^2 + 1);
end

function c = convergence(p_mask,n_mask,thresh,c)
    diff = p_mask - n_mask;
    n_diff = sum(abs(diff(:)));
    if n_diff < thresh
        c = c + 1;
    else c = 0;
    end
end