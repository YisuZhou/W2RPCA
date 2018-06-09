
clear;
clc;

%======================giraffe1=======================
Original_image_dir  =    'giraffe1/';
Noisy_image_dir='noisy20-40-30/';
fpath = fullfile(Original_image_dir, '*.jpg');
im_dir  = dir(fpath);
im_num = length(im_dir); 
nSig = [20 40 30];
Par.nSig = nSig;				%noise level

% parameters for denoising
Par.win          =   20;                                   % Non-local patch searching window
Par.Innerloop    =   1;                            % InnerLoop Num of between re-blockmatching
Par.ps           =   6;                           % Patch size
Par.step         =   1;                             % The step between neighbor patches
Par.Iter         =   6;                            
Par.display = true;

% this parameter is not finally determined yet
Par.lambda = 1.25;                   
% iterative regularization parameter
Par.delta = 0.0001; 

% record all the results in each iteration  
Par.PSNR = zeros(Par.Iter, im_num, 'single');
Par.SSIM = zeros(Par.Iter, im_num, 'single');

for i = 1:im_num
    Par.image = i;
    Par.nSig0 = nSig;
    Par.nlsp = 50;   % Initial Non-local Patch number
    Par.I = double( imread(fullfile(Original_image_dir, im_dir(i).name)) );
    [h, w, ch] = size(Par.I);       
	Par.nim = double(imread(fullfile(Noisy_image_dir, im_dir(i).name)));

    fprintf('%s :\n',im_dir(i).name);
    PSNR = csnr( Par.nim, Par.I, 0, 0 );
    SSIM = cal_ssim( Par.nim, Par.I, 0, 0 );
    fprintf(' initial: PSNR = %2.4f, SSIM = %2.4f \n', PSNR,SSIM);
	
    [im_out, Par] = W2RPCA_Denoising( Par.nim, Par.I, Par );

    im_out(im_out>255)=255;
    im_out(im_out<0)=0;
	
    % calculate the PSNR
    Par.PSNR(Par.Iter, Par.image)  =   csnr( im_out, Par.I, 0, 0 );
    Par.SSIM(Par.Iter, Par.image)      =  cal_ssim( im_out, Par.I, 0, 0 );
	
    imname = sprintf([ 'W2RPCA_' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '_' im_dir(i).name]);
    imwrite(im_out/255, imname);
	
end

