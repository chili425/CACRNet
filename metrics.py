import math
import cv2
import numpy as np

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.

    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def calculate_psnr(img1, img2, border=0):

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def compute_bef(img):

	block = 8
	height, width = img.shape[:2]

	H = [i for i in range(width-1)]
	H_B = [i for i in range(block-1,width-1,block)]
	H_BC = list(set(H)-set(H_B))

	V = [i for i in range(height-1)]
	V_B = [i for i in range(block-1,height-1,block)]
	V_BC = list(set(V)-set(V_B))

	D_B = 0
	D_BC = 0

	for i in H_B:
		diff = img[:,i] - img[:,i+1]
		D_B += np.sum(diff**2)

	for i in H_BC:
		diff = img[:,i] - img[:,i+1]
		D_BC += np.sum(diff**2)

	for j in V_B:
		diff = img[j,:] - img[j+1,:]
		D_B += np.sum(diff**2)

	for j in V_BC:
		diff = img[j,:] - img[j+1,:]
		D_BC += np.sum(diff**2)

	N_HB = height * (width/block - 1)
	N_HBC = height * (width - 1) - N_HB
	N_VB = width * (height/block -1)
	N_VBC = width * (height -1) - N_VB
	D_B = D_B / (N_HB + N_VB)
	D_BC = D_BC / (N_HBC + N_VBC)
	eta = math.log2(block) / math.log2(min(height, width)) if D_B > D_BC else 0
	return eta * (D_B - D_BC)

def calculate_psnrb(img1, img2, border=0):

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]
    img1 = img1.astype(np.float64)
    if img2.shape[-1]==3:
        img2_y = rgb2ycbcr(img2).astype(np.float64)
        bef = compute_bef(img2_y)
    else:
        img2 = img2.astype(np.float64)
        bef = compute_bef(img2)
    mse = np.mean((img1 - img2)**2)
    mse_b = mse + bef
    if mse_b == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse_b))

def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
