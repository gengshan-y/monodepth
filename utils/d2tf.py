import argparse
from evaluation_utils import read_text_lines, read_file_data, generate_depth_map, get_focal_length_baseline
import cv2
# from matplotlib import pyplot as plt
import numpy as np
import os, os.path
import pdb



parser = argparse.ArgumentParser(description='lidar label pre-processing.')
parser.add_argument('--name', type=str,   help='name of the image folder', default='disp_10_1')
parser.add_argument('--is_train',         help='if set, generate train lidar depth map', action='store_true')
parser.add_argument('--is_left',          help='if set, generate left lidar depth map', action='store_true')
args = parser.parse_args()


# Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if os.path.isdir(path):
            pass
        else: raise

if args.is_train:
    test_files = read_text_lines( "%s/%s" % ('filenames','eigen_train_files.txt'))
else:
    test_files = read_text_lines( "%s/%s" % ('filenames','eigen_val_files.txt'))
gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, '/ssd0/KITTI/' , is_left = args.is_left)


# discretize
# bins = np.linspace(0.,5.,num=50)  # log bins
# bins = np.linspace(1e-3,80.,num=100)  # log bins
nbins=100
max_depth = 80
min_depth = 1e-3
bins = np.logspace(np.log(min_depth),np.log(max_depth),num=nbins,base=np.e)

for t_id in range(len(gt_files)):
    gt_depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], cams[t_id], False, True)
    mask = np.logical_and(gt_depth < 80, gt_depth > 1e-3)
    gt_height, gt_width = gt_depth.shape
    crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,
                                     0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
    # mask = np.logical_and(mask, crop_mask)
    gt_depth[~mask] = -1

    disp = -1 * np.ones(gt_depth.shape)
    focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], cams[t_id])
    disp[mask] = (baseline * focal_length) / gt_depth[mask]
    # disp /= gt_width

    sep1 = gt_files[t_id].find('sync') + 4
    sep2 = im_files[t_id].find('.jpg') -11

    str1 = gt_files[t_id][:sep1]
    dir1 = '%s/%s' % (str1, args.name)
    mkdir_p(dir1)
    imname = im_files[t_id][sep2:]
    impath = '%s/%s' % (dir1,imname)
    if os.path.exists(impath):
        os.remove(impath)
    impath = impath[:-4] + '.png'


    # disp[mask] = np.log(disp[mask])  # log bins
    # disp = np.digitize(disp,bins)
    # cv2.imwrite(impath, disp)

    gt_depth = np.digitize(gt_depth,bins)
    cv2.imwrite(impath, gt_depth)
    # gt_depth = gt_depth[crop[0]:crop[1],crop[2]:crop[3]]
    # cv2.imwrite(impath, gt_depth)
    #im = exposure.rescale_intensity(disp, out_range='float')
    #im = img_as_uint(im)
    #io.imsave(impath, im)
    
    #pdb.set_trace()
    #t = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
    #print np.mean(abs(t-gt_depth))

    #imwrite_tf(disp, impath)
    #t = imread_tf_run(impath)

    if t_id % 1000 == 0:
      print '%d images left' % (len(gt_files) - t_id)
