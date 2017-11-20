import cv2
from evaluation_utils import generate_depth_map, get_focal_length_baseline, read_text_lines, read_file_data
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import pdb
import os

episode = 'disp_supl1+sm1_1l_upsamp_8b_upproj_60e'
# episode = 'model_kitti+cs_resnet_eigen'
# episode = 'full_supl1+sm1_1l_upsamp_8b_upproj_50e'
# episode = 'disp_supl1+sm1_1l_upsamp_8b_upproj_60e'
# episode = 'model_kitti_eigen'
is_depth = False
is_pp = False
base_path = '/scratch/gengshay/output'

if is_pp:
    pred_disp_path = '/scratch/gengshay/tmp/%s/disparities_pp.npy' % episode
    episode = episode + "_pp"
else:
    pred_disp_path = '/scratch/gengshay/tmp/%s/disparities.npy' % episode
    

test_files = read_text_lines( "%s/%s" % ('filenames','eigen_test_files.txt') )
gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, '/data/gengshay/KITTI/')

min_depth = 1e-3
max_depth = 80

num_samples = len(im_files)
pred_disparities = np.load(pred_disp_path)
gt_depths = []

directory = '%s/images-%s' % (base_path, episode)
if not os.path.exists(directory):
    os.makedirs(directory)

f1 = open('%s/mixed-%s.html' % (base_path,episode),'w')
txt = '<!DOCTYPE html>\n<html>\n<body>\n<div class="img-c">\n<ul>\n'
f1.write(txt)

for t_id in range(num_samples):
    camera_id = cams[t_id]  # 2 is left, 3 is right
    depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
    gt_depths.append(depth.astype(np.float32))
    
    disp_pred = cv2.resize(pred_disparities[t_id], (im_sizes[t_id][1], im_sizes[t_id][0]), interpolation=cv2.INTER_LINEAR)
    if is_depth:
        depth_pred = disp_pred * 1000.
    else:
        disp_pred = disp_pred * disp_pred.shape[1]
        # need to convert from disparity to depth
        focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], camera_id)
        depth_pred = (baseline * focal_length) / disp_pred
    depth_pred = 400. / depth_pred  # inverse depth
    depth_pred[np.isinf(depth_pred)] = 0
    depth_pred[depth_pred < min_depth] = min_depth
    depth_pred[depth_pred > max_depth] = max_depth

    im = plt.imread(im_files[t_id])
    mixed = cv2.addWeighted(im[:,:,0].astype(np.float32),0.05,depth_pred,1,0)

    #cv2.imwrite('%s/%04d-mix.jpg' % (directory,t_id), mixed )
    #cv2.imwrite('%s/%04d-im.jpg' % (directory,t_id), im )
    #cv2.imwrite('%s/%04d-pred.jpg' % (directory,t_id), depth_pred )
    plt.imsave('%s/%04d-mix.jpg' % (directory,t_id), mixed, cmap='plasma')
    plt.imsave('%s/%04d-im.jpg' % (directory,t_id), im, cmap='plasma')
    plt.imsave('%s/%04d-pred.jpg' % (directory,t_id), depth_pred, cmap='plasma')
    txt = '<li>%d.<img src="images-%s/%04d-mix.jpg" onclick="changeImage(this)"></li>\n' % (t_id,episode,t_id)
    f1.write(txt)

txt = '</ul>\n</div>\n</body>\n</html>\n'
txt += '<script language="javascript">\n\
        function changeImage(img) {\n\
        if (img.src.includes("-pred"))\n\
        {\n\
            img.src = img.src.replace("-pred","-im");\n\
        }\n\
        else if (img.src.includes("-mix"))\n\
        {\n\
            img.src = img.src.replace("-mix","-pred");\n\
        }\n\
        else\n\
        {\n\
            img.src = img.src.replace("-im","-mix");\n\
        }\n\
        }\n\
</script>\n'
f1.write(txt)

f1.close()


with open('%s/index.html' % base_path,'r') as f:
    lines = f.readlines()
    lines.insert(-3, '<a href="./mixed-%s.html">mixed-%s<p></a>\n' % (episode,episode))

    
with open('%s/index.html' % base_path,'w') as f:
    f.write(''.join(lines))
