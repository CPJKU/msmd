# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 18:02:31 2016

@author: matthias
"""

import os
import glob
import cv2
import numpy as np

# first call
# convert -density 150 Mozart_Piano_Sonata_Facile.pdf -quality 90 page.png

img_dir = "/home/matthias/cp/data/sheet_localization/real_music/Mozart_Sonata_k309_3rd_movement/sheet/"

target_width = 835

file_paths = glob.glob(img_dir  + "*.*")
file_paths = np.sort(file_paths)
for i, img_path in enumerate(file_paths):
    img = cv2.imread(img_path, -1)
    
    # compute resize stats
    ratio = float(target_width) / img.shape[1]
    target_height = img.shape[0] * ratio
    target_width = int(target_width)
    target_height = int(target_height)
    
    img_rsz = cv2.resize(img, (target_width, target_height))
    
    out_path = img_dir + "%02d.png" % (i + 1)
    cv2.imwrite(out_path, img_rsz)

