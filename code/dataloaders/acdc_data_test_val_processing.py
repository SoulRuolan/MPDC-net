import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk

import ToolDIY_fzyy

# test and val processing
slice_num = 0
mask_path = sorted(glob.glob("C:/Post/Dataset/ACDC/database/test/image/*.nii.gz))
for case in mask_path:
    img_itk = sitk.ReadImage(case)
    origin = img_itk.GetOrigin()
    spacing = img_itk.GetSpacing()
    direction = img_itk.GetDirection()
    image = sitk.GetArrayFromImage(img_itk)
    print("case: ", case)
    msk_path = case.replace("image", "label").replace(".nii.gz", "_gt.nii.gz")
    print("msk_path: ", msk_path)
    if os.path.exists(msk_path):
        msk_itk = sitk.ReadImage(msk_path)
        mask = sitk.GetArrayFromImage(msk_itk)
        image = (image - image.min()) / (image.max() - image.min())
        image = image.astype(np.float32)
        item = case.split("/")[-1].split(".")[0].split("\\")[-1]
        if image.shape != mask.shape:
            print("Error")
        print("item: ", item)
        f = h5py.File(
            root_path + '/h5py/test/{}.h5'.format(item), 'w')
        f.create_dataset(
            'image', data=image, compression="gzip")
        f.create_dataset('label', data=mask, compression="gzip")
        f.close()
        slice_num += 1
print("Converted all ACDC volumes to 2D slices")
print("Total {} slices".format(slice_num))
