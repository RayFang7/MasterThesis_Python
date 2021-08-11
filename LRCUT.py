from genericpath import exists
import os
from PIL import Image
import PIL
size = (224,224)
dataset_path = "/home/ray/TrainingData/321/5f/"
img_width = 512
img_height = 512

def fmkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

fmkdir(os.path.join(dataset_path,'plane'))

for plane in range(3):
    fmkdir(os.path.join(dataset_path,'plane',str(plane)))
    for fold in range(1,11):
        fmkdir(os.path.join(dataset_path,'plane',str(plane),str(fold)))
        for c in ["0_MI","1_H"]:
            fmkdir(os.path.join(dataset_path,'plane',str(plane),str(fold),c))

for fold in range(1,11):
    for c in ["0_MI","1_H"]:
        p = os.path.join(dataset_path,str(fold),c)
        file_list = os.listdir(p)
        for index,name in enumerate(file_list):
            img = Image.open(p+'/'+name)
            for plane in range(3):
                temp = img.crop((img_width / 2 * plane, 0, img_width /2 * (plane + 1), img_height)) if not plane == 2 else img
                temp = temp.resize(size,PIL.Image.BILINEAR)
                temp.save(os.path.join(dataset_path,'plane',str(plane),str(fold),c,name))
            if index%100 == 0:
                print("{}/{} {} {}/{}".format(index,len(file_list),c,fold,10))

