from utils import preprocess, write_filename, split_train_val
import os
import cv2


dir_path = r"D:\tablebank\ICDAR2019_cTDaR\ICDAR2019_cTDaR\training\TRACKB1\ground_truth"
save_path = r"D:\tablebank\ICDAR2019_cTDaR\ICDAR2019_cTDaR\training\TRACKB1\preprocess"
files =  os.listdir(dir_path)

image_name = []
for img_file in files:
    name = os.path.splitext(img_file)
    print("current file %s" %(name[0]))
    
    if name[1] == '.jpg' or name[1] == '.png':
        img_path = os.path.join(dir_path, img_file)
        xml_path = os.path.join(dir_path, name[0]+'.xml')
        img_table, fm_table = preprocess(img_path, xml_path)
        assert len(img_table)==len(fm_table), "image and feature map not match"
        
        for idx in range(len(img_table)):
            image_name.append(name[0]+ '_'+ str(idx))
            cv2.imwrite(os.path.join(save_path, name[0]+ '_'+ str(idx)+ '.jpg'), img_table[idx])
            cv2.imwrite(os.path.join(save_path, name[0]+ '_'+ str(idx)+ '_maskh.png'), fm_table[idx][0])
            cv2.imwrite(os.path.join(save_path, name[0]+ '_'+ str(idx)+ '_maskv.png'), fm_table[idx][1])
    if name[1] == '.tiff':
        img_path = os.path.join(dir_path, img_file)
        tif = TIFF.open(img_path, mode='r')
        for img in list(tif.iter_images()):
            tiff = img
        cv2.imwrite(os.path.join(dir_path, name[0]+'.jpg'))
        
        img_path = os.path.join(dir_path, name[0]+'.jpg')
        xml_path = os.path.join(dir_path, name[0]+'.xml')
        img_table, fm_table = preprocess(img_path, xml_path)
        assert len(img_table)==len(fm_table), "image and feature map not match"
        
        for idx in range(len(img_table)):
            image_name.append(name[0]+ '_'+ str(idx))
            cv2.imwrite(os.path.join(save_path, name[0]+ '_'+ str(idx)+ '.jpg'), img_table[idx])
            cv2.imwrite(os.path.join(save_path, name[0]+ '_'+ str(idx)+ '_maskh.png'), fm_table[idx][0])
            cv2.imwrite(os.path.join(save_path, name[0]+ '_'+ str(idx)+ '_maskv.png'), fm_table[idx][1])
            
dir_path = r'D:\a上财学习材料\20毕业论文\实验\data'
filename_path = 'cTDaR19_imagename.txt'
write_filename(os.path.join(dir_path, filename_path), image_name)

# split train and test
split_train_val(dir_path, filename_path)