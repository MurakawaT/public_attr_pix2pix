from ap2p_model import Ap2p
import os
from PIL import Image
import numpy as np
import csv

ap2p = Ap2p()
nowdir = os.path.dirname(os.path.abspath(__file__))

#重み読み込み
ap2p.load_weights(nowdir + "/../attr_pix2pix/weight_gender/" + "004")

#データパス
data_name = "00001"
#data_name = "0000012"
dataset_path = nowdir + "/created_img/" + data_name + ".png"
img = Image.open(dataset_path)
img = img.resize((128,128), Image.BICUBIC)
imgs = [img]

#予測
img_attr, img_non_attr = ap2p.pred(imgs,5)

#データ成型
img_attr = img_attr * 127.5 + 127.5
img_non_attr = img_non_attr * 127.5 + 127.5
img_attr = img_attr.reshape(128,128,3)
img_non_attr = img_non_attr.reshape(128,128,3)
img_1 = Image.fromarray(img_attr.astype(np.uint8))
img_2 = Image.fromarray(img_non_attr.astype(np.uint8))

#保存
img = img.resize((16,16), Image.BICUBIC)
img = img.resize((128,128), Image.BICUBIC)
img.save(nowdir + "/created_img/" + data_name +"_low_128.png")
img_1.save(nowdir + "/created_img/" + data_name +"_SR_male.png")
img_2.save(nowdir + "/created_img/" + data_name +"_SR_noMale.png")