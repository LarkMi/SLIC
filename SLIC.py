import skimage
from skimage.segmentation import slic,mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np
import cv2
#
# np.set_printoptions(threshold=np.inf)
path = 'C:\\Users\\Administrator\\Desktop\\SLIC\\'
img_name = 'test.png'
img = io.imread(path + img_name,as_gray=True) #as_gray是灰度读取，得到的是归一化值
segments = slic(img, n_segments=10, compactness=0.2,start_label = 1)#进行SLIC分割
out=mark_boundaries(img,segments)
out = out*255 #io的灰度读取是归一化值，若读取彩色图片去掉该行
img3 = Image.fromarray(np.uint8(out))
img3.show()
seg_img_name = 'seg.png'
img3.save(path +'\\' +seg_img_name)#显示并保存加上分割线后的图片

maxn = max(segments.reshape(int(segments.shape[0]*segments.shape[1]),))
for i in range(1,maxn+1):
    a = np.array(segments == i)
    b = img * a
    w,h = [],[]
    for x in range(b.shape[0]):
        for y in range(b.shape[1]):
            if b[x][y] != 0:
                w.append(x)
                h.append(y)
    
    c = b[min(w):max(w),min(h):max(h)]
    c = c*255
    d = c.reshape(c.shape[0],c.shape[1],1)
    e = np.concatenate((d,d),axis=2)
    e = np.concatenate((e,d),axis=2)
    img2 = Image.fromarray(np.uint8(e))
    img2.save(path +'\\'+str(i)+'.png')
    print('已保存第' + str(i) + '张图片')

wid,hig = [],[]
img = io.imread(path+'\\'+seg_img_name)

for i in range(1,maxn+1):
    w,h = [],[]
    for x in range(segments.shape[0]):
        for y in range(segments.shape[1]):
            if segments[x][y] == i:
                w.append(x)
                h.append(y)
                

    font=cv2.FONT_HERSHEY_SIMPLEX#使用默认字体
    #print((min(w),min(h)))
    img=cv2.putText(img,str(i),(h[int(len(h)/(2))],w[int(len(w)/2)]),font,1,(255,255,255),2)#添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
img = Image.fromarray(np.uint8(img))
img.show()
img.save(path +'\\'+seg_img_name+'_label.png')

