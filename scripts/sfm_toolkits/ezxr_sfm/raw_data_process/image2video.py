import cv2

img_root = '/media/netease/Dataset/LargeScene/Scene/Aomen/AomenAirGroundTest/preview'#这里写你的文件夹路径，比如：/home/youname/data/img/,注意最后一个文件夹要有斜杠
fps = 25    #保存视频的FPS，可以适当调整
size=(800,450)
#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
videoWriter = cv2.VideoWriter(img_root + '_ds.mp4',fourcc,fps,size)#最后一个是保存图片的尺寸

#for(i=1;i<471;++i)
for i in range(0,1099):
    print(i)
    frame = cv2.imread(img_root+"/frame%06d" % i+'.png')
    frame = cv2.resize(frame, (800,450))
    videoWriter.write(frame)
videoWriter.release()