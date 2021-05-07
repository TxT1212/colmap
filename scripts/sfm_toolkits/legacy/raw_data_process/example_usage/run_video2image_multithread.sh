

python3 export_image_from_video.py \
		--dataset_path $DATAPATH  \
        --interval 30  \
		--multithread	\
        --clip -1 \
       	--resize  \
     	--width 1280   \
     	--height 720

#  	DATAPATH
#   - date_device1
#       -videos
#       -images_inv10_分辨率_cutsize
#   - date_device2
#       -videos
#       -images_inv10_分辨率_cutsize

#	interval 拆帧间隔
#	multithread 是否使用多线程