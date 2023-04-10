cd data
#wget -c http://data.csail.mit.edu/places/iros18/data/floating_kinect1/floating_kinect1_object_train.tar.gz
#wget -c http://data.csail.mit.edu/places/iros18/data/floating_kinect1/floating_kinect1_object_val.tar.gz
#wget -c http://data.csail.mit.edu/places/iros18/data/floating_kinect1/floating_kinect1_mask_train.tar.gz
#wget -c http://data.csail.mit.edu/places/iros18/data/floating_kinect1/floating_kinect1_mask_val.tar.gz
wget -c https://www.dropbox.com/s/iczyzdu4izl5v34/floating_kinect1_object_train.tar.gz
wget -c https://www.dropbox.com/s/0on8hl22qsr1j2n/floating_kinect1_object_val.tar.gz
wget -c https://www.dropbox.com/s/rj9igkpx74mviqb/floating_kinect1_mask_train.tar.gz
wget -c https://www.dropbox.com/s/5mfxemp1712akje/floating_kinect1_mask_val.tar.gz
tar -xf floating_kinect1_object_train.tar.gz
tar -xf floating_kinect1_object_val.tar.gz
tar -xf floating_kinect1_mask_train.tar.gz
tar -xf floating_kinect1_mask_val.tar.gz
