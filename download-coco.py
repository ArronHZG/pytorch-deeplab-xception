import os
import re
import urllib.request
import progressbar

# Global variables
pbar = None
# downloaded = 0


class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()



base_dir = "E:/data/coco"
download_url = ["http://images.cocodataset.org/zips/train2017.zip",
                "http://images.cocodataset.org/zips/val2017.zip",
                "http://images.cocodataset.org/zips/test2017.zip",
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
                "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
                "http://images.cocodataset.org/annotations/image_info_test2017.zip"]
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
for url in download_url:
    file_name = url.split("/")[-1]
    print(file_name)
    if not os.path.exists(os.path.join(base_dir, file_name)):
        print("download: {}".format(file_name))
        urllib.request.urlretrieve(url, os.path.join(base_dir, file_name), MyProgressBar())
