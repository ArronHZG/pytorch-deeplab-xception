class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/arron//文档/notebook/侯正罡/zh.d2l.ai/data/VOCdevkit/VOC2012'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/home/arron/Documents/arron/dataSet/coco'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
