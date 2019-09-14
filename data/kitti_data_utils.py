import numpy as np
import os
import tensorflow as tf
from data.aug_flips import random_flip_images

class DirectoryIterator(object):
    """
    Class for managing data loading.of images and labels
    We assume that the folder structure is:

    """
    def __init__(self, directory, part='train'):
        self.directory = directory

        name_division ={'train': 'ImageSets/480p/train.txt',
                        'val' : 'ImageSets/480p/val.txt',
                        'trainval': 'ImageSets/480p/trainval.txt'}
        part_file = os.path.join(directory, name_division.get(part))
        if not os.path.isfile(part_file):
            raise IOError("Partition file not found")

class KittiReader(object):
    def __init__(self, root_dir, max_temporal_len=1,
                 min_temporal_len=1, num_threads=6, kitti_data_path='/media/filippo/Filippo/ComputerVision/Dataset/Godard_KITTI',
                 flow_datapath='/media/filippo/Filippo/ComputerVision/Dataset/SelFlowProxy_Eigen',
                 filename_file='./filenames/E_L.txt',
                 height=384,
                 width=1280):
        self.root_dir = root_dir
        self.max_temporal_len = max_temporal_len
        self.min_temporal_len = min_temporal_len
        assert min_temporal_len < max_temporal_len, "Temporal lenghts are not consistenst"
        assert min_temporal_len > 0, "Min temporal len should be positive"
        self.num_threads = num_threads
        self.filename_file = filename_file
        self.number_of_paths_in_line = self.get_number_of_paths(self.filename_file)
        self.kitti_data_path = kitti_data_path
        self.flow_datapath = flow_datapath
        self.number_samples = self.get_num_samples(filename_file)
        self.height = height
        self.width  = width
    
    def get_num_samples(self, filename_file):
        return sum(1 for line in open(filename_file))

    def preprocess_image(self, img):
        orig_width = self.width
        orig_height = self.height
        img = ( tf.cast(img, tf.float32) / tf.constant(255.0) ) - 0.5
        img = tf.image.resize_images(img, [orig_height, orig_width])
        return img

    def test_inputs(self, batch_size=32, partition='val', t_len=2, with_fname=False,
                    test_crop=1.0):
        """Reads batches at test time.
        Args:
          batch_size: Number of examples per returned batch.
          partition: partition used for testing (should be 'val')
          t_len: temporal difference used at test time. This is defined according to:
                    time(image2) - time(img1) = t_len
          test_crop: amount of (central) cropping of the image at test time.
          with_fname: whether to return the file name or not (for logging)
        Returns:
          A tuple ((img_1, img_2, gt_seg, [fname]), iterator) where:
          * img_1, img_2 are float tensors with shape [batch_size, H, W, 3]
          * gt_seg is the ground-truth segmentation of img_1
          * iterator is the dataset iterator.
          * fname are the file names
        """
        self.test_crop = test_crop

        # Form training batches
        dataset = tf.data.TextLineDataset(self.filename_file)
        dataset = dataset.map(self.test_dataset_map,
                              num_parallel_calls=1)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=3*batch_size)
        iterator = dataset.make_initializable_iterator()
        img1s, img2s, flow, fnames = iterator.get_next()
        if with_fname:
            return (img1s, img2s, flow, fnames), iterator
        return (img1s, img2s, flow), iterator
    
    def get_number_of_paths(self, filename_file):
        with open(filename_file, 'r') as f:
            line = f.readline()
            number_of_paths = len(line.split(' '))
        return number_of_paths

    def get_image_paths_from_line(self, line, number_of_paths):
        ''' Read relative path from line in filenames '''
        paths = []
        split_line = tf.string_split([line]).values
        for index in range(number_of_paths):
            paths.append(split_line[index])
        return paths
    
    def test_dataset_map(self, input_queue):
        '''
        Reads two images
        '''
        # Reading
        self.image_paths = self.get_image_paths_from_line(
            line, self.number_of_paths_in_line)

        fname_1 = tf.string_join([self.kitti_data_path, self.image_paths[0]])
        fname_2 = tf.string_join([self.kitti_data_path, self.image_paths[1]])
        flow_fname = tf.string_join([self.flow_datapath, self.image_paths[2]])

        file_content = tf.read_file(fname_1)
        image_1 = tf.image.decode_jpeg(file_content, channels=3)
        image_1 = self.preprocess_image(image_1)
        file_content = tf.read_file(fname_2)
        image_2 = tf.image.decode_jpeg(file_content, channels=3)
        image_2 = self.preprocess_image(image_2)
        flow = self.tf.load_flo(flow_fname)

        return image_1, image_2, flow, fname_1
    
    def load_flo(self, filename):
            '''
                Load a .flo file
                More info about flo format here: http://vision.middlebury.edu/flow/code/flow-code/README.txt
            '''
            with open(filename,'rb') as f:
                sanity_check = np.fromstring(f.read(4), dtype='<f4')
                assert sanity_check == 202021.25

                width = np.asscalar(np.fromstring(f.read(4), dtype=np.int32))
                height = np.asscalar(np.fromstring(f.read(4), dtype=np.int32))

                number_of_elements = width*height*2
                flows = np.fromstring(f.read(number_of_elements*4), dtype='<f4')
                flows = np.reshape(flows, (height, width,2))
                return flows

    def tf_load_flo(self, filename):
        ''' Read a flo flow '''
        file_flo = tf.py_func( self.load_flo, [filename], [tf.float32])
        flow = file_flo[0]
        flow.set_shape([self.params.height, self.params.width, 2])
        return 

