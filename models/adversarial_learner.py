import os
import sys
import time
from itertools import count
import math
import numpy as np
import tensorflow as tf
from keras.utils.generic_utils import Progbar
from .nets import recover_net, generator_net
from .utils.loss_utils import charbonnier_loss, train_op
from .utils.flow_utils import flow_to_image_tf, preprocess_flow_batch
from .utils.general_utils import compute_all_IoU, disambiguate_forw_back
from .PWCNet.model_pwcnet import ModelPWCNet
from data.davis2016_data_utils import Davis2016Reader
from data.fbms_data_utils import FBMS59Reader
from data.segtrackv2_data_utils import SegTrackV2Reader
from data.kitti_data_utils import KittiReader

class AdversarialLearner(object):
    def __init__(self):
        pass

    def build_test_graph(self):
        """This graph will be used for testing. In particular, it will
           compute the loss on a testing set, or some other utilities.
        """
        with tf.name_scope("data_loading"):

            if self.config.dataset == 'KITTI':
                reader = KittiReader(kitti_data_path=self.config.root_dir, flow_datapath=self.config.flow_dir, 
                 filename_file=self.config.filename, height=self.config.img_height, width=self.config.img_width)
                test_batch, test_iter = reader.test_inputs()
            else:
                raise IOError("Dataset should be KITTI")

            image_batch, images_2_batch, flow_batch, fname_batch = test_batch[0], \
                                            test_batch[1], test_batch[2], test_batch[3]
        # Flow computed on original image size
        #flow_network = ModelPWCNet()
        #flow_batch = flow_network.predict_from_img_pairs(image_batch, images_2_batch)

        # Normalize flow
        flow_batch = flow_batch / tf.constant(self.config.flow_normalizer)

        with tf.name_scope("MaskNet") as scope:
            generated_masks = generator_net(images=image_batch,
                                       flows=preprocess_flow_batch(flow_batch),
                                       training=False,
                                       scope=scope,
                                       reuse=False)

        flow_masked = flow_batch * (1.0 - generated_masks)

        self.input_image = image_batch
        self.generated_masks = generated_masks
        self.test_samples = reader.number_samples
        self.test_iterator = test_iter

    def setup_inference(self, config, aug_test=False):
        """Sets up the inference graph.
        Args:
            config: config dictionary.
        """
        self.config = config
        self.aug_test = aug_test
        self.build_test_graph()

    def inference(self, sess):
        """Outputs a dictionary with the results of the required operations.
        Args:
            sess: current session
        Returns:
            results: dictionary with output of testing operations.
        """
        fetches = {'mask': self.generated_masks}
        results = sess.run(fetches)

        return results
