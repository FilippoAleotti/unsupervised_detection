import cv2
import gflags
import numpy as np
import os
import sys
import scipy.io as sio
import tensorflow as tf
from keras.utils.generic_utils import Progbar
from models.adversarial_learner import AdversarialLearner
from models.utils.general_utils import postprocess_mask, postprocess_image, compute_boundary_score

from common_flags import FLAGS

des_width = 1280
des_height = 384

def _test_masks():
    assert os.path.exists(FLAGS.filename)
    assert os.path.exists(FLAGS.flow_dir)
    assert os.path.exists(FLAGS.root_dir)
    
    learner = AdversarialLearner()
    learner.setup_inference(FLAGS, aug_test=False)
    saver = tf.train.Saver([var for var in tf.trainable_variables()])


    # manages multi-threading
    sv = tf.train.Supervisor(logdir=FLAGS.test_save_dir,
                             save_summaries_secs=0,
                             saver=None)
    
    with open(FLAGS.filename,'r') as f:
        samples = f.readlines()

    with sv.managed_session() as sess:
        checkpoint = FLAGS.ckpt_file
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("Resume model from checkpoint {}".format(checkpoint))
        else:
            raise IOError("Checkpoint file not found")

        sess.run(learner.test_iterator.initializer)

        n_steps = int(np.ceil(learner.test_samples / float(FLAGS.batch_size)))

        progbar = Progbar(target=n_steps)

        i = 0

        for step in range(n_steps):
            if sv.should_stop():
                break
            try:
                inference = learner.inference(sess)
            except tf.errors.OutOfRangeError:
                  print("End of testing dataset")  # ==> "End of dataset"
                  break
            # Now write images in the test folder
            

            # select mask
            generated_mask = inference['mask']
            
            # Verbose image generation
            save_dir = os.path.join(FLAGS.test_save_dir)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            print(samples[step])
            filename = os.path.join(save_dir, "frame_{:08d}.png".format(step))
            cv2.imwrite(filename, generated_mask.squeeze()*255.)
            i+=1
            return
            progbar.update(step)

def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    _test_masks()

if __name__ == "__main__":
    main(sys.argv)
