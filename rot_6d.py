'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
The implementation of 6D rotatiton representation,
based on 

    https://arxiv.org/abs/1812.07035

"On the continuity of rotation representations in neural networks"
Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, Hao Li.
Conference on Neural Information Processing Systems (NeurIPS) 2019.
'''
import tensorflow as tf

def tf_rotation6d_to_matrix(r6d):
    """ Compute rotation matrix from 6D rotation representation.
        Implementation base on 
            https://arxiv.org/abs/1812.07035
        [Inputs]
            6D rotation representation (last dimension is 6)
        [Returns]
            flattened rotation matrix (last dimension is 9)
    """
    tensor_shape = r6d.get_shape().as_list()

    with tf.variable_scope('rot6d_to_matrix'):
        r6d   = tf.reshape(r6d, [-1,6])
        x_raw = r6d[:,0:3]
        y_raw = r6d[:,3:6]
    
        x = tf.nn.l2_normalize(x_raw, axis=-1)
        z = tf.cross(x, y_raw)
        z = tf.nn.l2_normalize(z, axis=-1)
        y = tf.cross(z, x)

        x = tf.reshape(x, [-1,3,1])
        y = tf.reshape(y, [-1,3,1])
        z = tf.reshape(z, [-1,3,1])
        matrix = tf.concat([x,y,z], axis=-1)

        if len(tensor_shape) == 1:
            matrix = tf.reshape(matrix, [9])
        else:
            output_shape = tensor_shape[:-1] + [9]
            matrix = tf.reshape(matrix, output_shape)

    return matrix