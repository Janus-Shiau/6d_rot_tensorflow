# 6d_rot_tensorflow
6D rotation representation (["On the Continuity of Rotation Representations in Neural Networks"](https://arxiv.org/abs/1812.07035)) for tensorflow.

### Environment 
This code is implemmented and tested with [tensorflow](https://www.tensorflow.org/) 1.11.0.

### Usage
Just add the tf_rotation6d_to_matrix after your output, whose last dimension of tensor should be 6.
```
"""
Any model output whose last dimension is 6.
e.g. output = tf.layers.dense(hidden, 6)
"""
rot = tf_rotation6d_to_matrix(output)
```

### Details
TBA.

### Contact & Copy Right
Code work by Jia-Yau Shiau <jiayau.shiau@gmail.com>.

