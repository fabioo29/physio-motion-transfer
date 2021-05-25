import os
import tensorflow as tf
import numpy as np
from glob import glob
from PIL import Image

slim = tf.contrib.slim

N_CLASSES = 20
INPUT_RESIZE = 1080

label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]

N_CLASSES = 20
INPUT_RESIZE = 1024
DEFAULT_PADDING = 'SAME'

def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(ckpt_path, ckpt_name))
        print("Restored model parameters from {}".format(ckpt_name))
        return True
    else:
        return False

def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated
 
def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs

class Network(object):

    def __init__(self, inputs, trainable=True, is_training=False, n_classes=20, keep_prob=1):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.setup(is_training, n_classes, keep_prob)

    def setup(self, is_training, n_classes, keep_prob):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable)

    def make_w_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        stddev = 0.01
        return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev), trainable=self.trainable)

    def make_b_var(self, name, shape):
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0), trainable=self.trainable)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_w_var('weights', shape=[k_h, k_w, c_i // group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_b_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def atrous_conv(self,
                    input,
                    k_h,
                    k_w,
                    c_o,
                    dilation,
                    name,
                    relu=True,
                    padding=DEFAULT_PADDING,
                    group=1,
                    biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_w_var('weights', shape=[k_h, k_w, c_i // group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_b_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(values=inputs, axis=axis, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name)

    @layer
    def batch_normalization(self, input, name, is_training, activation_fn=None, scale=True):
        with tf.variable_scope(name) as scope:
            output = slim.batch_norm(
                input,
                activation_fn=activation_fn,
                is_training=is_training,
                updates_collections=None,
                scale=scale,
                scope=scope)
            return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)

    @layer
    def upsample(self, input, size_h, size_w, name):
        with tf.variable_scope(name) as scope:
            return tf.image.resize_images(input, size=[size_h, size_w])

    @layer
    def pyramid_pooling(self, input, o_c, pool_size, name):
        with tf.variable_scope(name) as scope:
            dims = tf.shape(input)
            out_height, out_width = dims[1], dims[2]
            pool_ly = tf.nn.avg_pool(input, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1],
                                     padding=DEFAULT_PADDING, name='pool_ly')
            weight = self.make_w_var('weights', shape=[3, 3, pool_ly.get_shape()[-1], o_c])
            biases = self.make_var('biases', o_c)
            conv_ly = tf.nn.conv2d(pool_ly, weight, strides=[1, 1, 1, 1], padding='SAME', name='conv_ly')
            conv_ly = tf.nn.bias_add(conv_ly, biases)
            conv_ly = tf.nn.relu(conv_ly, name='relu_ly')
            output = tf.image.resize_bilinear(conv_ly, [out_height, out_width])
            return output

class PGNModel(Network):
    def setup(self, is_training, n_classes, keep_prob):
        '''Network definition.

        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of
                       the-pretrained model frozen.
        '''
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1'))

        (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch2c'))

        (self.feed('bn2a_branch1',
                   'bn2a_branch2c')
             .add(name='res2a')
             .relu(name='res2a_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2b_branch2c'))

        (self.feed('res2a_relu',
                   'bn2b_branch2c')
             .add(name='res2b')
             .relu(name='res2b_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2c_branch2c'))

        (self.feed('res2b_relu',
                   'bn2c_branch2c')
             .add(name='res2c')
             .relu(name='res2c_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch1'))

        (self.feed('res2c_relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch2c'))

        (self.feed('bn3a_branch1',
                   'bn3a_branch2c')
             .add(name='res3a')
             .relu(name='res3a_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b1_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b1_branch2c'))

        (self.feed('res3a_relu',
                   'bn3b1_branch2c')
             .add(name='res3b1')
             .relu(name='res3b1_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b2_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b2_branch2c'))

        (self.feed('res3b1_relu',
                   'bn3b2_branch2c')
             .add(name='res3b2')
             .relu(name='res3b2_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b3_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b3_branch2c'))

        (self.feed('res3b2_relu',
                   'bn3b3_branch2c')
             .add(name='res3b3')
             .relu(name='res3b3_relu')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch1'))

        (self.feed('res3b3_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch2c'))

        (self.feed('bn4a_branch1',
                   'bn4a_branch2c')
             .add(name='res4a')
             .relu(name='res4a_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b1_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b1_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b1_branch2c'))

        (self.feed('res4a_relu',
                   'bn4b1_branch2c')
             .add(name='res4b1')
             .relu(name='res4b1_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b2_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b2_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b2_branch2c'))

        (self.feed('res4b1_relu',
                   'bn4b2_branch2c')
             .add(name='res4b2')
             .relu(name='res4b2_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b3_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b3_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b3_branch2c'))

        (self.feed('res4b2_relu',
                   'bn4b3_branch2c')
             .add(name='res4b3')
             .relu(name='res4b3_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b4_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b4_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b4_branch2c'))

        (self.feed('res4b3_relu',
                   'bn4b4_branch2c')
             .add(name='res4b4')
             .relu(name='res4b4_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b5_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b5_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b5_branch2c'))

        (self.feed('res4b4_relu',
                   'bn4b5_branch2c')
             .add(name='res4b5')
             .relu(name='res4b5_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b6_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b6_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b6_branch2c'))

        (self.feed('res4b5_relu',
                   'bn4b6_branch2c')
             .add(name='res4b6')
             .relu(name='res4b6_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b7_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b7_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b7_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b7_branch2c'))

        (self.feed('res4b6_relu',
                   'bn4b7_branch2c')
             .add(name='res4b7')
             .relu(name='res4b7_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b8_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b8_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b8_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b8_branch2c'))

        (self.feed('res4b7_relu',
                   'bn4b8_branch2c')
             .add(name='res4b8')
             .relu(name='res4b8_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b9_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b9_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b9_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b9_branch2c'))

        (self.feed('res4b8_relu',
                   'bn4b9_branch2c')
             .add(name='res4b9')
             .relu(name='res4b9_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b10_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b10_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b10_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b10_branch2c'))

        (self.feed('res4b9_relu',
                   'bn4b10_branch2c')
             .add(name='res4b10')
             .relu(name='res4b10_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b11_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b11_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b11_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b11_branch2c'))

        (self.feed('res4b10_relu',
                   'bn4b11_branch2c')
             .add(name='res4b11')
             .relu(name='res4b11_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b12_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b12_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b12_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b12_branch2c'))

        (self.feed('res4b11_relu',
                   'bn4b12_branch2c')
             .add(name='res4b12')
             .relu(name='res4b12_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b13_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b13_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b13_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b13_branch2c'))

        (self.feed('res4b12_relu',
                   'bn4b13_branch2c')
             .add(name='res4b13')
             .relu(name='res4b13_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b14_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b14_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b14_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b14_branch2c'))

        (self.feed('res4b13_relu',
                   'bn4b14_branch2c')
             .add(name='res4b14')
             .relu(name='res4b14_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b15_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b15_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b15_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b15_branch2c'))

        (self.feed('res4b14_relu',
                   'bn4b15_branch2c')
             .add(name='res4b15')
             .relu(name='res4b15_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b16_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b16_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b16_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b16_branch2c'))

        (self.feed('res4b15_relu',
                   'bn4b16_branch2c')
             .add(name='res4b16')
             .relu(name='res4b16_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b17_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b17_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b17_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b17_branch2c'))

        (self.feed('res4b16_relu',
                   'bn4b17_branch2c')
             .add(name='res4b17')
             .relu(name='res4b17_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b18_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b18_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b18_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b18_branch2c'))

        (self.feed('res4b17_relu',
                   'bn4b18_branch2c')
             .add(name='res4b18')
             .relu(name='res4b18_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b19_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b19_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b19_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b19_branch2c'))

        (self.feed('res4b18_relu',
                   'bn4b19_branch2c')
             .add(name='res4b19')
             .relu(name='res4b19_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b20_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b20_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b20_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b20_branch2c'))

        (self.feed('res4b19_relu',
                   'bn4b20_branch2c')
             .add(name='res4b20')
             .relu(name='res4b20_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b21_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b21_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b21_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b21_branch2c'))

        (self.feed('res4b20_relu',
                   'bn4b21_branch2c')
             .add(name='res4b21')
             .relu(name='res4b21_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b22_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b22_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b22_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b22_branch2c'))

        (self.feed('res4b21_relu',
                   'bn4b22_branch2c')
             .add(name='res4b22')
             .relu(name='res4b22_relu'))

        # Parsing networks

        (self.feed('res4b22_relu')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch1'))

        (self.feed('res4b22_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch2c'))

        (self.feed('bn5a_branch1',
                   'bn5a_branch2c')
             .add(name='res5a')
             .relu(name='res5a_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5b_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5b_branch2c'))

        (self.feed('res5a_relu',
                   'bn5b_branch2c')
             .add(name='res5b')
             .relu(name='res5b_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5c_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5c_branch2b')
             .batch_normalization(activation_fn=tf.nn.relu, name='bn5c_branch2b', is_training=is_training)
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5c_branch2c'))

        (self.feed('res5b_relu',
                   'bn5c_branch2c')
             .add(name='res5c')
             .relu(name='res5c_relu'))

        (self.feed('res3b3_relu')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='parsing_branch1'))
        (self.feed('res4b22_relu')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='parsing_branch2'))
        (self.feed('res5c_relu')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='parsing_branch3'))

        (self.feed('parsing_branch1', 'parsing_branch2', 'parsing_branch3')
             .concat(axis=3, name='parsing_branch_concat')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='parsing_branch'))

        (self.feed('parsing_branch')
             .pyramid_pooling(512, 60, name='parsing_pp1'))
        (self.feed('parsing_branch')
             .pyramid_pooling(512, 30, name='parsing_pp2'))
        (self.feed('parsing_branch')
             .pyramid_pooling(512, 20, name='parsing_pp3'))
        (self.feed('parsing_branch')
             .pyramid_pooling(512, 10, name='parsing_pp4'))

        (self.feed('parsing_branch', 'parsing_pp1', 'parsing_pp2', 'parsing_pp3', 'parsing_pp4')
             .concat(axis=3, name='parsing_pp_out')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='parsing_pp_conv')
             .dropout(keep_prob, name='parsing_pp_dropout')
             .conv(3, 3, n_classes, 1, 1, biased=True, relu=False, name='parsing_fc'))

        # Edge networks

        (self.feed('res3b3_relu')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='edge_branch1'))
        (self.feed('res4b22_relu')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='edge_branch2'))
        (self.feed('res5c_relu')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='edge_branch3'))

        (self.feed('edge_branch1', 'edge_branch2', 'edge_branch3')
             .concat(axis=3, name='edge_branch_concat')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='edge_branch'))

        (self.feed('edge_branch')
            .pyramid_pooling(512, 60, name='edge_pp1'))
        (self.feed('edge_branch')
            .pyramid_pooling(512, 30, name='edge_pp2'))
        (self.feed('edge_branch')
            .pyramid_pooling(512, 20, name='edge_pp3'))
        (self.feed('edge_branch')
            .pyramid_pooling(512, 10, name='edge_pp4'))

        (self.feed('edge_branch', 'edge_pp1', 'edge_pp2', 'edge_pp3', 'edge_pp4')
             .concat(axis=3, name='edge_pp_out')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='edge_pp_conv')
             .dropout(keep_prob, name='edge_pp_dropout')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='edge_fc'))

        # Intermediate supervision

        (self.feed('edge_branch3')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='fc1_edge_c0_res5'))
        (self.feed('edge_branch3')
             .atrous_conv(3, 3, 1, 2, padding='SAME', relu=False, name='fc1_edge_c1_res5'))
        (self.feed('edge_branch3')
             .atrous_conv(3, 3, 1, 4, padding='SAME', relu=False, name='fc1_edge_c2_res5'))
        (self.feed('edge_branch3')
             .atrous_conv(3, 3, 1, 8, padding='SAME', relu=False, name='fc1_edge_c3_res5'))
        (self.feed('edge_branch3')
             .atrous_conv(3, 3, 1, 16, padding='SAME', relu=False, name='fc1_edge_c4_res5'))
        (self.feed('fc1_edge_c0_res5', 'fc1_edge_c1_res5', 'fc1_edge_c2_res5', 'fc1_edge_c3_res5', 'fc1_edge_c4_res5')
             .add(name='fc1_edge_res5'))

        (self.feed('edge_branch2')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='fc1_edge_c0_res4'))
        (self.feed('edge_branch2')
             .atrous_conv(3, 3, 1, 2, padding='SAME', relu=False, name='fc1_edge_c1_res4'))
        (self.feed('edge_branch2')
             .atrous_conv(3, 3, 1, 4, padding='SAME', relu=False, name='fc1_edge_c2_res4'))
        (self.feed('edge_branch2')
             .atrous_conv(3, 3, 1, 8, padding='SAME', relu=False, name='fc1_edge_c3_res4'))
        (self.feed('edge_branch2')
             .atrous_conv(3, 3, 1, 16, padding='SAME', relu=False, name='fc1_edge_c4_res4'))
        (self.feed('fc1_edge_c0_res4', 'fc1_edge_c1_res4', 'fc1_edge_c2_res4', 'fc1_edge_c3_res4', 'fc1_edge_c4_res4')
             .add(name='fc1_edge_res4'))

        (self.feed('edge_branch1')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='fc1_edge_c0_res3'))
        (self.feed('edge_branch1')
             .atrous_conv(3, 3, 1, 2, padding='SAME', relu=False, name='fc1_edge_c1_res3'))
        (self.feed('edge_branch1')
             .atrous_conv(3, 3, 1, 4, padding='SAME', relu=False, name='fc1_edge_c2_res3'))
        (self.feed('edge_branch1')
             .atrous_conv(3, 3, 1, 8, padding='SAME', relu=False, name='fc1_edge_c3_res3'))
        (self.feed('edge_branch1')
             .atrous_conv(3, 3, 1, 16, padding='SAME', relu=False, name='fc1_edge_c4_res3'))
        (self.feed('fc1_edge_c0_res3', 'fc1_edge_c1_res3', 'fc1_edge_c2_res3', 'fc1_edge_c3_res3', 'fc1_edge_c4_res3')
             .add(name='fc1_edge_res3'))

        # Refine networks

        (self.feed('parsing_pp_conv')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='parsing_fea'))
        (self.feed('parsing_fc')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='parsing_remap'))
        (self.feed('edge_pp_conv')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='edge_fea'))
        (self.feed('edge_fc')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='edge_remap'))

        (self.feed('parsing_fea', 'parsing_remap', 'edge_fea', 'edge_remap')
             .concat(axis=3, name='parsing_rf_concat')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='parsing_rf'))

        (self.feed('parsing_rf')
             .pyramid_pooling(512, 60, name='parsing_rf_pp1'))
        (self.feed('parsing_rf')
             .pyramid_pooling(512, 30, name='parsing_rf_pp2'))
        (self.feed('parsing_rf')
             .pyramid_pooling(512, 20, name='parsing_rf_pp3'))
        (self.feed('parsing_rf')
             .pyramid_pooling(512, 10, name='parsing_rf_pp4'))

        (self.feed('parsing_rf', 'parsing_rf_pp1', 'parsing_rf_pp2', 'parsing_rf_pp3', 'parsing_rf_pp4')
             .concat(axis=3, name='parsing_rf_out')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='parsing_rf_conv')
             .dropout(keep_prob, name='parsing_rf_dropout')
             .conv(3, 3, n_classes, 1, 1, biased=True, relu=False, name='parsing_rf_fc'))

        (self.feed('edge_fea', 'edge_remap', 'parsing_fea', 'parsing_remap')
             .concat(axis=3, name='edge_rf_concat')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='edge_rf'))

        (self.feed('edge_rf')
            .pyramid_pooling(512, 60, name='edge_rf_pp1'))
        (self.feed('edge_rf')
            .pyramid_pooling(512, 30, name='edge_rf_pp2'))
        (self.feed('edge_rf')
            .pyramid_pooling(512, 20, name='edge_rf_pp3'))
        (self.feed('edge_rf')
            .pyramid_pooling(512, 10, name='edge_rf_pp4'))

        (self.feed('edge_rf', 'edge_rf_pp1', 'edge_rf_pp2', 'edge_rf_pp3', 'edge_rf_pp4')
             .concat(axis=3, name='edge_rf_out')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='edge_rf_conv')
             .dropout(keep_prob, name='edge_rf_dropout')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='edge_rf_fc'))

def main(input_dir, output_dir, checkpoint_dir):
    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load input
    input_files = sorted(glob(os.path.join(input_dir, '*')))
    input_queue = tf.train.slice_input_producer([tf.convert_to_tensor(input_files, dtype=tf.string)], shuffle=False)
    img_contents = tf.io.read_file(input_queue[0])
    img = tf.io.decode_jpeg(img_contents, channels=3)
    # Resize to prevent OOM
    img = tf.image.resize(img, [INPUT_RESIZE, INPUT_RESIZE], preserve_aspect_ratio=True)
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2)
    image = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # TODO: Subtract by mean (see image_reader)
    image_rev = tf.reverse(image, tf.stack([1]))

    image_batch = tf.stack([image, image_rev])

    # Create network
    with tf.variable_scope('', reuse=False):
        net = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)

    # parsing net
    parsing_out1 = net.layers['parsing_fc']
    parsing_out2 = net.layers['parsing_rf_fc']

    # edge net
    edge_out2 = net.layers['edge_rf_fc']

    # combine resize
    parsing_out1 = tf.image.resize_images(parsing_out1, tf.shape(image_batch)[1: 3, ])
    parsing_out2 = tf.image.resize_images(parsing_out2, tf.shape(image_batch)[1: 3, ])

    edge_out2 = tf.image.resize_images(edge_out2, tf.shape(image_batch)[1: 3, ])

    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))

    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    raw_output_all = tf.argmax(raw_output_all, axis=3)
    pred_all = tf.expand_dims(raw_output_all, dim=3)  # Create 4-d tensor.

    raw_edge = tf.reduce_mean(tf.stack([edge_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_edge, num=2, axis=0)
    tail_output_rev = tf.reverse(tail_output, tf.stack([1]))
    raw_edge_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_edge_all = tf.expand_dims(raw_edge_all, dim=0)

    # Which variables to load.
    restore_var = tf.global_variables()
    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)

    if not load(loader, sess, checkpoint_dir):
        raise IOError('Checkpoint loading failed')

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps and output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input_file in input_files:
        parsing_ = sess.run(pred_all)
        img_id = os.path.splitext(os.path.basename(input_file))[0]

        msk = decode_labels(parsing_, num_classes=N_CLASSES)
        parsing_im = Image.fromarray(msk[0])
        parsing_im.save('{}/{}.png'.format(output_dir, img_id))

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main('datasets/CIHP/images', 'output', 'checkpoint')
