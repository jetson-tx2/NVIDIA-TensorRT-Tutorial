from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Conv1D, Conv2D, Activation, Input, Flatten, Dense, Reshape, \
    Concatenate, Lambda, Layer, BatchNormalization

import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib

import tensorrt as trt
from tensorrt.parsers import uffparser
import uff

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train, y_test = to_categorical(y_train), to_categorical(y_test)


### Build Network ###

# Basic LeNet-esque architecture w/ BatchNorm
def build_nn(use_BN=True):
    # This network will only work with BatchNorm turned on, because the data is not standard scaled. Try it out!
    inputs = Input(shape=(32, 32, 3))

    arch = inputs  # Makes it easier to comment things in and out
    if use_BN:
        arch = BatchNormalization()(arch)

    arch = Conv2D(32, 3, padding='same', activation='relu')(arch)
    if use_BN:
        arch = BatchNormalization()(arch)

    arch = Conv2D(32, 3, padding='same', activation='relu')(arch)
    if use_BN:
        arch = BatchNormalization()(arch)

    arch = Conv2D(32, 3, padding='same', activation='relu')(arch)
    if use_BN:
        arch = BatchNormalization()(arch)

    arch = Flatten()(arch)
    arch = Dense(10)(arch)
    arch = Activation('softmax')(arch)

    nn = Model(inputs=inputs, outputs=arch)
    nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return nn


# We gotta build two copies of the graph, 1 in train, 1 in inference

use_BN = True

# Train:
nn = build_nn(use_BN)
nn.summary()
nn.fit(x_train, y_train, batch_size=256, epochs=1)

# Compare the copied network against the gold standard accuracy generated here:
results = nn.evaluate(x_test, y_test)

# Inference:
# The network will be created in inference mode as long as the learning phase is set to 0
# Seems changing this on the fly does not work - layers must be initialized in the mode you want them to operate in
K.set_learning_phase(0)
nn_test = build_nn(use_BN)

# Copy the train network into the inference network
nn_test.set_weights(nn.get_weights())

# Make sure this matches the number above so we know everything transferred to the inference network
results_test = nn_test.evaluate(x_test, y_test)
print(results, results_test)

### Save Network ###

snapshot_dir = 'snapshot/'
sess = K.get_session()
saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
checkpoint_path = saver.save(sess, snapshot_dir, global_step=0, latest_filename='checkpoint_state')
inference_graph = sess.graph

graph_def_file = 'graphdef.pb'
graph_io.write_graph(inference_graph, '.', graph_def_file)

in_names = nn_test.inputs[0].op.name
out_names = nn_test.outputs[0].op.name

print('Input name:', in_names, 'Output name:', out_names)

frozen_model_file = 'frozen.pb'
freeze_graph.freeze_graph(graph_def_file, "", False, checkpoint_path, out_names, "save/restore_all", "save/Const:0",
                          frozen_model_file, False, "")

### Parse Network via UFF ###

uff_model = uff.from_tensorflow_frozen_model('frozen.pb', [out_names])

parser = uffparser.create_uff_parser()
parser.register_input(in_names, (3, 32, 32), 0)
parser.register_output(out_names)

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, 256, 1 << 20, trt.infer.DataType.HALF)
