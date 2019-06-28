import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    with tf.gfile.GFile('./artifacts/model2.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def)

        input_node    = sess.graph.get_tensor_by_name('import/input_1:0')
        output_node   = sess.graph.get_tensor_by_name('import/dense_2/Sigmoid:0')
        ph_1          = sess.graph.get_tensor_by_name("import/bn_conv1/keras_learning_phase:0")

        warmup_sample = np.expand_dims(np.uint8(np.random.randint(low=0, high=255, size=(256, 256, 3))), axis=0)

        warmup_logits = sess.run(output_node, feed_dict={ph_1: False, input_node: warmup_sample})
        print(warmup_logits.shape)
