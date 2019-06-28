import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    with tf.gfile.GFile('./artifacts/model3.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

        input_node    = sess.graph.get_tensor_by_name('tower_0/inference_input:0')
        output_node   = sess.graph.get_tensor_by_name('tower_0/inference_output:0')

        warmup_sample = np.expand_dims(np.uint8(np.random.randint(low=0, high=255, size=(256, 256, 3))), axis=0)

        warmup_logits = sess.run(output_node, feed_dict={input_node: warmup_sample})
        print(warmup_logits.shape)
