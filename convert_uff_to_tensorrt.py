# convert_uff_to_tensorrt.py
import os
import tensorrt as trt; print('TensorRT Version: {}'.format(trt.__version__))


# Set location of model
base_dir = os.path.join('/', 'workspace', 'optimization')
artifacts_dir = os.path.join(base_dir, 'artifacts')
model_file_name = 'model1.pb'
model_file_path = os.path.join(artifacts_dir, model_file_name)

# Set network settings
input_names = ['tower_0/inference_input:0']
output_names = ['tower_0/inference_output:0']
n_channel, n_height, n_width = 3, 128, 128
dimensions = [n_channel, n_height, n_width]
batch_size = 1
precision = 'fp32'  # options are 'fp16' (default), 'int8', and 'fp32'
architecture = 'v100'  # options are 't4' (default), 'v100', and 'xavier'

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

# Create builder, network, and parser
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()
parser = trt.UffParser()

# Configure the builder here.
builder.max_workspace_size = 2**30

# Parse the Uff Network
print('Registering Input:', parser.register_input(input_names[0], dimensions))
print('Registering Output:', parser.register_output(output_names[0]))
print('Parsing Model:', parser.parse(model_file_path, network))

# Set precision
if precision == 'fp16':
    builder.fp16_mode = True
elif precision == 'int8':
    builder.int8_mode = True

# Set batch size
#builder.max_batch_size = batch_size

# Build the engine
engine = builder.build_cuda_engine(network)
print(engine)

# Create engine file name
engine_file_name = model_file_name.replace('.pb', '') + '_{}_b{}_{}.engine'
engine_file_name = engine_file_name.format(architecture, batch_size, precision)
engine_file_path = os.path.join(artifacts_dir, engine_file_name)

# Save engine file
with open(engine_file_path, 'wb') as file:
    print('Saving engine file to:', engine_file_path)
    file.write(engine.serialize())
