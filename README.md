<img src="NV_TensorRT_Visual_2C_RGB-625x625-1.png" alt="pipeline" height="300px"/> 


For conversion to RT we have the following models:
- model1 = old school tensorflow convolutional network with no concat and no batch-norm
- model2 = pre-trained resnet50 keras model with tensorflow backend and added shortcuts
- model3 = modified resnet50 implemented in tensorflow and trained from scratch
- model4 = pre-trained resnet50 in pytorch

I have added for each a minimalist script which loads the graphs and inferences a random image. One should be able to deduce the name of input/output nodes and related sizes from the scripts.

For the first three scripts, our ML engineers tell me that the errors relate to the incompatibility between RT and the following blocks: 
- Cast
- Concat
- Batch_norm
