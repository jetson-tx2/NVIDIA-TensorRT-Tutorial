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

The fourth model has the error: 

----------------------------------------------------------------
Input filename:   resnet50.onnx
ONNX IR version:  0.0.3
Opset version:    9
Producer name:    pytorch
Producer version: 0.4
Domain:           
Model version:    0
Doc string:       
----------------------------------------------------------------
Parsing model
[2019-04-17 12:04:25    INFO] 321:Conv -> (64, 128, 128)
[2019-04-17 12:04:25    INFO] 322:BatchNormalization -> (64, 128, 128)
[2019-04-17 12:04:25    INFO] 323:Relu -> (64, 128, 128)
[2019-04-17 12:04:25    INFO] 324:MaxPool -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 325:Conv -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 326:BatchNormalization -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 327:Relu -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 328:Conv -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 329:BatchNormalization -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 330:Relu -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 331:Conv -> (256, 64, 64)
[2019-04-17 12:04:25    INFO] 332:BatchNormalization -> (256, 64, 64)
[2019-04-17 12:04:25    INFO] 333:Conv -> (256, 64, 64)
[2019-04-17 12:04:25    INFO] 334:BatchNormalization -> (256, 64, 64)
[2019-04-17 12:04:25    INFO] 335:Add -> (256, 64, 64)
[2019-04-17 12:04:25    INFO] 336:Relu -> (256, 64, 64)
[2019-04-17 12:04:25    INFO] 337:Conv -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 338:BatchNormalization -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 339:Relu -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 340:Conv -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 341:BatchNormalization -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 342:Relu -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 343:Conv -> (256, 64, 64)
[2019-04-17 12:04:25    INFO] 344:BatchNormalization -> (256, 64, 64)
[2019-04-17 12:04:25    INFO] 345:Add -> (256, 64, 64)
[2019-04-17 12:04:25    INFO] 346:Relu -> (256, 64, 64)
[2019-04-17 12:04:25    INFO] 347:Conv -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 348:BatchNormalization -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 349:Relu -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 350:Conv -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 351:BatchNormalization -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 352:Relu -> (64, 64, 64)
[2019-04-17 12:04:25    INFO] 353:Conv -> (256, 64, 64)
[2019-04-17 12:04:25    INFO] 354:BatchNormalization -> (256, 64, 64)
[2019-04-17 12:04:25    INFO] 355:Add -> (256, 64, 64)
[2019-04-17 12:04:25    INFO] 356:Relu -> (256, 64, 64)
[2019-04-17 12:04:25    INFO] 357:Conv -> (128, 64, 64)
[2019-04-17 12:04:25    INFO] 358:BatchNormalization -> (128, 64, 64)
[2019-04-17 12:04:25    INFO] 359:Relu -> (128, 64, 64)
[2019-04-17 12:04:25    INFO] 360:Conv -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 361:BatchNormalization -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 362:Relu -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 363:Conv -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 364:BatchNormalization -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 365:Conv -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 366:BatchNormalization -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 367:Add -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 368:Relu -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 369:Conv -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 370:BatchNormalization -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 371:Relu -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 372:Conv -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 373:BatchNormalization -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 374:Relu -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 375:Conv -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 376:BatchNormalization -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 377:Add -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 378:Relu -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 379:Conv -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 380:BatchNormalization -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 381:Relu -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 382:Conv -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 383:BatchNormalization -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 384:Relu -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 385:Conv -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 386:BatchNormalization -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 387:Add -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 388:Relu -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 389:Conv -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 390:BatchNormalization -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 391:Relu -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 392:Conv -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 393:BatchNormalization -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 394:Relu -> (128, 32, 32)
[2019-04-17 12:04:25    INFO] 395:Conv -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 396:BatchNormalization -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 397:Add -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 398:Relu -> (512, 32, 32)
[2019-04-17 12:04:25    INFO] 399:Conv -> (256, 32, 32)
[2019-04-17 12:04:25    INFO] 400:BatchNormalization -> (256, 32, 32)
[2019-04-17 12:04:25    INFO] 401:Relu -> (256, 32, 32)
[2019-04-17 12:04:25    INFO] 402:Conv -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 403:BatchNormalization -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 404:Relu -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 405:Conv -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 406:BatchNormalization -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 407:Conv -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 408:BatchNormalization -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 409:Add -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 410:Relu -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 411:Conv -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 412:BatchNormalization -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 413:Relu -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 414:Conv -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 415:BatchNormalization -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 416:Relu -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 417:Conv -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 418:BatchNormalization -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 419:Add -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 420:Relu -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 421:Conv -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 422:BatchNormalization -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 423:Relu -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 424:Conv -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 425:BatchNormalization -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 426:Relu -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 427:Conv -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 428:BatchNormalization -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 429:Add -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 430:Relu -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 431:Conv -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 432:BatchNormalization -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 433:Relu -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 434:Conv -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 435:BatchNormalization -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 436:Relu -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 437:Conv -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 438:BatchNormalization -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 439:Add -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 440:Relu -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 441:Conv -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 442:BatchNormalization -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 443:Relu -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 444:Conv -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 445:BatchNormalization -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 446:Relu -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 447:Conv -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 448:BatchNormalization -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 449:Add -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 450:Relu -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 451:Conv -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 452:BatchNormalization -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 453:Relu -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 454:Conv -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 455:BatchNormalization -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 456:Relu -> (256, 16, 16)
[2019-04-17 12:04:25    INFO] 457:Conv -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 458:BatchNormalization -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 459:Add -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 460:Relu -> (1024, 16, 16)
[2019-04-17 12:04:25    INFO] 461:Conv -> (512, 16, 16)
[2019-04-17 12:04:25    INFO] 462:BatchNormalization -> (512, 16, 16)
[2019-04-17 12:04:25    INFO] 463:Relu -> (512, 16, 16)
[2019-04-17 12:04:25    INFO] 464:Conv -> (512, 8, 8)
[2019-04-17 12:04:25    INFO] 465:BatchNormalization -> (512, 8, 8)
[2019-04-17 12:04:25    INFO] 466:Relu -> (512, 8, 8)
[2019-04-17 12:04:25    INFO] 467:Conv -> (2048, 8, 8)
[2019-04-17 12:04:25    INFO] 468:BatchNormalization -> (2048, 8, 8)
[2019-04-17 12:04:25    INFO] 469:Conv -> (2048, 8, 8)
[2019-04-17 12:04:25    INFO] 470:BatchNormalization -> (2048, 8, 8)
[2019-04-17 12:04:25    INFO] 471:Add -> (2048, 8, 8)
[2019-04-17 12:04:25    INFO] 472:Relu -> (2048, 8, 8)
[2019-04-17 12:04:25    INFO] 473:Conv -> (512, 8, 8)
[2019-04-17 12:04:25    INFO] 474:BatchNormalization -> (512, 8, 8)
[2019-04-17 12:04:25    INFO] 475:Relu -> (512, 8, 8)
[2019-04-17 12:04:25    INFO] 476:Conv -> (512, 8, 8)
[2019-04-17 12:04:25    INFO] 477:BatchNormalization -> (512, 8, 8)
[2019-04-17 12:04:25    INFO] 478:Relu -> (512, 8, 8)
[2019-04-17 12:04:25    INFO] 479:Conv -> (2048, 8, 8)
[2019-04-17 12:04:25    INFO] 480:BatchNormalization -> (2048, 8, 8)
[2019-04-17 12:04:25    INFO] 481:Add -> (2048, 8, 8)
[2019-04-17 12:04:25    INFO] 482:Relu -> (2048, 8, 8)
[2019-04-17 12:04:25    INFO] 483:Conv -> (512, 8, 8)
[2019-04-17 12:04:25    INFO] 484:BatchNormalization -> (512, 8, 8)
[2019-04-17 12:04:25    INFO] 485:Relu -> (512, 8, 8)
[2019-04-17 12:04:25    INFO] 486:Conv -> (512, 8, 8)
[2019-04-17 12:04:25    INFO] 487:BatchNormalization -> (512, 8, 8)
[2019-04-17 12:04:25    INFO] 488:Relu -> (512, 8, 8)
[2019-04-17 12:04:25    INFO] 489:Conv -> (2048, 8, 8)
[2019-04-17 12:04:25    INFO] 490:BatchNormalization -> (2048, 8, 8)
[2019-04-17 12:04:25    INFO] 491:Add -> (2048, 8, 8)
[2019-04-17 12:04:25    INFO] 492:Relu -> (2048, 8, 8)
[2019-04-17 12:04:25    INFO] 493:GlobalAveragePool -> (2048, 1, 1)
[2019-04-17 12:04:25    INFO] 494:Constant -> (1)
[2019-04-17 12:04:25    INFO] 495:Shape -> (4)
Unsupported ONNX data type: INT64 (7)
While parsing node number 175 [Gather -> "496"]:
--- Begin node ---
input: "495"
input: "494"
output: "496"
op_type: "Gather"
attribute {
  name: "axis"
  i: 0
  type: INT
}
doc_string: "/data/atoaca/anaconda3/lib/python3.7/site-packages/torchvision/models/resnet.py(161): forward\n/data/atoaca/anaconda3/lib/python3.7/site-packages/torch/nn/modules/module.py(477): _slow_forward\n/data/atoaca/anaconda3/lib/python3.7/site-packages/torch/nn/modules/module.py(487): __call__\n/data/atoaca/anaconda3/lib/python3.7/site-packages/torch/jit/__init__.py(252): forward\n/data/atoaca/anaconda3/lib/python3.7/site-packages/torch/nn/modules/module.py(489): __call__\n/data/atoaca/anaconda3/lib/python3.7/site-packages/torch/jit/__init__.py(197): get_trace_graph\n/data/atoaca/anaconda3/lib/python3.7/site-packages/torch/onnx/utils.py(192): _trace_and_get_graph_from_model\n/data/atoaca/anaconda3/lib/python3.7/site-packages/torch/onnx/utils.py(224): _model_to_graph\n/data/atoaca/anaconda3/lib/python3.7/site-packages/torch/onnx/utils.py(281): _export\n/data/atoaca/anaconda3/lib/python3.7/site-packages/torch/onnx/utils.py(104): export\n/data/atoaca/anaconda3/lib/python3.7/site-packages/torch/onnx/__init__.py(27): export\ndensenet.py(6): <module>\n"

--- End node ---
ERROR: /data/atoaca/code/torch2tensorrt/onnx-tensorrt/onnx2trt_utils.hpp:301 In function convert_axis:
[8] Assertion failed: axis >= 0 && axis < nbDims

