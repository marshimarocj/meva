# meva

## models
* model/birnn.py: sequence model without attention (p3 in slide)
* model/conv_transformer.py: convolutional self-attention (p4 in slide)
* model/multi_conv_transformer.py: deep convolutional self-attention (p5 in slide)

## experiment procedure
1. run expr/prepare_cfg.py to prepare the configuration files (one model configuration and one path configuration will be generated)
2. run driver/birnn.py, driver/conv_transformer.py, driver/multi_conv_transformer.py to train the model. Sample scripts are provided
3. run expr/predict.py to predict and report performance
