## FCOS object detector

This project implements object detection algorithm described in https://arxiv.org/pdf/1904.01355.pdf using PyTorch. Source paper added to docs folder.


### Install
`make install`


### Train
`make train`
![Alt text](attachments/training_1.png?raw=true "Tensorboard - loss")
![Alt text](attachments/training_2.png?raw=true "Tensorboard - metrics")
![Alt text](attachments/training_3.png?raw=true "Tensorboard - images 1/2")
![Alt text](attachments/training_4.png?raw=true "Tensorboard - images 2/2")


### Evaluate
`make run`
![Alt text](attachments/inference_1.png?raw=true "Inference - 1/4")
![Alt text](attachments/inference_2.png?raw=true "Inference - 2/4")
![Alt text](attachments/inference_3.png?raw=true "Inference - 3/4")
![Alt text](attachments/inference_4.png?raw=true "Inference - 4/4")


### Test
`make tests`
