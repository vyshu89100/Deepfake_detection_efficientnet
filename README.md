# Deepfake_detection_efficientnet
# Deepfake Detection with EfficientNet-V2

This project aims to detect fake images and videos that have been generated using deep learning techniques, such as DeepFakes and Face2Face. These methods can manipulate or replace facial information from the original content, creating realistic and misleading synthetic media. The goal of this project is to identify such manipulations and distinguish them from real images.

## Dataset

We use the Deepfake Detection - Faces - Part 0_0 dataset, This dataset includes all detectable faces of the corresponding part of the full dataset. Kaggle and the host expected and encouraged us to train our models outside of Kaggle’s notebooks environment; however, for someone who prefers to stick to Kaggle's kernels, this dataset would help a lot.

## Model

We use the EfficientNet-V2 network, which is a state-of-the-art convolutional neural network for image classification. EfficientNet-V2 is based on the compound scaling principle, which balances the network depth, width, and resolution. EfficientNet-V2 also introduces a new family of convolutional layers, called MBConv, which use inverted residuals and linear bottlenecks to improve efficiency and performance. We use the EfficientNet-V2-S variant, which has 21.5M parameters and achieves 83.9% top-1 accuracy on ImageNet.

## Training

We use the PyTorch framework to implement and train our model. We use the cross-entropy loss function and the Adam optimizer with a learning rate of 0.001. We also use data augmentation techniques, such as random cropping, flipping, and rotation, to increase the diversity and robustness of the training data. We train our model for 60 epochs on a single GPU, with a batch size of 35.

## Evaluation

We evaluate our model on the test set of the Deepfake Detection - Faces - Part 0_0 dataset, using the following metrics: accuracy, precision, recall, and F1-score. We also plot the confusion matrix and the receiver operating characteristic (ROC) curve to visualize the performance of our model. The results are shown below:

## Conclusion

We have successfully developed a deepfake detection system based on the EfficientNet-V2 network and the FaceForensics++ dataset. Our system can achieve high accuracy and robustness in distinguishing real and fake images and videos. We hope that our system can help combat the spread of misinformation and abusive content caused by deepfake technology.
