# TrafficSense

This repository contains a computer vision model for identifying police hand signals using TensorFlow & Tensorflow-lite. The model is trained on a custom dataset of images demonstrating various hand signals and achieves an accuracy of 80% on the test set.

Various techniques have been used to improve upon the model's accuracy such as Data Augmentation, Dropout, validation sets, etc.

Pose Detection was identified by `movenet-thunder` model which is lighter and achieves a realtime detection.
Classification of poses was done on a custom Neural Network.

A TensorFlow-lite model is also created using quantization and pruning, achieving similar accuracy with just a fraction of the original model size (26KB). This can be used in IoT devices like Raspberry Pi / Arduino for detection.

### Libraries Used
- TensorFlow - Used for model training and inference
- Numpy - Used for array manipulation
- OpenCV - Used for image preprocessing and display


### Run Locally

Clone the project

```bash
  git clone https://github.com/Cyber-Machine/TrafficSense
```

Go to the project directory

```bash
  cd TrafficSense
```

Install libraries

```python
  pip install -r requirements.txt
```

Run python file

```bash
  python detect.py
```