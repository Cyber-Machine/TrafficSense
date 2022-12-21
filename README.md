# TrafficSense

This repository contains a computer vision model for identifying police hand signals using TensorFlow & Tensorflow-lite. The model is trained on a custom dataset of images demonstrating various hand signals and achieves an `accuracy of ~86% on the test set`.

Various techniques have been used to improve upon the model's accuracy such as Data Augmentation, Dropout, validation sets, etc.

Pose
 Detection was identified by `movenet-thunder` model which is lighter and achieves a realtime detection.
Classification of poses was done on a custom layered Neural Network.

A TensorFlow-lite model is also created using quantization and pruning, achieving similar accuracy with just a fraction of the original model size (26KB). This can be used in IoT devices like Raspberry Pi / Arduino for detection.

It can classify between four poses :
- `front` - To stop vehicles coming from front
- `behind` - To stop vehicles coming from behind
- `frandbk` -  To stop vehicles simultaneously from front and behind
- `close` - Warning  signal closing all vehicles.

### Libraries Used
- TensorFlow - Used for model training and inference
- Numpy - Used for array manipulation
- OpenCV - Used for image preprocessing and display
- Tensorflow-lite - Used for model deployment on edge devices
- Docker - Used for model deployment on DockerHub

### Run Locally

#### Run on system
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

#### Run via Docker
In order to run this model through docker allow X server connection to access display.

On Terminal run

```bash
# Allow X server connection
xhost +local:*
```

And now run the app on docker

```bash
 docker run --rm -it --device /dev/video0 -e "DISPLAY=$DISPLAY" -v /tmp/.X11-unix/:/tmp/.X11-unix/ cybermachine/trafficsense:latest
```

Press `ESC` to close the screen.

Also revoke access to X server connection after use.

```bash
# Disallow X server connection
xhost -local:*
```

### Run Remotely

In order to run this model on remotely, upload `TrafficSense_Colab.ipynb` to [Google Colab](https://colab.research.google.com/) and 
run all the cells.

Output is generated as `output.mp4` inside TrafficSense folder in colab.