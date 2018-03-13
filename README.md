# Detecting Volleyball in Images



The application of object detection algorithms are only limited to data. Once you have enough data to train on, you can create an object detection algorithm to identify anything. In this project, I have implemented SSD with mobilenet trained on COCO dataset to identify volleyball in images. The dataset contains around 1300 images of a volleyball match and a csv file containing the coordinates of the volleyball. 

# Steps to train the model. 
## Preparing data.

I assume that you have tensorflow installed on your system. I am using Ubuntu 14.04. 

1. Split the data into training and testing folder. (With seperate csv file for training and testing.)
2. [Convert the data into tfRecords] (https://github.com/datitran/raccoon_dataset) 
3. Datitran provides a simple implementation of converting your images and label data into tfRecords. 
4. If you dont understand why we are converting the data into tfRecords, [please have a look here](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/).

## To train the model. 

1. Head over to the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and download the model of your choice. 

2. Also make sure to download the configuration file from [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs). NOTE: Download the configuration file for the model you have chosen.

3. Create a label file as follows:
- Open a notepad and enter the following information
'''
item {
  id: 1
  name: 'ball'
}
'''
- Save the file with a .pbtxt extension. 

## Running test_model.py

1. The test_model.py file takes in your test images directory path as input argument.
2. It searches for all the images ending with a .png extension in your test_images folders.
3. It saves the object coordinates in the image in a csv file and saves the output images. 

```python 
python test_model.py --im_dir=/path/to/test_images/ --outname=save_file_name.csv
```


## Results

### During Training. .

### Step 0:

![alt text](https://raw.githubusercontent.com/shreyas0906/Detecting-Volleyball/master/SSD_initial.png)

### Step 61:

![alt text](https://raw.githubusercontent.com/shreyas0906/Detecting-Volleyball/master/SSD-61.png)

### Step 257

![alt text](https://raw.githubusercontent.com/shreyas0906/Detecting-Volleyball/master/SSD-257.png)

