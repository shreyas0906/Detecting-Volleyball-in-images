# Detecting Volleyball in Images



The application of object detection algorithms are only limited to data. Once you have enough data to train on, you can create an object detection algorithm to identify anything. In this project, I have implemented SSD with mobilenet trained on COCO dataset to identify volleyball in images. The dataset contains around 1300 images of a volleyball match and a csv file containing the coordinates of the volleyball. 

To understand the tensorflow object detection API and how to use it to train the on your data, please refer [here](https://github.com/tensorflow/models/tree/master/research/object_detection)

After you run and evaluate your model, you will be wondering how to extract the detected object coordinates, scores and labels for all the detections as the eval.py from tensorflow does not show all this information directly. The test_model.py file saves the detected object coordinates in a csv file. I've modified the script as per my needs. I am detecting only one object in each image. However, you can modify my script to save the detections for the classes you have trained the model. 



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

