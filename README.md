# Detecting Volleyball in Images



The application of object detection algorithms are only limited to data. Once you have enough data to train on, you can create an object detection algorithm to identify anything. In this project, I have implemented SSD with mobilenet trained on COCO dataset to identify volleyball in images. The dataset contains around 1300 images of a volleyball match and a csv file containing the coordinates of the volleyball. 

## Running the test_model.py

1. The test_model.py file takes in your test images directory path as input argument.
2. It saves the object coordinates in the image in a csv file and saves the output images too.
3. It searches for all the images ending with a .png extension in your test_images folders. 

```python 
python test_model.py --im_dir=/path/to/test_images/ --outname=save_file.csv
```


## Results

### During Training. .

### Step 0:

![alt text](https://raw.githubusercontent.com/shreyas0906/Detecting-Volleyball/master/SSD_initial.png)

### Step 61:

![alt text](https://raw.githubusercontent.com/shreyas0906/Detecting-Volleyball/master/SSD-61.png)

### Step 257

![alt text](https://raw.githubusercontent.com/shreyas0906/Detecting-Volleyball/master/SSD-257.png)

