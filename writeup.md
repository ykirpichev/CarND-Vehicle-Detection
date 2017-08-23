**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog_examples.png
[image3]: ./output_images/sliding_windows_1.png
[image4]: ./output_images/sliding_windows_2.png
[image5]: ./output_images/sliding_windows_3.png
[image6]: ./output_images/sliding_windows_4.png
[image7]: ./output_images/output_bboxes_1.png
[image8]: ./output_images/output_bboxes_2.png
[video1]: ./output_video/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of three of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=18`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. I tried orientation={8,9,11,12,16,17,18}, pixels_per_cell={(8,8), (16,16)} and cells_per_block={(1,1),(2,2)}.
And estimated which one is better based on test accuracy of SVC. It turned out that the following parameters:
```python
color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 18  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```
gives the best accuracy on the test set.
```
Test Accuracy of SVC =  0.993
```
I also tried to use `tree.DecisionTreeClassifier` and SVC with `rbf` kernel. I've got worse results for `tree.DecisionTreeClassifier` and better results for SVC with `rbf` kernel.
But SVC with rbf kernel worked very slow, thus I decided to use LinearSVC in the final pipeline.
####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using similar approach as described by lections:
```python
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
svc = LinearSVC()

# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
I decided to search the following window postions:
```python
params =[
         {'ystart': 400, 'ystop':530, 'scale': 1, 'cells_per_step': 2},
         {'ystart': 400, 'ystop':550, 'scale': 1.5, 'cells_per_step': 2}, #1 is good
         {'ystart': 450, 'ystop':650, 'scale': 2, 'cells_per_step': 2},
         {'ystart': 450, 'ystop':720, 'scale': 3, 'cells_per_step': 2},
        ]
```
I experimented with different parameters for `find_cars_bboxes` by using auxiliary function `find_cars_debug`
Here are results which I've got (in the same order as mentioned above):

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image7]
![alt text][image8]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video/project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. Then I stored running window of history with positive detections.  From the history of detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected using approach which was described by video materials.

Here is code snippet of final pipeline:
```python
class VehicleDetector():
    def __init__(self, limit):
        self.history = []
        self.history_limit = limit
        
        
    def add_sample(self, bboxes):
        self.history.append(bboxes)
        if len(self.history) > self.history_limit:
            self.history = self.history[len(self.history) - self.history_limit:]
            
    def process_frame(self, img):
        bboxes = []
        for param in params:
            ystart = param['ystart']
            ystop = param['ystop']
            scale = param['scale']
            cells_per_step = param['cells_per_step']
            bboxes += find_cars_bboxes(img,
                                       ystart,
                                       ystop,
                                       scale,
                                       svc,
                                       X_scaler,
                                       orient,
                                       pix_per_cell,
                                       cell_per_block,
                                       cells_per_step,
                                       spatial_size,
                                       hist_bins,
                                       color_space,
                                       spatial_feat,
                                       hist_feat,
                                       hog_feat)

        if len(bboxes) > 0:
            self.add_sample(bboxes)


        heatmap_img = np.zeros_like(img[:,:,0])
        for bboxes in self.history:
            heatmap_img = add_heat(heatmap_img, bboxes)
        heatmap_img = apply_threshold(heatmap_img, max(3, 1 + len(self.history)//2))

        labels = label(heatmap_img)
        draw_img, rect = draw_labeled_bboxes(np.copy(img), labels)
        return draw_img
```
Yes, it is very basic, but because of very high precision of classifier, this simple pipeline worked reasonably well on project video.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think that approach using HOG features is a little bit outdated and it is good to study the classical computer vision algorithms, but if we want to implement something really robust and fast it is better to choose deep learning approach and use one of the modern neural networks for object detection.

Apart from that, in order to improve my pipeline, more sophisticated false positive detector need to be implemented.
Also, better turned sliding windows parameters might give better results.
In order to reduce false positives or false negatives detections, more train data examples might be collected.
May be different machine learning algorithm might give better results, but perhaps it will be slower.



