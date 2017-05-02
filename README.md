[//]: # (Image References)

[image1]: ./images/visualization.png "Visualization"
[image2]: ./images/color_distribution.png "Color Distribution"
[image3]: ./images/grayscale.png "Grayscale"
[image4]: ./images/histogram_equalization.png "Histogram equalization"
[image5]: ./images/input_images.png "Sample images"
[image6]: ./images/input_processed.png "Sample images pre-processed"

[image7]: ./images/top_5_softmax_1.png "Top 5 softmax probabilities"
[image8]: ./images/top_5_softmax_2.png "Top 5 softmax probabilities"
[image9]: ./images/top_5_softmax_3.png "Top 5 softmax probabilities"
[image10]: ./images/top_5_softmax_4.png "Top 5 softmax probabilities"
[image11]: ./images/top_5_softmax_5.png "Top 5 softmax probabilities"

[image12]: ./images/min_max.png "Min Max Scaling"

#### You're reading it! :rocket: Great! here is a link to my [project code.](https://github.com/d3r0n/sdcen-road-signs/blob/master/Road-Sign-Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Some numbers:
* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 pixels
* Images are in RGB
* The number of unique classes/labels in the data set is ?

> Please see paragraph called "Some numbers" how I have used pandas and numpy for exploration.

#### 2. Exploratory visualization of the dataset
Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among different traffic signs classes.

As you can see the data is not uniformly distributed some classes are underrepresented (< 250 samples) and other are overrepresented (~2000 samples). This will affect how the network learns.

>Code for visualization below is in section called "Exploratory visualization of the dataset"

![alt text][image1]

#### 3. Next lets look at color distribution in our data set.

We clearly see that the pictures tend to have low contract. We will address that.

>Code for visualization below is in cell called "Color distribution"

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Pre-process data

First we need to change our images to grayscale. There is no proof that including color improve performance of the network recognition (?link). But reducing redundant data by factor of 3 will definitely improve the training time.

Sample image with its distribution after changing to grayscale.

Property| Value
--------|------------------------
ClassId |                                            10
SignName|  No passing for vehicles over 3.5 metric tons

>Code for visualization below is in section "Grayscale"

![alt text][image3]

Next we address the problem of the contrast. More uniform histogram will help network distinguish features form the background and to faster descend to optimum.

Same sign image as above but after histogram equalization.

>Code for visualization below is in section "Histogram equalization"

![alt text][image4]

OK, but the network will work much better on values between -1 and 1 than 0 and 255. So here I do simple Min Max Scaling. Check out how distribution on dataset will look after this last pre-processing step.

>Code for visualization below is in section "Min Max Scaling"

![alt text][image12]

#### 2. Training, Test, Validation

All, training, test and validation data sets are loaded in first cell of the notebook.
Each consist of following number of samples.
* Training: 34799
* Test: 12630
* Validation:

#### 3. Model architecture
The model of choice is CNN which implementation is based on the LeNet-5.

>Code for model definition below is in section "Model definition"

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU | |
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Dropout | 10 % |
| Convolution 5x5	    | 1x1 stride, same padding, outputs 16x16x64 |
| RELU | |
| Max pooling | 2x2 stride, outputs 8x8x64     	|
| Dropout | 20 % |
| Convolution 3x3 | 1x1 stride, valid padding, outputs 6x6x128 |
| RELU | |
| Dropout | 30% |
| Convolution 3x3 | 1x1 stride, valid padding, outputs 4x4x256 |
| RELU | |
| Max pooling | 2x2 stride, outputs 2x2x256 |
| Dropout | 40% |
| Convolution 2x2 | 1x1 stride, valid padding, outputs 1x1x512 |
| RELU | |
| Dropout | 50% |
| Fully connected		| 512x1024 |
| RELU | |
|	Dropout | 50% |
| Fully connected | 1024x43 |

#### 4. Model training
After many tests. Best results gave reducing learning rate to 0.0001 and reducing size of the batch to 64.

Before each epoch training data ware shuffled and split into batches which ware then fed into model.

Either it was 1000 or 300 epochs accuracy stayed similar ~99%. Interesting is fact that the I did not see overfitting where with rising train accuracy the validation accuracy will sink. I suspect this is thanks to high dropout rates I have introduced. This simple action made model very resilient.

For gradient descend I have used AdamOptimizer which optimized over cross entropy with logits.

>Code for model training can be found in section "Train the Model"

#### 5. Solution

My final model results were:
* training set accuracy of - 100%
* validation set accuracy of - 99.4%
* test set accuracy of - 97%

I decided on iterative process of finding a solution. So here is how I thought:

First I started with same model as LeNet-5. Next I have worked on pre processing the data. Then, to improve I had to change the model. So I have removed fully connected layer. No significant difference. Added new convolutional layer. Bingo!, better. Added dropout. More stable!. Feed outputs of 3 convolutional layers to one first fully connected. Nope. Reduced batch size. Yes, good. Reduced leering rate. Nice. What if I learn whole night? So I set epochs to 1000. What? No difference? I is not even worse? hmm.

>Code for model training can be found in section "Test the Model"

### Test a Model on New Images

#### 1. Some five German traffic signs.

Here are five German traffic signs that I found on the web. They are exactly the size which my network accepts.
![alt text][image5]

I thought that easiest will be children and hardest to classify will be double curve.
But the model performed so well that even the latter was correct.

This how those signs looked after pre-processing (but before Min-Max Scaling!).
![alt text][image6]

Final performance on new samples was: 100% WOW!

#### 2. Analyse performance.

Lets have a look how well model is sure of its predictions. Lets look at top 5 softmax probabilities for each given sample.

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

>Code for model analysis is in section called "Top 5 Softmax Probabilities"

First 4 images are quite boring because model is 100% sure about them, and it is right about that. But for the last one, the "Double crossing" is not so sure. Interestingly model thinks that this might be in 20 % "Children crossing". But in 80% it is sure that it is double curve, and again that is correct!
