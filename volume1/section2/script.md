# Feature extraction and Data Preparation

Machine Learning, in a computer vision context, is almost exclusively used for classification purposes.
The first task we need to complete is the collection of training data.
Sometimes you can use the entire image data, but more often than not this will be both impractical
and inefficient.
Instead we should try to extract the most meaningful pieces of information from an image,
then use that for our machine learning purposes.

## The xfeatures2d module

In opencv, the xfeatures2d module contains most of the feature extraction classes.
In most cases you will have built opencv from sources so make sure that you have built opencv
with the contrib modules flag otherwise the module won't be present.
To verify its presence, simply type the following in a terminal window:

$ python
>>> import cv2

>>> help(cv2)

You should see xfeatures2d along with other extra modules, like face.

The xfeatures2d module contains a lot of utility classes for feature extraction, each suited for 
different purposes, or performing better at certain tasks than others.

Let's take a look at them:
```
    BriefDescriptorExtractor_create(...)
        BriefDescriptorExtractor_create([, bytes[, use_orientation]]) -> retval
    
    DAISY_create(...)
        DAISY_create([, radius[, q_radius[, q_theta[, q_hist[, norm[, H[, interpolation[, use_orientation]]]]]]]]) -> retval
    
    FREAK_create(...)
        FREAK_create([, orientationNormalized[, scaleNormalized[, patternScale[, nOctaves[, selectedPairs]]]]]) -> retval
    
    LATCH_create(...)
        LATCH_create([, bytes[, rotationInvariance[, half_ssd_size]]]) -> retval
    
    LUCID_create(...)
        LUCID_create(lucid_kernel, blur_kernel) -> retval
    
    SIFT_create(...)
        SIFT_create([, nfeatures[, nOctaveLayers[, contrastThreshold[, edgeThreshold[, sigma]]]]]) -> retval
    
    SURF_create(...)
        SURF_create([, hessianThreshold[, nOctaves[, nOctaveLayers[, extended[, upright]]]]]) -> retval
    
    StarDetector_create(...)
        StarDetector_create([, maxSize[, responseThreshold[, lineThresholdProjected[, lineThresholdBinarized[, suppressNonmaxSize]]]]]) -> retval
```

As mentioned before, features extractors are not the only way of extracting features. You can build
your own if you have a powerful algorithm under your sleeve, maybe even extending the Algorithm
class of opencv.
You can use the entire image data, or its color histogram, or whatever else you can think of that may
be pertinent to your use case.
For example, if your dataset only consists of pictures of marine landscapes, and your purpose is 
to classify daytime pictures and sunset pictures then colour is really the only information you 
would be looking for.
Let's go ahead and look at an example that utilizes the SIFT algorithm, which is one of the most popular.


