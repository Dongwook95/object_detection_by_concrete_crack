# object_detection_by_concrete_crack (based on Transfer Learning)
# This research topic and related codes were produced and proceeded to create a model capable of simple crack detection using a small amount of concrete crack image data using transfer learning.
#
# The related code operation method is as follows:
# 
# A. Few-shot learning by Bounding Box making
# 1. Cloning model with tensorflow model (importing os & pathlib)
# 2. Installing Object Detection API
# 3. Importing object_detection.utils tools (Caution! it is not possible to import in many cases.)
# 4. Upload 5 sample concrete crack images for Few-shot Learning.
# 5. Labeling bounding boxes on each uploaded images.
# 6. Prepping image datas & Create label criteria for Few-shot Learning.
# 7. Calculate fine-tuned model on each batches.
# 
# 1. Main Object Detection (Concrete Crack Detecting Preparation)
# Reference code made so that labeling can be done with any photo using a model that is already capable of object detection, such as transfer learning.
# 
# 2. Make Own Image .tgz Dataset
# Reference code used to create a dataset containing private concrete crack images and crack labeling used in research activities.
#
# As additional research activities are ongoing, we will upload the relevant code!
