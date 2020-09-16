# FaceNetCustomized
Based on FacenetEvolve, add some customized modules to create/save face features from random images and reuse them on face recognition.

# File hierarchy
 1. evolveface     ***This contains backbone net, utils and picture alignment tools***
  - align     ***This contains picture alignment tools and OPR net (Modified & Customized applied)***
  - backbone      ***This contains backbone structures (Use IR50 in application)***
  - head
  - loss
  - util      ***This contains some utils for extracting features nad performance recoder (Modified & Customized applied)***
  
 2. instance     ***This is the instance I create for face detection & recognition***
  - Detection   ***This contains face detection via camera***
  - Recognition     ***This contains face recognintion via camera & video***
  - align     ***This contains face align tools for console to use***
  - featureExtract      ***This contains feature extraction tools for console to use***
  - calcDistance.py     ***Calculate the distance between two given pics***
  - load_utils.py     ***Provide methods of loading model and dataset***
  
 3. tools     *We don't use this*

# Baidu Drive
链接：https://pan.baidu.com/s/1K-hlMyl82-NjTgsOsWQayg 
提取码：c0yw
