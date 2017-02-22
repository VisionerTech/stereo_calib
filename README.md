
## stereo calibration project

this is the source code stereo calibration project for VisionerTech VMG-PROV 01. use this to get camera intrinsic and rectify map for see-through, marker-based AR and SLAM-based AR.

## Requirement:

1.  recommended specs: Intel Core i5-4460/8G RAM/GTX 660/at least two USB3.0/
2.  windows x64 version.(tested on win7/win10)

## Installation

1.  download and install the  Visual Studio 2012, the download address is here: https://www.microsoft.com/zh-cn/download/details.aspx?id=30682

2.  download and install OpenCV(version 2.4.X) as:
http://docs.opencv.org/2.4.11/doc/tutorials/introduction/windows_install/windows_install.html

3.  open "\stereo_calib\stereo_calib.sln", then change settings for visual studio project linked with OpenCV:
http://docs.opencv.org/2.4.11/doc/tutorials/introduction/windows_visual_studio_Opencv/windows_visual_studio_Opencv.html



## How to Run
1.  print out ["acircles_pattern.png"](https://github.com/VisionerTech/stereo_calib_executable/blob/master/acircles_pattern.png) and stick it to a rigid surface, make it a "calibration board".
2.  compile the stereo_calib opencv project and run.
3.  place the "calibration board" in front of the camera, press "c" on the key board to capture an image pair(a blue flash appears with success capture). try to capture around 10 pair of the "calibration board" with different angle orientation and distance, try to cover the whole field of view of the cameras.  you can watch a video sample here. the captured image pairs is list in /save_image/ folder.
![alt text](https://github.com/VisionerTech/stereo_calib_executable/blob/master/readme_image/calib_snap1.png "snap1")
![alt text](https://github.com/VisionerTech/stereo_calib_executable/blob/master/readme_image/calib_snap2.png "snap2")
4.  press "Esc" on the keyboard. program finds the pattern, perform stereo calibration and show the result. check the blue parallel lines to see left and right image is rectified.
![alt text](https://github.com/VisionerTech/stereo_calib_executable/blob/master/readme_image/rectified.png "rectified")
5.  press "Esc" on the keyboard to save the result to /save_param/ folder. copy the whole folder to where it's needed.
![alt text](https://github.com/VisionerTech/stereo_calib_executable/blob/master/readme_image/saved_files.png "saved_files")
