# Overview
This app give you ability to score images and video files frame by frame based on onnx model. The model was trained using CustomVision.ai and exported to be used in UWP APP written on C#. The app is based on example of Windows ML sdk from [here](https://github.com/Microsoft/Windows-Machine-Learning) and extended to use with video files.

## Model
Model was trained on data-set that sits here https://acronishackstorage.blob.core.windows.net/race-events-recognition/Pitstop-different-views-dataset/ there is 6 different classes that model is trying to classify images: 
1. Pitstops, 
2. 1st party view, 
3. 3d party view, 
4. Command center, 
5. People,
6. 1st party back view.

## Requirements
- [Visual Studio 2017 Version 15.7.4 or Newer](https://developer.microsoft.com/en-us/windows/downloads)
- [Windows 10 - Build 17738 or higher](https://www.microsoft.com/en-us/software-download/windowsinsiderpreviewiso)
- [Windows SDK - Build 17738 or higher](https://www.microsoft.com/en-us/software-download/windowsinsiderpreviewSDK)

## Dependencies
- [Win2D](https://github.com/Microsoft/Win2D)
