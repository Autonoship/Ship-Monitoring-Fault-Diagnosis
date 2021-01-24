# Anomaly-Detection
Apply deep learning methods on time series data to detect anomaly

Dataset taken from: [NAB_Datasets](https://github.com/numenta/NAB)
# Data-Augmentation
Augmenting image data with PyTorch

# Defect-Detection
A Machine Learning Approach for Engine Room Monitoring and Fault Prediction

Dataset taken from: [NEU_Datasets](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html)

## Installation Steps:

1.Install Anaconda3.

2.Install PyCharm – Community Version.

3.Clone the repository, navigate to the folder where the repository is located. 

4.Create a conda environment to run the code. Run following commands in the terminal of Pycharm.

```
conda env create -f environment.yml
conda activate Defect_detection
```
## Usage:
Run
```
python Training_Window.py
```
to train the model. Follow the steps in the "Guidelines" file.

## Train the model with other datasets(credited to Honda san from BEMAC):
1.If dataset image is in color, so you need to convert the color image to gray.

* Put dataset (with the same structure) in Defect-Detection-master.
* Modify function "bundle_grayscale" arguments to dataset name.
* Execute "python convert2grayscale.py"

2.Create new datasets folder with label names and put the corresponding data inside it, like an existing dataset.

3.Change the variable "names" in GLCM.py and LBGLCM.py to the corresponding label names.

4.Change the variable "class_labels" and the paths like "data_Cr_dir" in Classifiers.py to the correspondings.

5.Change the variables "IMG_HEIGHT" and "IMG_WIDTH" in Classifiers.py to the correspondings.

6.Change the output layer of function "create_model" to the corresponding.

7.Trained models are saved in the "Models" folder.

## Design GUI

1.Activate "Defect_detection" environment in Pycharm. Type in ‘pip list’ in the terminal of Pycharm to check whether you have installed PyQt5 and pyqt5-tools. You can use ‘pip show PyQt5’ to know more information about PyQt5.

2.If PyQt5 and pyqt5-tools are installed, type in ‘pyqt5designer’ to run the GUI designer.

3.If pyqt5-tools is not installed because of some errors,
```
a.For Windows:
Use “Qt_Designer_Setup.exe” on the website(See Issues) to install “Qt Designer” program.
b.For Mac:
Use “Qt Designer.dmg” on the website(See Issues) to install “Qt Designer” program. You need to go to your 
security settings and allow the application to be downloaded.
This GUI tool will create .ui files (e.g., myTest.ui) after you save your design.
```
4.After finishing the GUI design, convert .ui file into .py file 
```
a.Type in ‘pyuic5 -x myGUI.ui -o myTest.py’ in the terminal of Pycharm to convert ‘myGUI.ui’ to ‘myTest.py’.
b.You can use ‘pyuic5 --help’ to know more information about pyuic5.
c.If your system doesn’t recognize ‘pyuic5’, type in ‘where pyuic5’ in the terminal and use the
full path of ‘pyuic5’. For example, ‘C:\Anaconda3\envs\environment_name\Scripts\pyuic5.exe -x myGUI.ui -o myTest.py’.
```
5.Incorporate myGUI.py into PyCharm Project folder.

## Issues:
Python-Include Packages:
HDD:/Library/Frameworks/Python.Framework/Versions/3.6/lib/Python3.6/*****

Site Packages:
HDD:/Library/Frameworks/Python.Framework/Versions/3.6/lib/Python3.6/site-packages/*****

Qt Designer Download Website:
https://build-system.fman.io/qt-designer-download
