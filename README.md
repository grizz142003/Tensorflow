# Tensorflow
Learning and going through tensorflow documentation   
Using this opportunity to also learn how github works as well (2 birds with 1 stone)  

## Venv
Using A virtual environment with tensorflow installed. there is also other libraries such as pyplot used which are also installed in the Venv not sure how to integrate that whole folder to github so Im copy pasting the code over to the files in this repo for now  
  
Setting up the Venv is relatively simple follow along with the official tensorflow Documentation https://www.tensorflow.org/install/pip   

## Venv Using Bash Script
I also created a simple bash script which you can run which should create a venv with the necessary modules to run the code  
Before running the gpu script make sure you have proper drivers install for your gpu

1. Download the script from the repository,
2. Make sure script is inside Directory where you want to create your Python virtual environment
3. Right click on the script and click "Run as a Program"

NOTE: SCRIPT IS MADE ON UBUNTU SYSTEM AND IS NOT TESTED FOR OTHER OPERATING SYSTEMS IF YOU ARE HAVING ISSUES ITS CAUSE YOUR NOT ON UBUNTU
## Skipped TFdoc 3 (not the actual document just the name)
## TFdoc 4 ISSUE
when attempting to run the code inside TFdoc4 posted in this repo, you may run into an issue, this can be resolved by switching to google collab and running these 2 lines of code before running any code.  
pip install tensorflow==2.13.0  
tensorflow-hub==0.13.0  
The issue when running the code outside google collab is an incompatability between versions of tensorflow hub and tensorflow itself. To fix them you would need to install these two on your local Virtual Env but they are not available from the command line, the oldest available versions for both are 2.16 and 0.16 which unfortunately are not compatible. 

unfortunately I haven't figured out how to install older versions of tensorflow on my ubuntu machine, Moved onto the next learning document, will revisit later.

## TFdoc 8  ISSUE 
Data set not working, load it from iternet, doesn't work in google collab or on local machine.  
Cant be asked to figure it out not sure where the dataset is being saved on local machine to even check the issue with it so im leaving it for now skipped to TFdoc 9  
