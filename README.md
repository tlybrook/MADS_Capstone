# MADS_Capstone
We leverage pickle files in our project so please make sure you have Python version 3.10.11
Also, to ensure you have all the right libraries please download the requirements.txt.

## Flask App Instructions
1. Ensure you have run the requirements.txt file to download all necessary libraries in your virtual environment. 
2. Change directory to the flask_application folder by running "cd flask_application"
3. Next, you can simply run "flask run" in terminal.
4. After a few seconds, click on the local host URL that shows up in your terminal and this will redirect you to the web where you can interact with our flask application.
5. From here you can learn about us and our project, navigate to our GitHub, and upload a Lung CT Scan and generate a prediction using our model. 
Please note that this flask application is intended only for research purposes and should not be used for medical decision making. 

## Directions for replicating our work
### Data Reading and Cleaning
First, download the data from kaggle: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images and move to root folder where you downloaded this repository. 

Run python data_read.py in your terminal. This will read in the data, remove duplicates, and determine the average image dimensions that we use to resize each image to for consistency. 

### Logsitic Regression
To replicate the logistic regression model, after you have the cleaned data and successfuly created final_dataset folder then you can just run the following command in terminal:
python logistic_regression.py 

This will print the train accuracy, test accuracy, train recall, and test recall. It will also generate a confusion matrix visualization and output it to the visualization_outputs folder.

## Other file information
### Visualizations
The visualizations.py contains functions for all the visualization for our project. This includes confusion matrix, line chart showing metrics over each CNN epoch, and a heatmap to show CNN convolution layer features.