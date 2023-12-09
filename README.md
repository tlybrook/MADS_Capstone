# MADS_Capstone
We leverage pickle files in our project so please make sure you have Python version 3.10.11

## Setup 
1. Change the current working directory to the location where you want the cloned directory. 
2. Clone the repository by running the following line in terminal:

        git clone https://github.com/tlybrook/MADS_Capstone.git

    For additional help on cloning a repository please look here: https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository

3. Open the cloned repository.
4. Ensure the Python version you are using is 3.10.11. Run the following to check:

        python --version

    If you do not have the correct version please run the following:

        pip uninstall python
        pip install python 3.10.11

4. Next, let's create a virtual enviornment by running the following in terminal:

        python -m venv .venv

5. Activate the virtual environment. 

    In Windows terminal run:

        .\venv\Scripts\activate

    In Mac terminal run:

        source venv/bin/activate

6. Next, download all the needed libraries in the virtual environment.

        pip install -r requirements.txt

## Flask App Instructions
1. Ensure you have run the requirements.txt file to download all necessary libraries in your virtual environment. 
2. Change directory to the flask_application folder by running:

        cd flask_application

3. Next, in terminal you can simply run: 

        flask run

4. After a few seconds, click on the local host URL that shows up in your terminal and this will redirect you to the web where you can interact with our flask application.
5. From here you can learn about us and our project, navigate to our GitHub, and upload a Lung CT Scan and generate a prediction using our model. 
Please note that this flask application is intended only for research purposes and should not be used for medical decision making. 

## Directions for replicating our work
### Data Reading and Cleaning
First, download the data from kaggle: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images and move to root folder where you downloaded this repository. 

In your terminal run: 

    python data_read.py

This will read in the data, remove duplicates, and determine the average image dimensions that we use to resize each image to for consistency. 

### Logsitic Regression
To replicate the logistic regression model, after you have the cleaned data and successfuly created final_dataset folder then you can just run the following command in terminal:

    python logistic_regression.py

This will print the train accuracy, test accuracy, train recall, and test recall. It will also generate a confusion matrix visualization and output it to the visualization_outputs folder.

## Other file information
### Visualizations
The visualizations.py contains functions for all the visualization for our project. This includes confusion matrix, line chart showing metrics over each CNN epoch, and a heatmap to show CNN convolution layer features.