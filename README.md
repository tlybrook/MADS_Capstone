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

5. Next, let's create a virtual enviornment by running the following in terminal:

        python -m venv venv

6. Activate the virtual environment. 

    In Windows terminal run:

        .\venv\Scripts\activate

    In Mac terminal run:

        source venv/bin/activate

7. Next, download all the needed libraries in the virtual environment.

        pip install -r requirements.txt

If you do not have pip installed please follow instructions here: https://pip.pypa.io/en/stable/installation/

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

### Final CNN
To recreate our final CNN, you can run the following command:

    python final_pytorch_cnn.py

Output from the model will print to the console including charts and metrics.

## Flask App Instructions
1. Ensure you have run the requirements.txt file to download all necessary libraries in your virtual environment. 
2. Change directory to the flask_application folder by running:

        cd flask_application

3. Next, in terminal you can simply run: 

        flask run

4. After a few seconds, click on the local host URL that shows up in your terminal and this will redirect you to the web where you can interact with our flask application.
5. From here you can learn about us and our project, navigate to our GitHub, and upload a Lung CT Scan and generate a prediction using our model. 

Please note that this flask application is intended only for research purposes and should not be used for medical decision making. 

## Files in Processes folder
### visualizations.py
The visualizations.py contains functions for all the visualization for our project. This includes confusion matrix, line chart showing metrics over each CNN epoch, and a heatmap to show CNN convolution layer features.

### utils.py
This file mainly contains the functions that help us build and manage our model tracking .pickle
files. 

### preprocessing.py
This file contains the functions for preprocessing our data. It has the functions we used from both Keras and PyTorch.

### model_designs_pytorch.py
This file contains all of the model structures we tested in PyTorch. It also has the functions we used to train our models and predict using those models. 

## Additional File Information
### pytorch_cnn_tuning
This is the file we used to help tune our CNN. We import our candidate model structures
and save the results of each model to a .pickle file so we can always compare any two models.
There is a file called pytorch_model_tracker.pickle in the repo which has all our model runs saved and serves as our 'candidate model inventory'. You can also delete this file from your local repository and start running models to create a new model inventory.

### vgg_pytorch_cnn.py
This file will run the VGG16 model structure on our dataset. This is one of the first models
that we tried and since it performs so much worse than the rest of our candidate models,
we decided to keep the code for this model separate so it is more convenient to run this example.
The structure of this file is nearly identical to final_pytorch_cnn.py

### cnn_keras.py
This was the file we originally used to build our models in keras. We also created a 
model_tracker file for this script but shortly after, we had a hard time reconciling the model results. We tried PyTorch and got it working so we stopped using this file and Keras altogether but decided to keep the code in here as reference for researchers.

### _model_designs_keras.py
This file is not meant to be run. It just stores our 13 model designs that we wanted to test 
in Keras. We found a better way to manage our model designs once we switched to PyTorch 
but we still wanted to keep this in here as reference.