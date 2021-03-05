# Home Value Predictor


## Setup

Create a conda environment as defined in ```environment.yml```. 
```
conda env create -f environment.yml
conda activate home-value-predictor
```

We will use pip-tools to install python libraries for a couple reasons:
- Separate out dev from production dependencies(```requirements-dev.in``` vs ```requirements.in```).
- Pin exact versions for all dependencies (the auto-generated ```requirements-dev.txt``` vs ```requirements.txt```).
- Allow us to easily deploy to targets that may not support the ```conda``` environment.
```
pip-sync requirements.txt requirements-dev.txt
```
If you add, remove, or need to update versions of some requirements, edit the ```.in``` files, then fun
```
pip-compile requirements.in && pip-compile requirements-dev.in
```

If you recieve any errors like ```ModuleNotFoundError: No module named 'home_value_predictor'``` you may need to update your path.
```
export PYTHONPATH=.
```

## Running the code

Once you have your environment setup, you can view the notebook ```test.ipynb``` to see how to run some of the code. 

The code is setup in a way to be easily extendable for fast iteration and experimentation. 

To create a dataset from the given train.csv file, run 
```
data = HomeDataset()
```
You can then load a processed version of the data with all features ready for training
```
processed_df = data.load_data(processed=True)
```
Split the data into train/test
```
X_train, X_test, y_train, y_test = data.split_data(processed_df)
```
Create a xgb_regressor model for training
```
xgb_regressor = XGBoostModel()
```
Create a dictionary of training parameters you wish to use for this experiment
```
parameters = {'n_estimators':range(10, 200, 10), 
              'min_samples_leaf':range(5, 40, 5), 
              'max_depth':range(3, 5, 1)}
```
Then pass the parameters to the ```xgb_regressor``` 
```
xgb_regressor.train(X_train, y_train, params=test_params, save_best=True)
```
which will find the best model and save it to use later for inference


## Future Improvements

My next goal for this project was to implement a script to take a config file as a parameter and run an experiment with a simple command. All is needed to do is to edit the config file to specify which version of the data you'd like to use, which model you'd like to use, and specifiy the hyperparameters for the model and grid/random search. Once other models and other datasets/other versions of the dataset are implemented, you can imagine how much easier it would be to run different experiments with whichever data/model/hyperparameters you'd like.