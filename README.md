# Training 
Create an experiments folder in the parent directory of ./code/

    mkdir ./experiments/

Edit the cfg dictionary in train.py according to the desired specifications
Move to the parent directory and run the train.py script

    python ./code/train.py

During training an automatic experiment name is generated and the best model weights will be stored at ./experiments/<experiment_name>/best_weights.pt

The cfg dictionary is saved as ./experiments/<experiment_name>/cfg.pt

The tracker CSV with train and validation accuracies and losses is saved as ./experiments/<experiment_name>/tracker.csv 

# Testing
For testing a trained model simply provide the experiment name and the shot/way/query regime you want to test it on 

    python ./code/test.py ./experiments/<experiment_name> <shot> <way> <query>
