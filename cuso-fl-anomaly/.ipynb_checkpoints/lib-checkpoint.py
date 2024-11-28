import logging
from logging import DEBUG, ERROR, INFO, WARNING
LOGGER_NAME = "flwr"
FLOWER_LOGGER = logging.getLogger(LOGGER_NAME)
FLOWER_LOGGER.setLevel(logging.DEBUG)
logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name
log = logger.log  # pylint: disable=invalid-name


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch

def one_hot_encode(df, cols):
    '''
    Takes all column names in cols and one-hot-encodes them.
    '''

    df_combined = df.copy()

    for col in cols:
        if col not in df.columns:
            continue

        # Initialize OneHotEncoder
        one_hot_encoder = OneHotEncoder(sparse_output=False)

        # Fit and transform the 'color' column
        encoded = one_hot_encoder.fit_transform(df_combined[[col]])
        
        # Get new column names for one-hot encoded
        colnam = one_hot_encoder.get_feature_names_out([col])

        # Convert to DataFrame        
        encoded_df = pd.DataFrame(encoded, columns=colnam, index=df_combined.index)
        
        # Combine with original DataFrame
        df_combined = pd.concat([df_combined, encoded_df], axis=1)
        # print(df_combined)

        # Remove the converted feature
        df_combined.drop(columns=[col], inplace=True)

    return df_combined


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def training(model, optimizer, criterion, train_loader, device):
    '''
    Train model on training dataset for one epoch
    '''
    
    # Set the model to training mode
    model.to(device)
    model.train()

    # Initialize epoch-wise cumulative training metrics
    train_loss = 0.0
    train_preds, train_targets = [], []

    # for inputs, labels in train_loader:
    for inputs, labels in train_loader:
        # print(inputs.shape)
        # print(labels.shape)
        # print(inputs)
        # print(labels)

        # Move Tensors containing inputs and labels to GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
                
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Aggregate metrics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, targets = torch.max(labels.data, 1)
        train_preds.extend(predicted.cpu().numpy())
        train_targets.extend(targets.cpu().numpy())

    # Compute epoch-wise training metrics
    train_loss /= len(train_loader)
    train_acc = accuracy_score(train_targets, train_preds)
    train_prec = precision_score(train_targets, train_preds, average='weighted', zero_division=0)
    train_rec = recall_score(train_targets, train_preds, average='weighted', zero_division=0)
    train_f1 = f1_score(train_targets, train_preds, average='weighted', zero_division=0)
    
    # Pack training metrics
    train_metric = {'loss': train_loss, 'accuracy': train_acc, 'precision': train_prec, 'recall': train_rec, 'f1': train_f1}

    return train_metric

def evaluation(model, criterion, test_loader, device):
    '''
    Evaluate model performance on test dataset (one epoch)
    '''
    
    # Set the model to evaluation mode
    model.eval()

    # Initialize epoch-wise cumulative test metrics
    test_loss = 0
    test_preds, test_targets = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
    
            # Compute loss            
            loss = criterion(outputs.squeeze(), labels)
            
            # Aggregate metrics
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, targets = torch.max(labels.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    
    # Compute epoch-wise test metrics
    test_loss /= len(test_loader)
    

    # Compute validation metrics
    test_acc = accuracy_score(test_targets, test_preds)
    test_prec = precision_score(test_targets, test_preds, average='weighted', zero_division=0)
    test_rec = recall_score(test_targets, test_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(test_targets, test_preds, average='weighted', zero_division=0)

    # Pack test metrics
    test_metric = {'loss': test_loss, 'accuracy': test_acc, 'precision': test_prec, 'recall': test_rec, 'f1': test_f1}

    return test_metric

import numpy as np
from collections import OrderedDict
from typing import List, Tuple

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

from torch.utils.data import Dataset

class XyDataset(Dataset):
    def __init__(self, X, y):
        # X and y are PyTorch tensors
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define Model
import torch.nn as nn
class FFNN_Tiny(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FFNN_Tiny, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(train_metrics, test_metrics):
    epochs = range(1, len(train_metrics) + 1)
    train_df = pd.DataFrame(train_metrics)
    test_df = pd.DataFrame(test_metrics)

    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    titles = ['Loss', 'Accuracy', 'Precision', 'Recall' , 'F1-score']

    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(5, 4))
        plt.plot(epochs, train_df[metric], 'b', label='Training ' + title)
        plt.plot(epochs, test_df[metric], 'r', label='Test ' + title)
        plt.title('Training and Test ' + title)
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.legend()
        plt.show()

from typing import List, Tuple
from flwr.common import Metrics

# TODO: This may not necessarily keep order across rounds
def all_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    '''
    Takes a list of metrics and aggregates them by type into a dictionary.

    Args:
        metrics: A list of tuples (num_examples: int, metric: Dict), where:
            num_example: is the integer representing the number of samples used to train the client for this round.
            metric: is a dictionary containing the metric_name and metric_value

     
    Returns:
        A dictionary containing the aggregated metrics

    Example:
        input: [(10, {'accuracy': 0.5}), (20, {'accuracy': 0.6}), (30, {'accuracy': 0.7})]
        output: {'num_examples': [10, 20, 30], 'accuracy': [0.5, 0.6, 0.7]}
    '''  

    log(INFO, "Client metrics: " + str(metrics))
    
    # Initialize an empty dictionary to store the results
    result_dict = {'num_examples': []}

    # Iterate through the list of tuples
    for num_examples, metrics in metrics:
        result_dict['num_examples'].append(num_examples)  # Append the number of examples
        for key, value in metrics.items():
            if key not in result_dict:
                result_dict[key] = []  # Create a new list if the key is not already in the dict
            result_dict[key].append(value)  # Append the value to the list for that key
    return  result_dict 

######################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_metrics(training_metrics_df, test_metrics_df):
    '''
    Example DataFrame structures
    training_metrics_df and test_metrics_df would be the collected data after training
    training_metrics_df should have columns: ["round", "client_id", "loss", "accuracy", ...] (other client metrics)
    test_metrics_df should have columns: ["round", "loss", "accuracy", ...] (other server evaluation metrics)   
    '''

    # Set up a grid of subplots based on the number of metrics
    # metrics = [col for col in training_metrics_df.columns if col not in ["round", "client_id"]]
    metrics = ['accuracy','f1','loss','precision', 'recall'] # Excluded:'client_id','round'
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20,8))
    axs = axs.flatten() # Access subplots using a single index

    # Iterate over each metric to create a line plot with confidence interval and a violin plot
    for i, metric in enumerate(metrics):
                
        # Line plot with confidence interval for training metrics
        sns.lineplot(
            data=training_metrics_df,
            x=training_metrics_df["round"],
            y=training_metrics_df[metric],
            ax=axs[i],
            label="Train",
        )

        sns.lineplot(
            x=test_metrics_df["round"],
            y=test_metrics_df[metric],
            ax=axs[i],
            color="red",
            label="Test",
            # linewidth=4
        )

        axs[i].set_xticks(range(training_metrics_df["round"].min(), training_metrics_df["round"].max()+1))
        # axs[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axs[i].set_xlabel("Round")
        axs[i].set_ylabel(metric)
        axs[i].legend()

    # Adjust layout for aesthetics
    plt.tight_layout()
    plt.show()

def plot_metrics_full(training_metrics_df, *args):
    # , test_metrics=None,
    #                   centralized_training_metrics=None, centralized_test_metrics=None,
    #                    single_training_metrics=None, single_test_metrics=None):
    plt.clf()

    metrics = ['accuracy','f1','loss','precision', 'recall'] # Excluded:'client_id','round'
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20,8))
    axs = axs.flatten() # Access subplots using a single index

    # Iterate over each metric to create a line plot with confidence interval and a violin plot
    for i, metric in enumerate(metrics):

        sns.boxplot(
            data=training_metrics_df,
            x="round",
            y=metric,
            ax=axs[i],
            width=0.3,
            positions=training_metrics_df['round'].unique(),
            boxprops=dict(alpha=0.2),  # Adjust box transparency #facecolor="skyblue", 
            whiskerprops=dict(alpha=0.2),  # Whisker transparency #color="blue", linewidth=1.5, 
            flierprops=dict(alpha=0.2),  # Outlier transparency #markerfacecolor='r', marker='o', 
            medianprops=dict(alpha=0.2)
        )
                
        # Line plot: training metrics for each client without confidence interval (hue='client_id')
        sns.lineplot(
            data=training_metrics_df,
            x=training_metrics_df["round"],
            y=training_metrics_df[metric],
            ax=axs[i],
            legend=False,
            alpha=0.3,
            hue="client_id"
        )

        # Line plot: mean training metrics (with confidence interval) all clients
        sns.lineplot(
            data=training_metrics_df,
            x=training_metrics_df["round"],
            y=training_metrics_df[metric],
            ax=axs[i],
            legend=False,
            color="blue"
        )

        # Plot additional metrics
        for arg in args:
            if arg is not None:
                arg_title = arg[0]
                arg_df = arg[1]
                if metric in arg_df.columns:
                    sns.lineplot(
                        x=arg_df["round"],
                        y=arg_df[metric],
                        ax=axs[i],
                        label=arg_title,
                        linewidth=3
                    )

        axs[i].set_xticks(range(training_metrics_df['round'].min(), training_metrics_df['round'].max()+1))
        axs[i].set_xticklabels(range(training_metrics_df['round'].min(), training_metrics_df['round'].max()+1))
                
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axs[i].set_xlabel("Round")
        axs[i].set_ylabel(metric)
        
        # Add legend if it exists
        if axs[i].get_legend():
            axs[i].legend()

    # Adjust layout for aesthetics
    plt.tight_layout()
    plt.show()