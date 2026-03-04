import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CnnNet
from evaluation import eval_model
from dataset import dataprep
import yaml, os
import random
import numpy as np
import wandb

def set_seed(seed_value=42):
    """Sets the seed for reproducibility across the entire pipeline."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cwd = os.getcwd()
    print(f"Current Working Directory: {cwd}")

    with open(cwd+r"\src\config.yaml",'r') as cf:
        data = yaml.load(cf, Loader=yaml.SafeLoader)

    print("Status: Config file loaded! and proceeding to prepare data..")


    train_set, test_set = dataprep(path = data['paths']['data_dir'])

    print("Status: Data Preparation completed and train and test sets loaded! and preparing to load model..")


    model = CnnNet(in_channels=data['model']['in_channels'], hidden_size=data['model']['fc_hidden_size'], num_classes = data['model']['num_classes'], dropout_rate=data['model']['dropout_rate']).to(device)

    print("Status: Model loaded and preparing to start Training!")


    if data['training']['optimizer'] =='adam':
        optimizer = optim.Adam(model.parameters(),lr=data['training']['learning_rate'])
    else:
        optimizer = optim.SGD(model.parameters(),lr=data['training']['learning_rate'])

    criterion = nn.CrossEntropyLoss()
    epochs = data['training']['num_epochs']
    dataloader = DataLoader(train_set, batch_size=data['training']['batch_size'], shuffle=True, num_workers=data['system']['num_workers'])

    # Initialize W&B
    if data['wandb']['enabled']:
        wandb.init(project=data['wandb']['project'],config=data)

    print("Status: Starting Epoch Loop......................")
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch_img, batch_label in dataloader:
            batch_img, batch_label = batch_img.to(device), batch_label.to(device)
            optimizer.zero_grad()
            predictions = model(batch_img)
            loss = criterion(predictions, batch_label)
            loss.backward()
            optimizer.step()

            total_loss +=loss.item()

        avg_loss = total_loss/len(dataloader)
        print(f"epoch: {epoch+1}/{epochs} | avg_loss: {avg_loss}")
        
        if data['wandb']['enabled']:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
            })

    print("Status: Model Training Completed and Preparing for model evaluation!")

    final_accuracy = eval_model(model, test_set, device)

    if config['wandb']['enabled']:
        wandb.log({"test_accuracy": final_accuracy})


    # Save the learned weights
    torch.save(model.state_dict(), data['paths']['checkpoint_dir']+'/custom_cnn.pth')
    print(f"Model weights saved successfully - {final_accuracy} !")
    if data['wandb']['enabled']:
        artifact = wandb.Artifact("custom_cnn", type="model")
        artifact.add_file(data['paths']['checkpoint_dir']+'/custom_cnn.pth')
        wandb.log_artifact(artifact)
        wandb.finish()