import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def eval_model(model,test_set,device):
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False) # We don't need to shuffle test data
    correct_predictions = 0
    total_samples = 0
    model.eval()

    with torch.no_grad(): #  Turn off gradient tracking to save memory and compute power
        for batch_img, batch_label in test_loader:
            batch_img, batch_label = batch_img.to(device), batch_label.to(device)
            predictions = model(batch_img)
            _, predicted_classes = torch.max(predictions, 1)
            correct_predictions += (predicted_classes == batch_label).sum().item()
            total_samples += batch_label.size(0)

    final_accuracy = (correct_predictions / total_samples) * 100

    print(f"Test Set Accuracy: {final_accuracy:.2f}% ({correct_predictions}/{total_samples} correct)")
    return final_accuracy


if __name__=='__main__':
    final_accuracy = eval_model(model,test_set)