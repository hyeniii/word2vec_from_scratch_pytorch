import os
import numpy as np
import json
import torch
from dataclasses import dataclass
from typing import Any

@dataclass
class Trainer:
    """Main class for model training. Handles the training and validation processes, checkpointing, and saving the final model."""
    model: torch.nn.Module  # PyTorch model to be trained
    epochs: int  # Number of training epochs
    train_dataloader: torch.utils.data.DataLoader  # DataLoader for training data
    train_steps: int  # Number of steps per training epoch
    val_dataloader: torch.utils.data.DataLoader  # DataLoader for validation data
    val_steps: int  # Number of steps per validation epoch
    checkpoint_frequency: int  # Frequency of saving checkpoints
    criterion: Any  # Loss function
    optimizer: Any  # Optimizer for training
    lr_scheduler: Any  # Learning rate scheduler
    device: str  # Device to train the model on (e.g., 'cuda' or 'cpu')
    model_dir: str  # Directory to save model checkpoints
    model_name: str  # Name of the model

    def __post_init__(self):
        # Initialize training and validation loss storage
        self.loss = {"train": [], "val": []}
        # Move model to the specified device
        self.model.to(self.device)
    
    def train(self):
        """Conducts the training and validation process across all epochs."""
        for epoch in range(self.epochs):
            # Train for one epoch
            self._train_epoch()
            # Validate for one epoch
            self._validate_epoch()
            print(f"Epoch: {epoch + 1}/{self.epochs}, Train Loss={self.loss['train'][-1]:.5f}, Val Loss={self.loss['val'][-1]:.5f}")

            # Step the learning rate scheduler
            self.lr_scheduler.step()

            # Save a checkpoint if applicable
            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)
    
    def _train_epoch(self):
        """Handles the training process for a single epoch."""
        self.model.train()  # Set model to training mode
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # Forward pass
            outputs = self.model(inputs)
            # Compute loss
            loss = self.criterion(outputs, labels)
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Accumulate loss
            running_loss.append(loss.item())

            # Break if the number of training steps is reached
            if i == self.train_steps:
                break
        
        # Compute and store the average loss for the epoch
        epoch_loss = np.mean(running_loss)
        self.loss['train'].append(epoch_loss)
    
    def _validate_epoch(self):
        """Handles the validation process for a single epoch."""
        self.model.eval()  # Set model to evaluation mode
        running_loss = []

        with torch.no_grad():  # Disable gradient computation
            for i, batch_data, in enumerate(self.val_dataloader, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                # Compute loss
                loss = self.criterion(outputs, labels)
                # Accumulate loss
                running_loss.append(loss.item())

                # Break if the number of validation steps is reached
                if i == self.val_steps:
                    break
            # Compute and store the average loss for the epoch
            epoch_loss = np.mean(running_loss)
            self.loss['val'].append(epoch_loss)

    def _save_checkpoint(self, epoch):
        """Saves a checkpoint of the model."""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = os.path.join(self.model_dir, f"checkpoint_{str(epoch_num).zfill(3)}.pt")
            torch.save(self.model, model_path)

    def save_model(self):
        """Saves the final trained model."""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Saves the training and validation loss history."""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)          
