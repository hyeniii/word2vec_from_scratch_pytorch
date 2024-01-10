import argparse
import yaml
import os 
import torch
import torch.nn as nn

from utils.dataloader import get_dataloader_and_vocab
from utils.trainer import Trainer 
import utils.helper as h

def train(config):
    # Create necessary directories for saving model artifacts
    if not os.path.exists(config['model_dir']):
        os.makedirs(config['model_dir'])

    # Load training data and vocabulary
    train_dataloader, vocab = get_dataloader_and_vocab(
        model_name = config['model_name'],
        ds_name = config['dataset'],
        ds_type = "train",
        data_dir = config['data_dir'],
        batch_size = config['train_batch_size'],
        shuffle = config['shuffle'],
        vocab = None
    )

    # Load validation data using the same vocabulary
    val_dataloader, _ = get_dataloader_and_vocab(
        model_name = config['model_name'],
        ds_name = config['dataset'],
        ds_type = "valid",
        data_dir = config['data_dir'],
        batch_size = config['val_batch_size'],
        shuffle = config['shuffle'],
        vocab = vocab
    )

    # Determine the vocabulary size
    vocab_size = len(vocab.get_stoi())
    print(f"Vocabulary size: {vocab_size}")

    # Initialize the model, loss criterion, optimizer, and learning rate scheduler
    model_class = h.get_model_class(config['model_name'])
    model = model_class(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()

    optimizer_class = h.get_optimizer_class(config['optimizer'])
    optimizer = optimizer_class(model.parameters(), lr=config['learning_rate'])
    lr_scheduler = h.get_lr_scheduler(optimizer, config['epochs'], verbose=True)

    # Set the device (GPU or CPU) for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the trainer with all the training components
    trainer = Trainer(
        model = model,
        epochs = config['epochs'],
        train_dataloader = train_dataloader,
        train_steps=config['train_steps'],
        val_dataloader = val_dataloader,
        val_steps = config['val_steps'],
        criterion = criterion,
        optimizer = optimizer,
        checkpoint_frequency = config['checkpoint_frequency'],
        lr_scheduler = lr_scheduler,
        device = device,
        model_dir = config['model_dir'],
        model_name = config['model_name'],
    )
    
    # Start the training process
    trainer.train()
    print("Training finished")

    # Save the model, loss history, vocabulary, and configuration
    trainer.save_model()
    trainer.save_loss()
    h.save_vocab(vocab, config['model_dir'])
    h.save_config(config, config['model_dir'])
    print("Model artifacts saved to folder:", config['model_dir'])

if __name__ == '__main__':
    # Parse arguments for configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, default='config.yaml', help='path to yaml config')
    args = parser.parse_args()

    # Load configuration from the yaml file
    with open(args.config, 'r') as f:
