import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from hoovenet.model import HoovesModel
from hoovenet.utils import get_dataloaders
from common.constants import LEARNING_RATE, NUM_EPOCHS, PATIENCE, MODEL_FOLDER, BEST_MODEL_FOLDER, BEST_MODEL_PATH
import time
import os
import glob

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_loss = val_loss

# Training and Validation Loop
def train(resume_training=False, learning_rate=LEARNING_RATE):
    # Ensure the directories exist
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    os.makedirs(BEST_MODEL_FOLDER, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HoovesModel().to(device)

    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss here
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.1)

    print(f"Using device: {device}")

    if resume_training:
        latest_model = sorted(glob.glob(os.path.join(MODEL_FOLDER, 'hoofnet_*.pth')))[-1]
        if os.path.exists(latest_model):
            model.load_state_dict(torch.load(latest_model, map_location=device))
            print(f"Resuming training from model: {latest_model}")
        else:
            print("No checkpoint found. Starting training from scratch.")

    date = time.strftime("%Y-%m-%d-%H-%M")
    writer = SummaryWriter(log_dir=f'runs/hoofnet_experiment_{date}')

    train_loader, val_loader = get_dataloaders()
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")

        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('Learning Rate', scheduler.optimizer.param_groups[0]['lr'], epoch)

        # Save after x epochs
        if epoch % 5 == 0:
            epoch_model_path = os.path.join(MODEL_FOLDER, f"hoofnet_{date}_epoch_{epoch}_{epoch_val_loss:.4f}.pth")
            torch.save(model.state_dict(), epoch_model_path)

        scheduler.step(epoch_val_loss)

        early_stopping(epoch_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    writer.close()
    print("Finished Training")
    print(f"Best model saved to: {BEST_MODEL_PATH}")

if __name__ == '__main__':
    train(resume_training=False, learning_rate=LEARNING_RATE)
