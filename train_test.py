import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unet import UNet
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
import os
import pickle

# source files for training data
input_data_path = '/home/dario/Desktop/FlameSentinels/TILES_INPUT_DATA'
labels_path = '/home/dario/Desktop/FlameSentinels/TILES_LABELS'



# Load unified input data (contains bands + NDVI + NDMI)
print("Loading unified input data from TILES_INPUT_DATA...")
input_files = sorted(glob.glob(os.path.join(input_data_path, '*.npy')))
label_files = sorted(glob.glob(os.path.join(labels_path, '*.npy')))

print(f"Found {len(input_files)} input files and {len(label_files)} label files")

# Load all input data (15 channels: 13 bands + NDVI + NDMI)
x = np.array([np.load(f) for f in input_files])
print(f"Input data shape: {x.shape}")

# Try to load band info to understand channel organization
try:
    # Look for any band info file to understand the channel structure
    band_info_files = glob.glob(os.path.join(input_data_path, '*_band_info.pkl'))
    if band_info_files:
        with open(band_info_files[0], 'rb') as f:
            band_info = pickle.load(f)
        print(f"Band information loaded: {band_info['band_names']}")
        print(f"Total channels: {len(band_info['band_names'])}")
        band_names = band_info['band_names']
    else:
        print("No band info file found, assuming standard order")
        band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12', 'NDVI', 'NDMI']
except Exception as e:
    print(f"Could not load band info: {e}")
    band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12', 'NDVI', 'NDMI']



# Extract specific channels for the model
# RGB bands (B02, B03, B04), NIR (B08), SWIR (B11), NDVI, NDMI
try:
    # Find indices of the bands we want
    b02_idx = band_names.index('B02')  # Blue
    b03_idx = band_names.index('B03')  # Green 
    b04_idx = band_names.index('B04')  # Red
    b08_idx = band_names.index('B08')  # NIR
    b11_idx = band_names.index('B11')  # SWIR
    ndvi_idx = band_names.index('NDVI')  # NDVI
    ndmi_idx = band_names.index('NDMI')  # NDMI
    
    selected_indices = [b02_idx, b03_idx, b04_idx, b08_idx, b11_idx, ndvi_idx, ndmi_idx]
    print(f"Selected channel indices: {selected_indices}")
    print(f"Selected channels: {[band_names[i] for i in selected_indices]}")
    
except ValueError as e:
    print(f"Error finding band indices: {e}")
    print("Using default indices [1, 2, 3, 7, 11, 13, 14]")
    selected_indices = [1, 2, 3, 7, 11, 13, 14]

# Extract selected channels
x = x[:, :, :, selected_indices]
print(f"Selected input data shape: {x.shape}")

# Load labels
y = np.array([np.load(f) for f in label_files])
print(f"Labels shape: {y.shape}")

# Convert to tensors
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Convert from [num_samples, height, width, channels] to [num_samples, channels, height, width]
x = x.permute(0, 3, 1, 2)
y = y.permute(0, 3, 1, 2)

print(x.shape)
print(y.shape)

# train-val split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=33)

# create datasets and dataloaders
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)


# main train loop
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    best_val_loss = float('inf')

    not_improved_epochs = 0

    for epoch in range(epochs):

        # put model in training mode (needed for batch normalization)
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            x, y = x.to(device), y.to(device).float()

            optimizer.zero_grad()
            # output probabilities
            outputs = model(x)
            # loss calculation
            loss = criterion(outputs, y)
            # backpropagation
            loss.backward()
            # gradient descent
            optimizer.step()

            train_loss += loss.item()

            outputs = outputs.cpu().detach()
            all_preds.extend(outputs.numpy())
            all_labels.extend(y.cpu().numpy())

        # epoch training metrics calculation
        train_loss /= len(train_loader)
        
        # Flatten arrays for metrics calculation
        all_preds_flat = np.array(all_preds).flatten()
        all_labels_flat = np.array(all_labels).flatten()
        
        train_mse = mean_squared_error(all_labels_flat, all_preds_flat)
        train_mae = mean_absolute_error(all_labels_flat, all_preds_flat)

        # validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        # no update of parameters
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                x, y = x.to(device), y.to(device).float()

                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item()

                outputs = outputs.cpu().detach()
                all_val_preds.extend(outputs.numpy())
                all_val_labels.extend(y.cpu().numpy())

        # validation metrics calculation
        val_loss /= len(val_loader)
        
        # Flatten arrays for metrics calculation
        all_val_preds_flat = np.array(all_val_preds).flatten()
        all_val_labels_flat = np.array(all_val_labels).flatten()
        
        val_mse = mean_squared_error(all_val_labels_flat, all_val_preds_flat)
        val_mae = mean_absolute_error(all_val_labels_flat, all_val_preds_flat)

        # print results update
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f} | MSE: {train_mse:.4f} | MAE: {train_mae:.4f}")
        print(f"  Val Loss: {val_loss:.4f}   | MSE: {val_mse:.4f} | MAE: {val_mae:.4f}\n")

        # saving best model at each epoch, if val loss improves
        if val_loss < best_val_loss:
            not_improved_epochs = 0
            best_val_loss = val_loss
            # Save the entire model (architecture + weights)
            torch.save(model.state_dict(), f"/home/dario/Desktop/FlameSentinels/best_prediction_model.pth")
            print(f"  --> Saved best model (val_loss improved)\n")
        else: 
            not_improved_epochs += 1
            if not_improved_epochs == 5:
                print(f"Model training stopped early due to no improvement for f{not_improved_epochs} epochs.\n")
                return model
        


    print("\nTraining completed!")
    return model

def evaluate_model(model, test_loader, device, output_dir='/home/dario/Desktop/FlameSentinels/TEST_PRED_IMGS'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device).float()

            outputs = model(x).squeeze(1)
            preds = outputs.cpu().detach()

            all_preds.extend(preds.numpy())
            all_labels.extend(y.cpu().numpy())

            # Save each prediction in the batch
            for j, pred in enumerate(preds):
                # Convert to uint8 and squeeze if single channel
                pred_img = (pred.squeeze() * 255).astype(np.uint8)
                img = Image.fromarray(pred_img, mode='L' if pred_img.ndim == 2 else 'RGB')
                img.save(os.path.join(output_dir, f'prediction_{i*len(preds)+j}.png'))

        all_preds_flat = np.array(all_preds).flatten()
        all_labels_flat = np.array(all_labels).flatten()

    mse = mean_squared_error(all_labels_flat, all_preds_flat)
    mae = mean_absolute_error(all_labels_flat, all_preds_flat)

    print(f"\nTest MAE: {mae:.4f} | Test MSE: {mse:.4f}")

#-------------------------------------------------------------------

# architecture definition (setting n channels as following x_train channels)
unet_model = UNet(x_train.shape[1])

"""
# optimizer & loss definition
optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001)
criterion = nn.L1Loss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


trained_model = train_model(unet_model, train_loader, val_loader, criterion, optimizer, device, 25)
"""
#---------------------------------------------------------------

# to reload previous model
unet_model.load_state_dict('/home/dario/Desktop/FlameSentinels/best_prediction_model.pth')




evaluate_model(unet_model, )