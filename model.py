import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
import os
import pickle


# Updated paths for unified data structure
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

def visualize_predictions(model, val_loader, device, num_examples=5):
    """
    Visualize model predictions on validation set
    """
    model.eval()
    examples_shown = 0
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            if examples_shown >= num_examples:
                break
                
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
            predictions = model(x_batch)
            
            # Move to CPU and convert to numpy
            x_batch = x_batch.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            predictions = predictions.cpu().numpy()
            
            # Show examples from this batch
            batch_size = x_batch.shape[0]
            for i in range(min(batch_size, num_examples - examples_shown)):
                row = examples_shown
                
                # Show RGB channels (B02, B03, B04 - indices 0, 1, 2 in our selected channels)
                rgb_img = x_batch[i, :3, :, :].transpose(1, 2, 0)
                rgb_img = rgb_img[:, :, [2, 1, 0]]
                # Normalize RGB for display
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                axes[row, 0].imshow(rgb_img)
                axes[row, 0].set_title('RGB Input (B02,B03,B04)')
                axes[row, 0].axis('off')
                
                # Show NDVI (channel 5 in our selected channels: B02, B03, B04, B08, B11, NDVI, NDMI)
                ndvi_img = x_batch[i, 5, :, :]
                im1 = axes[row, 1].imshow(ndvi_img, cmap='RdYlGn')
                axes[row, 1].set_title('NDVI')
                axes[row, 1].axis('off')
                plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
                
                # Show ground truth
                gt_img = y_batch[i, 0, :, :]
                im2 = axes[row, 2].imshow(gt_img, cmap='Reds')
                axes[row, 2].set_title('Ground Truth')
                axes[row, 2].axis('off')
                plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)
                
                # Show prediction
                pred_img = predictions[i, 0, :, :]
                im3 = axes[row, 3].imshow(pred_img, cmap='Reds')
                axes[row, 3].set_title('Prediction')
                axes[row, 3].axis('off')
                plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)
                
                examples_shown += 1
                if examples_shown >= num_examples:
                    break
    
    plt.tight_layout()
    plt.savefig('/home/dario/Desktop/FlameSentinels/validation_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved as 'validation_examples.png'")

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
            torch.save(model, f"/home/dario/Desktop/FlameSentinels/best_prediction_model_full.pth")
            print(f"  --> Saved best model (val_loss improved)\n")
        else: 
            not_improved_epochs += 1
            if not_improved_epochs == 5:
                print(f"Model training stopped early due to no improvement for f{not_improved_epochs} epochs.\n")
                return model
        


    print("\nTraining completed!")
    return model

unet_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=7, out_channels=1, init_features=32, pretrained=False)

# optimizer & loss definition
optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001)
criterion = nn.L1Loss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


trained_model = train_model(unet_model, train_loader, val_loader, criterion, optimizer, device, 25)


# Visualize model predictions on validation set
print("Generating validation examples...")

# Load the entire saved model (much simpler!)
unet_model = torch.load('/home/dario/Desktop/FlameSentinels/best_prediction_model_full.pth')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
visualize_predictions(unet_model, val_loader, device, num_examples=5)

#--------------------------------------------------------------------------------------
# TESTING SECTION - Evaluate trained model on new test datasets
#--------------------------------------------------------------------------------------

def load_test_data(test_input_path, test_labels_path, selected_indices):
    """
    Load test data from a separate test folder
    
    Args:
        test_input_path (str): Path to test input data folder
        test_labels_path (str): Path to test labels folder  
        selected_indices (list): Indices of channels to select
        
    Returns:
        tuple: (test_data, test_labels) as torch tensors
    """
    print(f"Loading test data from {test_input_path}...")
    
    # Load test input files
    test_input_files = sorted(glob.glob(os.path.join(test_input_path, '*.npy')))
    test_label_files = sorted(glob.glob(os.path.join(test_labels_path, '*.npy')))
    
    print(f"Found {len(test_input_files)} test input files and {len(test_label_files)} test label files")
    
    if len(test_input_files) == 0:
        print("No test files found!")
        return None, None
    
    # Load test data
    x_test = np.array([np.load(f) for f in test_input_files])
    y_test = np.array([np.load(f) for f in test_label_files])
    
    print(f"Test input shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Extract selected channels
    x_test = x_test[:, :, :, selected_indices]
    print(f"Selected test input shape: {x_test.shape}")
    
    # Convert to tensors and permute dimensions
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    x_test = x_test.permute(0, 3, 1, 2)
    y_test = y_test.permute(0, 3, 1, 2)
    
    return x_test, y_test

def evaluate_model_on_test(model, test_data, test_labels, device, batch_size=16):
    """
    Evaluate model on test dataset
    
    Args:
        model: Trained model to evaluate
        test_data: Test input data
        test_labels: Test ground truth labels
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    model.to(device)
    
    # Create test dataset and dataloader
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.L1Loss()
    
    print("Evaluating model on test dataset...")
    
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Testing"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
            
            # Get predictions
            predictions = model(x_batch)
            
            # Calculate loss
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
            
            # Store predictions and labels
            predictions = predictions.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(y_batch)
    
    # Calculate metrics
    all_predictions_flat = np.array(all_predictions).flatten()
    all_labels_flat = np.array(all_labels).flatten()
    
    test_loss = total_loss / len(test_loader)
    test_mse = mean_squared_error(all_labels_flat, all_predictions_flat)
    test_mae = mean_absolute_error(all_labels_flat, all_predictions_flat)

    # Calculate additional metrics
    test_rmse = np.sqrt(test_mse)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(all_labels_flat, all_predictions_flat)[0, 1]
    
    results = {
        'test_loss': test_loss,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'correlation': correlation,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels)
    }
    
    return results

def visualize_test_results(model, test_data, test_labels, device, num_examples=5, save_path='/home/dario/Desktop/FlameSentinels/test_results.png'):
    """
    Visualize test results
    
    Args:
        model: Trained model to use for predictions
        test_data: Test input data
        test_labels: Test ground truth labels
        device: Device to run evaluation on
        num_examples: Number of examples to visualize
        save_path: Path to save the visualization
    """
    model.eval()
    model.to(device)
    
    # Create test dataset and dataloader
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    examples_shown = 0
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            if examples_shown >= num_examples:
                break
                
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
            predictions = model(x_batch)
            
            # Move to CPU and convert to numpy
            x_batch = x_batch.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            predictions = predictions.cpu().numpy()
            
            # Show examples from this batch
            batch_size = x_batch.shape[0]
            for i in range(min(batch_size, num_examples - examples_shown)):
                row = examples_shown
                
                # Show RGB channels (B02, B03, B04 - indices 0, 1, 2 in our selected channels)
                rgb_img = x_batch[i, :3, :, :].transpose(1, 2, 0)
                rgb_img = rgb_img[:, :, [2, 1, 0]]  # Convert to RGB order
                # Normalize RGB for display
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                axes[row, 0].imshow(rgb_img)
                axes[row, 0].set_title('RGB Input (Test)')
                axes[row, 0].axis('off')
                
                # Show NDVI
                ndvi_img = x_batch[i, 5, :, :]
                im1 = axes[row, 1].imshow(ndvi_img, cmap='RdYlGn')
                axes[row, 1].set_title('NDVI (Test)')
                axes[row, 1].axis('off')
                plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
                
                # Show ground truth
                gt_img = y_batch[i, 0, :, :]
                im2 = axes[row, 2].imshow(gt_img, cmap='Reds')
                axes[row, 2].set_title('Ground Truth (Test)')
                axes[row, 2].axis('off')
                plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)
                
                # Show prediction
                pred_img = predictions[i, 0, :, :]
                im3 = axes[row, 3].imshow(pred_img, cmap='Reds')
                axes[row, 3].set_title('Prediction (Test)')
                axes[row, 3].axis('off')
                plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)
                
                examples_shown += 1
                if examples_shown >= num_examples:
                    break
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Test visualization saved as '{save_path}'")

def test_model_on_dataset(model, test_dataset_name='test'):
    """
    Main function to test a trained model on a new dataset
    
    Args:
        model: Trained model to evaluate
        test_dataset_name (str): Name of the test dataset folder
    
    Returns:
        dict: Dictionary containing test results and metrics
    """
    print(f"\n{'='*60}")
    print(f"TESTING MODEL ON NEW DATASET: {test_dataset_name}")
    print(f"{'='*60}")
    
    # Define test data paths - multiple possible locations
    possible_paths = [
        # Option 1: Test data in main directory
        (f'/home/dario/Desktop/FlameSentinels/CHILE_INPUT',
         f'/home/dario/Desktop/FlameSentinels/CHILE_LABELS'),
        
        # Option 2: Test data in dataset-specific subfolder
        (f'/home/dario/Desktop/FlameSentinels/{test_dataset_name}_data/TEST_INPUT_DATA',
         f'/home/dario/Desktop/FlameSentinels/{test_dataset_name}_data/TEST_LABELS'),
        
        # Option 3: Test data in test subfolder
        (f'/home/dario/Desktop/FlameSentinels/test_data/{test_dataset_name}/TEST_INPUT_DATA',
         f'/home/dario/Desktop/FlameSentinels/test_data/{test_dataset_name}/TEST_LABELS'),
         
        # Option 4: Test data with sample_data prefix
        (f'/home/dario/Desktop/FlameSentinels/sample_data_{test_dataset_name}/TEST_INPUT_DATA',
         f'/home/dario/Desktop/FlameSentinels/sample_data_{test_dataset_name}/TEST_LABELS')
    ]
    
    test_input_path = None
    test_labels_path = None
    
    # Find the first existing path
    for input_path, labels_path in possible_paths:
        if os.path.exists(input_path) and os.path.exists(labels_path):
            test_input_path = input_path
            test_labels_path = labels_path
            break
    
    if test_input_path is None:
        print(f"Test data not found! Searched in the following locations:")
        for input_path, labels_path in possible_paths:
            print(f"  - Input: {input_path}")
            print(f"  - Labels: {labels_path}")
        print(f"\nPlease ensure test data exists in one of these locations.")
        return None
    
    print(f"Found test data in:")
    print(f"  Input: {test_input_path}")
    print(f"  Labels: {test_labels_path}")
    
    # Load test data
    x_test, y_test = load_test_data(test_input_path, test_labels_path, selected_indices)
    
    if x_test is None:
        print("Failed to load test data!")
        return None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Evaluate model
    test_results = evaluate_model_on_test(model, x_test, y_test, device)
    
    # Print results
    print(f"\n{'='*50}")
    print("🔥 FIRE PREDICTION MODEL - TEST RESULTS:")
    print(f"{'='*50}")
    print(f"📊 Test Loss (L1):     {test_results['test_loss']:.4f}")
    print(f"📊 Test MSE:           {test_results['test_mse']:.4f}")
    print(f"📊 Test MAE:           {test_results['test_mae']:.4f}")
    print(f"📊 Test RMSE:          {test_results['test_rmse']:.4f}")
    print(f"📊 Correlation:        {test_results['correlation']:.4f}")
    print(f"{'='*50}")
    
    # Interpret results
    if test_results['correlation'] > 0.7:
        print("🎯 Model shows STRONG correlation with ground truth!")
    elif test_results['correlation'] > 0.5:
        print("👍 Model shows GOOD correlation with ground truth!")
    elif test_results['correlation'] > 0.3:
        print("⚠️  Model shows MODERATE correlation with ground truth.")
    else:
        print("❌ Model shows WEAK correlation with ground truth.")
    
    # Visualize test results
    print("\n🖼️  Generating test visualizations...")
    save_path = f'/home/dario/Desktop/FlameSentinels/test_results_{test_dataset_name}.png'
    visualize_test_results(model, x_test, y_test, device, num_examples=5, save_path=save_path)
    
    return test_results

def load_trained_model(model_path='/home/dario/Desktop/FlameSentinels/best_prediction_model_full.pth'):
    """
    Helper function to load a trained model from file
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        model: Loaded PyTorch model, or None if loading fails
    """
    try:
        # Load UNet architecture from torch hub
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                              in_channels=7, out_channels=1, init_features=32, pretrained=False)
        
        # Load the saved state dict
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint.state_dict())
        
        print(f"✓ Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Example usage - uncomment and modify as needed:

# Option 1: Test with the currently trained model variable
test_results = test_model_on_dataset(trained_model, 'greece')

# Option 2: Test with a loaded model from file
# loaded_model = load_trained_model()
# if loaded_model:
#     test_results = test_model_on_dataset(loaded_model, 'turkey')
#     test_results = test_model_on_dataset(loaded_model, 'spain')

# Option 3: Test with the loaded model from validation section
# test_results = test_model_on_dataset(unet_model, 'greece')
