from validate import *
from architectures import *
from utils import *
from datasets import *
from submission_serialization import serialize
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from statistics import mean
import numpy as np

import config
import torch

def training_loop(network: torch.nn.Module,
                  train_data: torch.utils.data.Dataset,
                  eval_data: torch.utils.data.Dataset,
                  num_epochs: int,
                  batch_size: int,
                  learning_rate: float,
                  weight_decay: float, 
                  stop_progress_after: int,
                  show_progress: bool = False) -> tuple[list, list]:

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    train_data_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=stack_with_padding, num_workers=8)
    eval_data_loader = DataLoader(eval_data, shuffle=True, batch_size=batch_size, collate_fn=stack_with_padding, num_workers=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    network = network.to(device)

    training_losses = []
    evaluation_losses = []
    best_eval_loss = float('inf')
    no_improve_epochs = 0
    best_model_state_dict = None

    for epoch in range(num_epochs):
        network.train()

        epoch_training_losses = []
        epoch_eval_losses = []

        loop = tqdm(train_data_loader, disable=not show_progress)

        for stacked_pixelated_images, stacked_known_arrays, target_arrays, _ in loop:
            stacked_pixelated_images = stacked_pixelated_images.to(device)
            stacked_known_arrays = stacked_known_arrays.to(device)
            target_arrays = [target.to(device) for target in target_arrays]

            optimizer.zero_grad()

            outputs = network(stacked_pixelated_images)

            losses = []
            for i in range(len(outputs)):
                output = outputs[i]
                target = target_arrays[i]
                known_array = stacked_known_arrays[i]

                masked_output = output[~known_array.bool()]
                target_flatten = target.flatten()

                loss = criterion(masked_output, target_flatten)
                losses.append(loss)

            loss = sum(losses) / len(losses)
            loss.backward()  
            optimizer.step()  

            epoch_training_losses.append(loss.item())

            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=np.mean(epoch_training_losses))

        training_losses.append(mean(epoch_training_losses))

        print(f"Training loss at epoch {epoch+1}: {mean(epoch_training_losses)}")

        network.eval()
        with torch.no_grad():
            for stacked_pixelated_images, stacked_known_arrays, target_arrays, _ in eval_data_loader:
                stacked_pixelated_images = stacked_pixelated_images.to(device)
                stacked_known_arrays = stacked_known_arrays.to(device)
                target_arrays = [target.to(device) for target in target_arrays]

                outputs = network(stacked_pixelated_images)

                eval_losses = []
                for i in range(len(outputs)):
                    output = outputs[i]
                    target = target_arrays[i]
                    known_array = stacked_known_arrays[i]

                    masked_output = output[~known_array.bool()]
                    target_flatten = target.flatten()

                    loss = criterion(masked_output, target_flatten)
                    eval_losses.append(loss.item())

                epoch_eval_losses.append(mean(eval_losses))

        eval_loss = mean(epoch_eval_losses)
        evaluation_losses.append(eval_loss)

        print(f"Evaluation loss at epoch {epoch+1}: {eval_loss}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            no_improve_epochs = 0
            print(f"Best evaluation loss so far at epoch {epoch+1}: {best_eval_loss}")

            # Saving the best model
            torch.save(network.state_dict(), 'best_model.pth')
            best_model_state_dict = network.state_dict()
            print("Best model saved at epoch: ", epoch+1)

        else:
            no_improve_epochs += 1
            print(f"No improvement in evaluation loss at epoch {epoch+1}. Count: {no_improve_epochs}")

        if no_improve_epochs >= stop_progress_after:
            print(f"No improvement in evaluation loss for {stop_progress_after} consecutive epochs, stopping training at epoch {epoch+1}")
            break

    if best_model_state_dict is not None:
            network.load_state_dict(best_model_state_dict)

    return training_losses, evaluation_losses, network



def predict_original_values(model: torch.nn.Module, test_set_path: str) -> list:

    test_set = ImageDepixelationTestDataset(test_set_path)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )


    # Use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # To store predictions
    predictions = []

    # Evaluate the model
    with torch.no_grad():
        for stacked_pixelated_images, stacked_known_arrays, in test_loader:
            stacked_pixelated_images = stacked_pixelated_images.to(device)
            stacked_known_arrays = stacked_known_arrays.to(device)

            outputs = model(stacked_pixelated_images)

            for i in range(len(outputs)):
                output = outputs[i]
                known_array = stacked_known_arrays[i]

                # Get the output where the known array is False, i.e., the pixelated images
                predicted_output = output[~known_array.bool()]

                predicted_output = predicted_output * 255

                predicted_output = predicted_output.detach().cpu().numpy().astype(np.uint8)

                # Append flattened array to the list
                predictions.append(predicted_output.flatten())

    return predictions

    

def main():
    np.random.seed(0)
    torch.manual_seed(0)
    # validation stage:
    if not config.SKIP_VALIDATION:
        print("Start validating images")
        count, total = validate_images(config.IMAGES, config.LOG_FILE)
        if count / total == 1:
            print("All images seem to be fine")
        elif count / total < 1 and count / total >= 0.999:
            print("WARNING: There are faulty images, check the logs for more information")
        else:
            raise ValueError("There are more than 0.01%% faulty images, please check the logs")

    train_data, eval_data = create_datasets(
        image_path=config.IMAGES, 
        train_ratio=5/6, 
        width_range=(4,32), 
        height_range=(4,32), 
        size_range=(4,16), 
        crop_size=64, 
        num_crops=2
    )


    model = ImageDepixelationModel([    
        {'in_channels': 2, 'out_channels': 32, 'kernel_size': 3, 'activation': nn.LeakyReLU, 'batchnorm': False},
        {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'activation': nn.LeakyReLU, 'batchnorm': False},
        {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'activation': nn.LeakyReLU, 'batchnorm': False},
        {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'activation': nn.LeakyReLU, 'batchnorm': False},
        {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'activation': nn.LeakyReLU, 'batchnorm': False},
        {'in_channels': 32, 'out_channels': 1, 'kernel_size': 3, 'activation': nn.LeakyReLU, 'batchnorm': False},
    ])
    #model = ImageDepixelationModel([    
        #{'in_channels': 2, 'out_channels': 8, 'kernel_size': 3, 'activation': nn.ReLU, 'batchnorm': False},
        #{'in_channels': 8, 'out_channels': 8, 'kernel_size': 3, 'activation': nn.ReLU, 'batchnorm': False},
        #{'in_channels': 8, 'out_channels': 1, 'kernel_size': 3, 'activation': nn.Identity, 'batchnorm': False},
    #])


    # Train the model
    train_losses, eval_losses, model = training_loop(model, train_data, eval_data, num_epochs=10, show_progress=True, learning_rate=0.001, batch_size=64, stop_progress_after=2, weight_decay=0.00001)

    # save the network/model
    save(model,config.MODELS_DIR)

    # Plot the losses
    plot_losses(train_losses, eval_losses)

    predictions = predict_original_values(model, config.TEST_SET)
    serialize(predictions, 'predictions.bin')


    

        
    


    



if __name__ == "__main__":
    main()
