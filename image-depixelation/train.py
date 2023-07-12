import torch
from statistics import mean
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import stack_with_padding
import numpy as np


def training_loop(network: nn.Module,
                  train_data: torch.utils.data.Dataset,
                  eval_data: torch.utils.data.Dataset,
                  num_epochs: int,
                  show_progress: bool = False) -> tuple[list, list]:

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_data_loader = DataLoader(train_data, shuffle=False, batch_size=32, collate_fn=stack_with_padding)
    eval_data_loader = DataLoader(eval_data, shuffle=False, batch_size=32, collate_fn=stack_with_padding)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_losses = []
    evaluation_losses = []
    best_eval_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        network.train()

        epoch_training_losses = []
        epoch_eval_losses = []

        loop = tqdm(train_data_loader, disable=not show_progress)

        for stacked_pixelated_images, stacked_known_arrays, target_arrays, _ in loop:
            stacked_pixelated_images = stacked_pixelated_images.to(device)
            stacked_known_arrays = stacked_known_arrays.to(device)
            target_arrays = [target.to(device) for target in target_arrays]
            #target_arrays = [print(target for target in target_arrays]


            optimizer.zero_grad()

            outputs = network(stacked_pixelated_images)

            losses = []
            for i in range(len(outputs)):
                output = outputs[i]
                target = target_arrays[i]
                known_array = stacked_known_arrays[i]

                masked_output = output[known_array.bool()]
                target_flatten = target.flatten()

                loss = criterion(masked_output, target_flatten)
                losses.append(loss)


            loss = sum(losses) / len(losses)
            loss.backward()  # Don't forget to call backward on your final loss
            optimizer.step()  # Update weights based on the gradient


            epoch_training_losses.append(loss.item())

            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=np.mean(epoch_training_losses))

        training_losses.append(mean(epoch_training_losses))

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

                    masked_output = output[known_array.bool()]
                    target_flatten = target.flatten()

                    loss = criterion(masked_output, target_flatten)
                    eval_losses.append(loss.item())

                epoch_eval_losses.append(mean(eval_losses))

        eval_loss = mean(epoch_eval_losses)
        evaluation_losses.append(eval_loss)


        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= 3:
            print(f"We have no improvement anymore, stopping the training")
            break

    return training_losses, evaluation_losses


def plot_losses(train_losses: list, eval_losses: list):
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(eval_losses, label='Evaluation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error Loss')
    plt.legend()
    plt.title('Training and Evaluation Losses')
    plt.show()

