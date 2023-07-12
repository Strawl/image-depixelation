from validate import validate_images
import config
from utils import create_datasets
from train import training_loop, plot_losses
from network import ImageDepixelationModel
import torch.nn as nn

def main():
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

    train_data, eval_data = create_datasets(config.IMAGES, 4/5, (50,80), (50,80), (5,10))

    #model = ImageDepixelationModel([    
        #{'in_channels': 1, 'out_channels': 64, 'kernel_size': 9, 'activation': nn.ReLU, 'batchnorm': True},
        #{'in_channels': 64, 'out_channels': 32, 'kernel_size': 1, 'activation': nn.ReLU, 'batchnorm': True},
        #{'in_channels': 32, 'out_channels': 1, 'kernel_size': 5, 'activation': nn.Identity, 'batchnorm': True},
    #])
    model = ImageDepixelationModel([    
        {'in_channels': 1, 'out_channels': 8, 'kernel_size': 3, 'activation': nn.ReLU, 'batchnorm': False},
        {'in_channels': 8, 'out_channels': 1, 'kernel_size': 3, 'activation': nn.Identity, 'batchnorm': False},
    ])


    # Train the model
    train_losses, eval_losses = training_loop(model, train_data, eval_data, num_epochs=1, show_progress=True)

    # Plot the losses
    plot_losses(train_losses, eval_losses)




        
    


    



if __name__ == "__main__":
    main()
