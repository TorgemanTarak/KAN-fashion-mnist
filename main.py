import argparse

import numpy as np
from torchinfo import summary
import torch.nn as nn
import torch
from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT,KAN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes, create_validation_set,compute_mean,compute_std
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """



    xtrain, xtest, ytrain = load_data(args.data)
    data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(25),  # Randomly rotate images
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),  # Randomly shift images
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # Randomly shift images
    transforms.ToTensor()
])
    """
    # Lists to collect transformed samples and labels
    x_train_transformed = []
    y_train_transformed = []


    #print(ytrain[0],ytrain[60000])
 
    ## 2. Then we must prepare it. This is were you can create a validation set,
    # Make a validation set
    if not args.test:
        xtrain,ytrain,xtest,ytest = create_validation_set(xtrain,ytrain,0.2)

    ### WRITE YOUR CODE HERE to do any other data processing
    # Apply transformations to each sample in xtrain
    for sample, label in zip(xtrain, ytrain):
    # Reshape to 2D (28, 28)
     sample_2d = sample.reshape(28, 28)
    # Transform the sample
     transformed_sample = data_transforms(sample_2d)
    # Convert to numpy array and flatten back to 1D
     x_train_transformed.append(transformed_sample.numpy().flatten())
    # Append the corresponding label
     y_train_transformed.append(label)

# Convert lists of transformed samples and labels back to NumPy arrays
    x_train_transformed = np.array(x_train_transformed)

# Concatenate original and transformed data
    x_train_combined = np.concatenate((xtrain, x_train_transformed), axis=0)
    y_train_combined = np.concatenate((ytrain, ytrain), axis=0)

# Print the shape of the combined dataset
    print("Shape of x_train_combined:", x_train_combined.shape)
    print("Shape of y_train_combined:", y_train_combined.shape)
    xtrain=x_train_combined
    ytrain=y_train_combined
    """
    mean_xtrain = compute_mean(xtrain)
    std_xtrain = compute_std(xtrain)
    xtrain = normalize_fn(xtrain,mean_xtrain,std_xtrain)
    xtest = normalize_fn(xtest,mean_xtrain,std_xtrain)

    xtrain,ytrain,xtest,ytest = create_validation_set(xtrain,ytrain,0.2)
    


    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data


    if torch.cuda.is_available():
      device = "cuda:0"
    else :
      device = "cpu"
    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    if args.nn_type == "mlp":
      print("using MLP")
      model = MLP(input_size=784, n_classes=get_n_classes(ytrain), hidden_dimensions=[256, 128,64], activations=[nn.ReLU, nn.ReLU, nn.ReLU])
    
    if args.nn_type == "cnn":
        xtrain = xtrain.reshape(-1,1,28,28)
        xtest = xtest.reshape(-1,1,28,28)
        model = CNN(input_channels=1, n_classes=get_n_classes(ytrain), conv_channels=[64, 64], kernel_size=4, pool_size=2, fc_units=[256, 64])

    if args.nn_type == "transformer":
        xtrain = xtrain.reshape(-1, 1, 28, 28)
        xtest = xtest.reshape(-1, 1, 28, 28)
        chw = (1, 28, 28)
    
    if args.nn_type == "kan":
        xtest = xtest.reshape(-1,28*28)
        model = KAN([28 * 28, 64, 10])     
    
    # Ajustements des paramètres
    n_patches = 7  # Essayez différentes valeurs comme 4, 7, ou 14
    n_blocks = 6  # Augmenter le nombre de blocs
    hidden_d = 200  # Augmenter la dimension cachée
    n_heads = 10  # Assurez-vous que hidden_d est divisible par n_heads
    out_d = 10
    
  
    """
    model = MyViT(
        chw=chw,
        n_patches=n_patches,
        n_blocks=n_blocks,
        hidden_d=hidden_d,
        n_heads=n_heads,
        out_d=out_d,
     
    )
    """


    model.to(device)
    print("model running on device :",next(model.parameters()).device)
    summary(model)

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size,device = device)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
  
    #print(ytrain.shape)
    preds_train = method_obj.fit(xtrain, ytrain,device)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
   

    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)