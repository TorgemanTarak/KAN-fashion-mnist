import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """


    def __init__(self, input_size, n_classes,hidden_dimensions,activations):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__() ## wtf is this ???? do we need to addd it parameters 
        ##
        ###
        #### WRITE YOUR CODE HERE!
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_dimensions[0]))
        self.layers.append(activations[0]())
        
         # Hidden layers
        for i in range(len(hidden_dimensions) - 1):
            self.layers.append(nn.Linear(hidden_dimensions[i], hidden_dimensions[i+1]))
            self.layers.append(activations[i]())
    
            
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dimensions[-1], n_classes)) #no need to add relu i guess since the last output of the last layer will be provided to a softmax function ???
        
        ###
        ##

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        for layer in self.layers:
            x = layer(x) ## on le fait passer à travers les hidden layer mais je ne sait pas si je dois prendre en compte les poids quand j'utilise pytorch car dans le tp il prend en compte les pods 
        preds= x

        ###
        ##
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes, conv_channels, kernel_size, pool_size, fc_units):
        """
        Initialize the network.
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
            conv_channels (list): number of channels for each convolutional layer
            kernel_size (int): size of the convolutional kernel
            pool_size (int): size of the pooling kernel
            fc_units (list): number of units for each fully connected layer
        """
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(1,momentum = 0.90)
        self.conv1 = nn.Conv2d(1, 64, (4,4), padding="same")
        self.activation = torch.nn.ReLU()
        self.pool1 = nn.MaxPool2d((2,2))
        self.dropout1 = torch.nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(64, 64, (4, 4))
        # use relu at this stage too
        self.pool2 = nn.MaxPool2d((2,2))
        self.dropout2 = torch.nn.Dropout(0.5)
        self.linear1 = nn.Linear(1600,256)
        ## use relu 
        self.linear2 = nn.Linear(256,64)
        ## use rellu
        self.bn2 = torch.nn.BatchNorm1d(64,momentum = 0.99)
        self.linear3 = nn.Linear(64,10)
        self.softmax = torch.nn.Softmax()
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear3.weight, mode='fan_in', nonlinearity='relu')
        ###
        ##
        
        """self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.4)
        #MLP
        self.flatten = nn.Flatten()
        self.bn2 = nn.BatchNorm1d(128 * 3 * 3)  # Adjust based on your input dimensions after flattening
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.dropout3 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 10)"""

        """super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(1,momentum = 0.90)
        self.conv1 = nn.Conv2d(1, 64, (4,4), padding="same")
        self.activation = torch.nn.ReLU()
        self.pool1 = nn.MaxPool2d((2,2))
        self.dropout1 = torch.nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(64, 64, (4, 4))
        # use relu at this stage too
        self.pool2 = nn.MaxPool2d((2,2))
        self.dropout2 = torch.nn.Dropout(0.5)
        self.linear1 = nn.Linear(1600,256)
        ## use relu 
        self.linear2 = nn.Linear(256,64)
        ## use rellu
        self.bn2 = torch.nn.BatchNorm1d(64,momentum = 0.99)
        self.linear3 = nn.Linear(64,10)
        self.softmax = torch.nn.Softmax()
        """


        """
        # Calculate the size of the input to the first fully connected layer
        fc_input_size = conv_channels[1] * (28 // pool_size // pool_size) ** 2
        
        # Define the fully connected layers dynamically
        self.fc_layers = nn.ModuleList()
        in_features = fc_input_size
        for units in fc_units:
            self.fc_layers.append(nn.Linear(in_features, units))
            in_features = units
        
        # Final output layer
        self.fc_layers.append(nn.Linear(in_features, n_classes))
        """
    def forward(self, x):
        ##
        ###
        #### WRITE YOUR CODE HERE!
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape((x.shape[0], -1))  # Flatten the tensor
        
        for fc in self.fc_layers[:-1]:  # Apply all fc layers except the last one with ReLU
            x = F.relu(fc(x))
        
        preds = self.fc_layers[-1](x)  # Apply the last fc layer without activation
        ###
        ##
        """
        
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        ### MLP
        
        x = x.reshape(x.shape[0],-1)
        x = self.linear1(x)
        x = self.activation(x)
        
        x = self.linear2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x
    """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.bn2(x)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x"""


class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        super(MyViT, self).__init__()

        # Initialisation des attributs
        self.chw = chw  # Dimensions de l'entrée (C, H, W)
        self.n_patches = n_patches  # Nombre de patches par dimension
        self.n_blocks = n_blocks  # Nombre de blocs de l'encodeur Transformer
        self.n_heads = n_heads  # Nombre de têtes dans l'attention multi-têtes
        self.hidden_d = hidden_d  # Dimension cachée
        
        # Vérification que les dimensions sont divisibles par le nombre de patches
        assert chw[1] % n_patches == 0 and chw[2] % n_patches == 0, "Les dimensions d'entrée ne sont pas divisibles par le nombre de patches"
        self.patch_size = (chw[1] // n_patches, chw[2] // n_patches)

        # Calcul de la dimension d'entrée pour le mapping linéaire
        self.input_d = chw[0] * self.patch_size[0] * self.patch_size[1]
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)  # Mapper linéaire des patches vers la dimension cachée
        self.class_token = nn.Parameter(torch.zeros(1, self.hidden_d))  # Changement : Initialisé à zéro pour une initialisation plus déterministe
        self.positional_embeddings = nn.Parameter(self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d), requires_grad=False)  # Changement : Embeddings positionnels non appris
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])  # Liste des blocs de l'encodeur Transformer
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),  # Couche linéaire finale
            nn.Softmax(dim=-1)  # Softmax pour obtenir les probabilités de classe
        )

    def patchify(self, images, n_patches):
        # Changement : Utilisation de unfold pour créer des patches efficacement au lieu des boucles
        n, c, h, w = images.shape
        assert h == w, "La méthode patchify est implémentée uniquement pour les images carrées"
        patch_size = h // n_patches
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # Réorganise les dimensions pour obtenir la forme désirée
        patches = patches.contiguous().view(n, c, n_patches * n_patches, -1).permute(0, 2, 1, 3).contiguous()
        return patches.view(n, n_patches ** 2, -1)

    def get_positional_embeddings(self, sequence_length, d):
        # Changement : Génère les embeddings positionnels de manière vectorisée au lieu des boucles
        position = torch.arange(sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2) * -(np.log(10000.0) / d))
        pe = torch.zeros(sequence_length, d)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        """
        Predicts the class of a batch of samples with the model.

        Arguments:
            x (tensor): Input batch of shape (N, C, H, W)
        Returns:
            preds (tensor): Logits of predictions of shape (N, C)
"""
        n, c, h, w = x.shape
        patches = self.patchify(x, self.n_patches).to(self.positional_embeddings.device)  # Conversion des images en patches
        tokens = self.linear_mapper(patches)  # Mapping linéaire des patches vers la dimension cachée
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)  # Ajout du token de classification
        out = tokens + self.positional_embeddings  # Ajout des embeddings positionnels

        for block in self.blocks:
            out = block(out)  # Passage par chaque bloc de l'encodeur

        out = out[:, 0]  # Extraction du token de classification
        preds = self.mlp(out)  # Mapping vers la dimension de sortie et softmax
        return preds


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_d)  # Normalisation par couche avant l'attention
        self.mhsa = MyMSA(hidden_d, n_heads)  # Attention multi-têtes
        self.norm2 = nn.LayerNorm(hidden_d)  # Normalisation par couche avant le MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),  # Première couche du MLP
            nn.GELU(),  # Activation GELU
            nn.Linear(mlp_ratio * hidden_d, hidden_d)  # Deuxième couche du MLP
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))  # Résidu avec attention multi-têtes
        out = out + self.mlp(self.norm2(out))  # Résidu avec MLP
        return out


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads  # Dimension de chaque tête d'attention

        # Changement : Utilisation d'une seule couche linéaire pour générer Q, K, V en une seule opération
        self.qkv = nn.Linear(d, d * 3)
        self.softmax = nn.Softmax(dim=-1)  # Softmax pour l'attention

    def forward(self, x):
        batch_size, seq_len, d = x.shape
        # Changement : Génération de Q, K, V en une seule opération linéaire
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / np.sqrt(self.d_head)  # Calcul des scores d'attention
        attn = self.softmax(scores)  # Application du softmax
        context = attn @ v  # Application de l'attention aux valeurs
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # Recombinaison des têtes
        return context

class KAN(nn.Module):
    """
    Kolmogorov-Arnold Network which does classification.
    """
    def __init__(self, input_size, hidden_layer_size, inner_size, n_classes, activation=nn.ReLU):
        super(KAN, self).__init__()
        
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.inner_size = inner_size

        self.u_layers = nn.ModuleList()
        for i in range(input_size):
            for j in range(i + 1, input_size):
                self.u_layers.append(nn.Sequential(
                    nn.Linear(2, inner_size),
                    activation()
                ))

        self.v_layer = nn.Sequential(
            nn.Linear(input_size * (input_size - 1) // 2 * inner_size, hidden_layer_size),
            activation()
        )

        self.output_layer = nn.Linear(hidden_layer_size, n_classes)
    
    def forward(self, x):
        N, D = x.shape
        
        u_outputs = []
        index = 0
        for i in range(D):
            for j in range(i + 1, D):
                u_ij = self.u_layers[index](torch.cat((x[:, i:i+1], x[:, j:j+1]), dim=1))
                u_outputs.append(u_ij)
                index += 1
        
        u_concat = torch.cat(u_outputs, dim=1)
        v_output = self.v_layer(u_concat)
        preds = self.output_layer(v_output)
        
        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size,device):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader,ep)

            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        ##
        ###
        self.model.train()
        for i, (inputs, labels) in enumerate(dataloader):
            self.optimizer.zero_grad() #on commence par reset a 0 les grafient à 0 
            outputs = self.model(inputs) # on calcul l'output 
            #print(outputs.device)
            loss = self.criterion(outputs, labels) # on calcul la loss 
            loss.backward() # on backpropagate 
            self.optimizer.step() # on met à jour les poids 

            if i % 10 == 0:  # on print la loss tout 10 batches, on peut aussi la stocker dans un tableau puis l'afficher sur un graphe 
                print(f'Epoch [{ep+1}/{self.epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        ###
        ##

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        ##
        ###
        self.model.eval() # permet de passer au model evaluation c'est a dire plus de dropout ni de batch normalization 
        pred_labels = []
        with torch.no_grad(): # Désactive la calcul des gradients
            for inputs in dataloader:
                inputs = inputs[0]  # Extract the tensor from the tuple
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1) # Obtient les étiquettes prédites
                pred_labels.append(predicted) 
        return torch.cat(pred_labels)# Concatène les étiquettes prédites en un seul tenseur et les retourne
        ###
        ##
        
    
    def fit(self, training_data, training_labels,device):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float().to(device), 
                                      torch.from_numpy(training_labels).type(torch.LongTensor).to(device))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        x,y = next(iter(train_dataset))
        print("data and labels are on")
        print(x.device)
        print(y.device)

        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).to(self.device).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()