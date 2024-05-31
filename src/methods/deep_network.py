import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import math

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

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        if x.size(-1) != self.in_features:
            print(x.shape)
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x


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