# Importing libraries
import time
import os
import imghdr
from torch import nn
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets.folder import default_loader


class Extractor:

    def __init__(self, model_children, DS_layer_name='downsample'):
        self.model_children = model_children
        self.DS_layer_name = DS_layer_name

        self.CNN_layers = []
        self.Linear_layers = []
        self.DS_layers = []

        self.CNN_weights = []
        self.Linear_weights = []

        self.__no_sq_layers = 0  # number of sequential layers
        self.__no_containers = 0  # number of containers

        self.__verbose = []

        self.__bottleneck = models.resnet.Bottleneck
        self.__basicblock = models.resnet.BasicBlock

    def __Append(self, layer, Linear=False):
        """
        This function will append the layers weights and
        the layer itself to the appropriate variables

        params: layer: takes in CNN or Linear layer
        returns: None
        """

        if Linear:
            self.Linear_weights.append(layer.weight)
            self.Linear_layers.append(layer)

        else:
            self.CNN_weights.append(layer.weight)
            self.CNN_layers.append(layer)

    def __Layer_Extractor(self, layers):
        """
        This function(algorithm) finds CNN and linear layer in a Sequential layer

        params: layers: takes in either CNN or Sequential or linear layer
        return: None
        """

        for x in range(len(layers)):

            if type(layers[x]) == nn.Sequential:
                # Calling the fn to loop through the layer to get CNN layer
                self.__Layer_Extractor(layers[x])
                self.__no_sq_layers += 1

            if type(layers[x]) == nn.Conv2d:
                self.__Append(layers[x])

            if type(layers[x]) == nn.Linear:
                self.__Append(layers[x], True)

            # This statement makes sure to get the down-sampling layer in the model
            if self.DS_layer_name in layers[x]._modules.keys():
                self.DS_layers.append(layers[x]._modules[self.DS_layer_name])

            # The below statement will loop throgh the containers and append it
            if isinstance(layers[x], (self.__bottleneck, self.__basicblock)):
                self.__no_containers += 1
                for child in layers[x].children():
                    if type(child) == nn.Conv2d:
                        self.__Append(child)

    def __Verbose(self):

        for cnn_l, cnn_wts in zip(self.CNN_layers, self.CNN_weights):
            self.__verbose.append(f"CNN Layer : {cnn_l} ---> Weights shape :\
 {cnn_wts.shape}")

        for linear_l, linear_wts in zip(self.Linear_layers, self.Linear_weights):
            self.__verbose.append(f"Linear Layer : {linear_l}  --->\
 Weights shape : {linear_wts.shape}")

    def activate(self):
        """Activates the algorithm"""

        start = time.time()
        self.__Layer_Extractor(self.model_children)
        self.__Verbose()
        self.__ex_time = str(round(time.time() - start, 5)) + ' sec'

    def info(self):
        """Information"""

        return {
            'Down-sample layers name': self.DS_layer_name,
            'Total CNN Layers': len(self.CNN_layers),
            'Total Sequential Layers': self.__no_sq_layers,
            'Total Downsampling Layers': len(self.DS_layers),
            'Total Linear Layers': len(self.Linear_layers),
            'Total number of Bottleneck and Basicblock': self.__no_containers,
            'Total Execution time': self.__ex_time
        }

    def __repr__(self):
        return '\n'.join(self.__verbose)

    def __str__(self):
        return '\n'.join(self.__verbose)


class FeatureMap:

    def __init__(self, model, layer_nums, input_dir, output_dir, transform=None):
        self.extractor = Extractor(list(model.children()))
        self.layer_nums = layer_nums
        self.input_dir = input_dir
        self.filter_dir = os.path.join(output_dir, 'filter')
        self.feature_map_dir = os.path.join(output_dir, 'feature_map')
        self.loader = default_loader
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
        else:
            self.transform = transform

        self.extractor.activate()

        os.makedirs(self.filter_dir, exist_ok=True)
        os.makedirs(self.feature_map_dir, exist_ok=True)

    def plot_filters(self):
        for layer in self.layer_nums:
            plt.figure(figsize=(35, 35))
            for index, filter in enumerate(self.extractor.CNN_weights[layer]):
                plt.subplot(8, 8, index + 1)
                plt.imshow(filter[0, :, :].detach().cpu(), cmap='gray')
                plt.axis('off')
            img_name = os.path.join(self.filter_dir, 'layer_%d.png' % layer)
            plt.savefig(img_name)
            plt.close()

            layer_path = os.path.join(self.filter_dir, 'layer_%d' % layer)
            os.makedirs(layer_path, exist_ok=True)
            for index, filter in enumerate(self.extractor.CNN_weights[layer]):
                plt.figure(figsize=(12, 12))
                plt.imshow(filter[0, :, :].detach().cpu(), cmap='gray')
                plt.axis('off')
                img_name = os.path.join(layer_path, 'layer_%d_%d' % (layer, index))
                plt.savefig(img_name)
                plt.close()

    def plot_feature_maps(self):
        img_type_list = ['jpg', 'bmp', 'png', 'jpeg', 'jfif']
        for file in os.listdir(self.input_dir):
            img_path = os.path.join(self.input_dir, file)
            if os.path.isfile(img_path) and (imghdr.what(img_path) in img_type_list):
                img = self.loader(img_path)
                img = self.transform(img).unsqueeze(0).cuda()
                feature_maps = [self.extractor.CNN_layers[0](img)]
                for index, layer in enumerate(self.extractor.CNN_layers):
                    if index == 0:
                        continue
                    feature_maps.append(layer(feature_maps[-1]))

                for layer_num in self.layer_nums:
                    plt.figure(figsize=(30, 30))
                    layers = feature_maps[layer_num][0, :, :, :].detach().cpu()
                    output_dir = os.path.join(self.feature_map_dir, 'layer_%d' % layer_num)
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, file)
                    for i, feature_map in enumerate(layers):
                        if i == 64:
                            break
                        plt.subplot(8, 8, i + 1)
                        plt.imshow(feature_map, cmap='gray')
                        plt.axis('off')
                    plt.savefig(output_path)
                    plt.close()

                    layer_path = os.path.join(output_dir, file.split('.')[0])
                    os.makedirs(layer_path, exist_ok=True)
                    for i, feature_map in enumerate(layers):
                        plt.figure(figsize=(12, 12))
                        if i == 64:
                            break
                        plt.imshow(feature_map, cmap='gray')
                        plt.axis('off')
                        img_name = os.path.join(layer_path, str(i))
                        plt.savefig(img_name)
                        plt.close()


