from turtle import forward
from typing import Any, NamedTuple

from numpy import append
from functions import *
from mnist_loader import *
from layer import Layer

class Model_relu_softmax():
    def __init__(self, neurons_in_layers:list[int]) -> None:
        self.all_layers:list[Layer | Activation_function] = []
        self.loss = Loss_CategoricalCrossentropy()
        if(len(neurons_in_layers)<2):
            return

        #tworzymy warstwy
        for (inputs, neurons) in zip(neurons_in_layers[:-2], neurons_in_layers[1:-1]):
            self.all_layers.append(Layer(inputs, neurons))
            self.all_layers.append(Activation_ReLU())


        self.all_layers.append(Layer(neurons_in_layers[-2], neurons_in_layers[-1]))
        self.all_layers.append(Activation_Softmax())

        pass
    
    @classmethod
    def from_file(cls, path:str=""):
        layers:list[Layer | Activation_function] = []
        model = cls([])
        loaded_data:dict = np.load(path)
        for i in range(len(loaded_data)-1):
            
            if f'w{i}' in loaded_data and f'b{i}' in loaded_data:
                (w,b) = (loaded_data[f'w{i}'],loaded_data[f'b{i}'])
                layers.append(Layer.from_matrix(w,b))
            else:
                layers.append(Activation_ReLU())
        layers.append(Activation_Softmax())
        model.all_layers=layers
        return model

    

    def train(self,images, labels, batch_size:int=128, learning_rate:float=0.01, epochs:int = 20):
        
        def forward(X):
            for layer in self.all_layers:
                X = layer.forward(X)
            return X
        
        def backward(dValues):
            for layer in reversed(self.all_layers):
                dValues = layer.backward(dValues)


        for epoch in range(epochs):
        # Mieszamy dane na początku epoki (ważne!)
            permutation = np.random.permutation(len(images))
            images_shuffled = images[permutation]
            labels_shuffled = labels[permutation]
            total_loss = 0

            correct_predictions=0

            for i in range(0, len(images), batch_size):

                # Wycinamy kawałek danych
                batch_images = images_shuffled[i:i+batch_size]
                batch_labels = labels_shuffled[i:i+batch_size]
                

                # --- FORWARD ---
                output = forward(batch_images)
                total_loss += self.loss.forward(output, batch_labels)

                predictions = np.argmax(output, axis=1) #type:ignore
                targets = np.argmax(batch_labels, axis=1)
                correct_predictions += np.sum(predictions == targets)


                # --- BACKWARD ---
                dZ = self.loss.backward(output, batch_labels)
                backward(dZ)




                # --- UPDATE ---
                for layer in self.all_layers:
                    if isinstance(layer, Layer): # Sprawdzamy, czy to warstwa liniowa
                        layer.weights -= learning_rate * layer.dweights
                        layer.biases -= learning_rate * layer.dbiases

            
            # Po każdej epoce sprawdźmy wynik na całym zbiorze
            avg_loss = total_loss / (len(images) / batch_size)
            accuracy = (correct_predictions / len(images)) * 100
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
    
    def test(self, images, labels):

        def forward(X):
            for layer in self.all_layers:
                X = layer.forward(X)
            return X
        
        output = forward(images)
        predictions = np.argmax(output, axis=1) #type: ignore
        targets = np.argmax(labels, axis=1)
        correct_predictions = np.sum(predictions == targets)
        print("acc: "+ str(correct_predictions/len(images)))


    def export(self, path):
        data_to_save={}
        for i, layer in enumerate(self.all_layers):
            if isinstance(layer, Layer):
                data_to_save[f'w{i}']=layer.weights
                data_to_save[f'b{i}']=layer.biases

        np.savez(path, **data_to_save)


        



        
#'train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz'
if __name__ == "__main__":
    model = Model_relu_softmax([784,64,64,10])

    (images, labels) = normalise_data(10, 'train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz')
    print(len(images))
    model.train(images, labels, 25, 0.015, 20)
    print("Test")
    (images, labels) = normalise_data(10, 't10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz')
    model.test(images, labels)

    #model.export('/media/maciej/pliki/MSiD/ML/model/testowy_zapis')

    # model = Model_relu_softmax.from_file('/media/maciej/pliki/MSiD/ML/model/testowy_zapis.npz')
    # print("Test")
    # (images, labels) = normalise_data(10, 't10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz')
    # model.test(images, labels)