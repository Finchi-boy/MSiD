import numpy as np
from abc import ABC, abstractmethod

class Activation_function(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, dvalues):
        pass


class Activation_ReLU(Activation_function):
    def forward(self, inputs):
        self.inputs = inputs

        # f(x) = max(0, x)
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, dvalues):
        # Tworzymy kopię błędu, żeby nie psuć oryginału
        self.dinputs = dvalues.copy()
        
        # Tam, gdzie wejście (z forward) było <= 0, zerujemy gradient
        self.dinputs[self.inputs <= 0] = 0
        
        return self.dinputs
    
class Activation_Softmax(Activation_function):
    def forward(self, inputs):
        self.inputs = inputs

        #e^(input-max(inputs)) <0 i <=1
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Dzielimy przez sumę wiersza
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    def backward(self, dvalues):
        # UWAGA: Jeśli używasz combo z Cross-Entropy, 
        # to dZ z Loss jest już gotowym gradientem i często 
        # tę metodę się pomija lub implementuje specyficznie.
        self.dinputs = dvalues 
        return self.dinputs
    



class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        # 1. Przycinamy wartości, żeby uniknąć log(0) -> -inf
        # np.clip(tablica, min, max)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 2. Obliczamy logarytmy tylko dla poprawnych klas
        # Jeśli y_true to One-Hot, mnożymy i sumujemy
        confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # 3. Wyliczamy ujemny logarytm naturalny
        negative_log_likelihoods = -np.log(confidences)
        
        # 4. Zwracamy średni błąd dla całego batcha (jedna liczba)
        return np.mean(negative_log_likelihoods)
    
    def backward(self, y_pred, y_true):
        return np.subtract(y_pred, y_true)/len(y_pred)