from layer import Layer
from mnist_loader import normalise_data
from functions import *

images, labels = normalise_data(10, 'train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz')
print(len(images))


# --- 3. Proces Forward Pass ---
# Przejście przez warstwy
learning_rate = 0.001  # Jak duże kroki robimy?

batch_size = 128

for epoch in range(50):
    # Mieszamy dane na początku epoki (ważne!)
    permutation = np.random.permutation(len(images))
    images_shuffled = images[permutation]
    labels_shuffled = labels[permutation]
    
    for i in range(0, batch_size*128, batch_size):
        # Wycinamy kawałek danych
        batch_images = images_shuffled[i:i+batch_size]
        batch_labels = labels_shuffled[i:i+batch_size]
        
        # --- FORWARD ---
        out1 = l1.forward(batch_images)
        act1 = r1.forward(out1)
        out2 = l2.forward(act1)
        act2 = r2.forward(out2)
        out3 = l3.forward(act2)
        probs = soft.forward(out3)
        
        # --- BACKWARD ---
        dZ = probs - batch_labels
        d1 = l3.backward(dZ)
        d2 = r2.backward(d1)
        d3 = l2.backward(d2)
        d4 = r1.backward(d3)
        l1.backward(d4)
        
        # --- UPDATE ---
        for layer in [l1, l2, l3]:
            layer.weights -= learning_rate * layer.dweights
            layer.biases -= learning_rate * layer.dbiases
            
    # Po każdej epoce sprawdźmy wynik na całym zbiorze
    full_out = soft.forward(l3.forward(r2.forward(l2.forward(r1.forward(l1.forward(images))))))
    loss = loss_func.forward(full_out, labels)
    acc = np.mean(np.argmax(full_out, axis=1) == np.argmax(labels, axis=1))
    print(f"Epoch {epoch}, Loss: {loss:.4f}, Acc: {acc*100:.2f}%")