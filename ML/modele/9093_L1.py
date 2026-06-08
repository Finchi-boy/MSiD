import numpy as np
from functions import (
    Loss_BinaryCrossentropy,
    Activation_ReLU,
    Activation_Sigmoid,
    Activation_function,
)

# from mnist_loader import *
from layer import Layer


class Model_relu_sigmoid:
    def __init__(self, neurons_in_layers: list[int]) -> None:
        self.all_layers: list[Layer | Activation_function] = []
        self.loss = Loss_BinaryCrossentropy()
        if len(neurons_in_layers) < 2:
            return

        # tworzymy warstwy
        for inputs, neurons in zip(neurons_in_layers[:-2], neurons_in_layers[1:-1]):
            self.all_layers.append(Layer(inputs, neurons))
            self.all_layers.append(Activation_ReLU())

        self.all_layers.append(Layer(neurons_in_layers[-2], neurons_in_layers[-1]))
        self.all_layers.append(Activation_Sigmoid())

        pass

    @classmethod
    def from_file(cls, path: str = ""):
        layers: list[Layer | Activation_function] = []
        model = cls([])
        loaded_data: dict = np.load(path)
        for i in range(len(loaded_data) - 1):
            if f"w{i}" in loaded_data and f"b{i}" in loaded_data:
                (w, b) = (loaded_data[f"w{i}"], loaded_data[f"b{i}"])
                layers.append(Layer.from_matrix(w, b))
            else:
                layers.append(Activation_ReLU())
        layers.append(Activation_Sigmoid())
        model.all_layers = layers
        return model

    def train(
        self,
        X,
        y,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        epochs: int = 20,
        l1_lambda: float = 0.001
    ):

        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        half = batch_size // 2

        def forward(X):
            for layer in self.all_layers:
                X = layer.forward(X)
            return X

        def backward(dValues):
            for layer in reversed(self.all_layers):
                dValues = layer.backward(dValues)

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            n_batches = len(idx_neg) // half
            np.random.shuffle(idx_neg)

            for i in range(n_batches):
                # Połowa batcha z chorych (z powtórzeniami bo ich mniej)
                batch_pos = idx_pos[np.random.randint(0, len(idx_pos), half)]
                # Połowa z zdrowych (sekwencyjnie, bez powtórzeń)
                batch_neg = idx_neg[i * half : (i + 1) * half]

                batch_idx = np.concatenate([batch_pos, batch_neg])
                np.random.shuffle(batch_idx)

                X_batch = X[batch_idx]
                y_batch = y[batch_idx].reshape(-1, 1)

                # print(f"Batch pos: {y_batch.sum()}, neg: {(y_batch == 0).sum()}")

                output = forward(X_batch)
                total_loss += self.loss.forward(output, y_batch)

                preds = (output >= 0.5).astype(int)  # type: ignore
                correct += np.sum(preds == y_batch)

                dZ = self.loss.backward(output, y_batch)
                backward(dZ)

                # Update weights and biases
                # for layer in self.all_layers:
                #     if isinstance(layer, Layer):
                #         layer.weights -= learning_rate * layer.dweights
                #         layer.biases -= learning_rate * layer.dbiases

                for layer in self.all_layers:
                    if isinstance(layer, Layer):
                        # Obliczamy karę L1: pochodna z wartości bezwzględnej to po prostu znak (sign)
                        l1_penalty = l1_lambda * np.sign(layer.weights)

                        # Aktualizujemy wagi dodając karę do gradientu
                        layer.weights -= learning_rate * (layer.dweights + l1_penalty)
                        layer.biases -= learning_rate * layer.dbiases

            avg_loss = total_loss / n_batches
            accuracy = correct / (n_batches * batch_size) * 100
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%"
            )

    def test(self, X, y, filepath="raport1.txt"):
        # 1. Przepływ w przód (Forward Pass) przez wszystkie warstwy
        current_X = X
        for layer in self.all_layers:
            current_X = layer.forward(current_X)

        # Surowe prawdopodobieństwa z przedziału (0, 1) przed zaokrągleniem.
        # Są niezbędne do prawidłowego obliczenia AUC ROC!
        probs = current_X.flatten()

        # Klasyfikacja binarna (0 lub 1) na podstawie progu 0.5
        preds = (probs >= 0.5).astype(int)

        # 2. Obliczanie elementów macierzy pomyłek (Confusion Matrix)
        tp = np.sum((preds == 1) & (y == 1))
        fp = np.sum((preds == 1) & (y == 0))
        fn = np.sum((preds == 0) & (y == 1))
        tn = np.sum((preds == 0) & (y == 0))

        # 3. Obliczanie podstawowych oraz nowych metryk
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        # Sensitivity (Czułość) to matematycznie dokładnie to samo co Recall
        sensitivity = recall

        # Specificity (Swoistość) - zdolność do poprawnego wykrywania zdrowych osób
        specificity = tn / (tn + fp + 1e-7)

        # AUC ROC - importujemy dedykowaną funkcję ze sklearn
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, probs)

        # 4. Formatowanie czytelnego raportu tekstowego
        report_text = (
            f"==================================================\n"
            f"          MEDYCZNY RAPORT KLASYFIKACJI CUKRZYCY   \n"
            f"==================================================\n"
            f"Accuracy (Dokładność ogólna): {np.mean(preds == y):.4f}\n"
            f"Precision (Precyzja):         {precision:.4f}\n"
            f"Sensitivity / Recall (Czułość):{sensitivity:.4f}\n"
            f"Specificity (Swoistość):      {specificity:.4f}\n"
            f"F1-Score:                     {f1:.4f}\n"
            f"AUC ROC:                      {auc:.4f}\n"
            f"--------------------------------------------------\n"
            f"Macierz Pomyłek (Confusion Matrix):\n"
            f"                  Zdiagnozowany:  Zdiagnozowany: \n"
            f"                  Zdrowy          Chory          \n"
            f"Faktycznie Zdrowy:  {tn:<15} {fp:<15}\n"
            f"Faktycznie Chory:   {fn:<15} {tp:<15}\n"
            f"==================================================\n"
        )

        # 5. Wypisanie raportu w konsoli
        print(report_text)

        # 6. Zapis raportu do wskazanego pliku tekstowego
        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(report_text)
                print(f"[INFO] Raport został pomyślnie zapisany do pliku: {filepath}\n")
            except Exception as e:
                print(f"[BŁĄD] Nie udało się zapisać pliku: {e}\n")

    # def test(self, X, y):
    #     for layer in self.all_layers:
    #         X = layer.forward(X)
    #     preds = (X >= 0.5).astype(int).flatten()  # type: ignore

    #     tp = np.sum((preds == 1) & (y == 1))
    #     fp = np.sum((preds == 1) & (y == 0))
    #     fn = np.sum((preds == 0) & (y == 1))

    #     precision = tp / (tp + fp + 1e-7)
    #     recall = tp / (tp + fn + 1e-7)
    #     f1 = 2 * precision * recall / (precision + recall + 1e-7)

    #     print(f"Accuracy:  {np.mean(preds == y):.4f}")
    #     print(f"Precision: {precision:.4f}")
    #     print(f"Recall:    {recall:.4f}")
    #     print(f"F1 Score:  {f1:.4f}")


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from model import Model_relu_sigmoid

    df = pd.read_csv("D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\NHANES data\\nhanes_mergedv2.csv")

    X = df.drop(columns=["DIQ010 - Diabetes status"]).values
    y = df["DIQ010 - Diabetes status"].values

    # Normalizacja — fit tylko na train, żeby nie było data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,  # type: ignore
        random_state=42,  # type: ignore
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)  # transform, nie fit_transform!

    # 12 featurów wejściowych, 1 wyjście
    model = Model_relu_sigmoid([12, 128, 256, 128, 32, 1])
    model.train(X_train, y_train, batch_size=256, learning_rate=0.001, epochs=200, l1_lambda=0.001)
    model.test(X_test, y_test, filepath="D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\new_model\\raport4.txt")
