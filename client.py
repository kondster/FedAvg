import os
import requests
from helper import create_model

class Client:
    def __init__(self, client_id, data, server_url):
        self.client_id = client_id
        self.data = data
        self.model = create_model()
        self.server_url = server_url

    def get_global_model(self):
        response = requests.get(f'{self.server_url}/global_model')
        global_weights = response.json()['weights']
        global_weights = [np.array(w) for w in global_weights]
        self.model.set_weights(global_weights)

    def train(self, global_weights, epochs, batch_size):
        self.model.set_weights(global_weights)
        history = self.model.fit(self.data[0], self.data[1], epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Print the final training results for the client
        print(f"Client {self.client_id} - Final Training Results:")
        print(f"Loss: {history.history['loss'][-1]:.4f}, Accuracy: {history.history['accuracy'][-1]:.4f}")
        
        # Save the training loss and accuracy plots
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.title(f'Client {self.client_id} - Training Performance')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), f'client_{self.client_id}_performance.png'))
        plt.close()
        
        return self.model.get_weights()

    def send_model_update(self):
        client_weights = self.model.get_weights()
        requests.post(f'{self.server_url}/update_model', json={'weights': [w.tolist() for w in client_weights]})