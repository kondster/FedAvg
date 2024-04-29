import os
import numpy as np
from flask import Flask, request, jsonify
from helper import load_data, create_model
from client import Client

app = Flask(__name__)

class Server:
    def __init__(self, clients, test_data):
        self.clients = clients
        self.test_data = test_data
        self.global_model = create_model()
        self.rounds_loss = []
        self.rounds_accuracy = []

    def federated_averaging(self, epochs, batch_size, num_rounds, fraction_fit):
        for round in range(num_rounds):
            # Select a fraction of clients
            num_clients = max(int(fraction_fit * len(self.clients)), 1)
            selected_clients = np.random.choice(self.clients, num_clients, replace=False)

            # Broadcast global model weights to selected clients
            global_weights = self.global_model.get_weights()

            # Perform local training on selected clients
            client_weights = []
            for client in selected_clients:
                client_weights.append(client.train(global_weights, epochs, batch_size))

            # Perform weighted averaging of client models
            new_global_weights = [np.zeros_like(w) for w in global_weights]
            total_data_size = sum([len(client.data[0]) for client in selected_clients])
            for c in range(len(selected_clients)):
                for i in range(len(new_global_weights)):
                    new_global_weights[i] += client_weights[c][i] * len(selected_clients[c].data[0]) / total_data_size

            # Update global model weights
            self.global_model.set_weights(new_global_weights)

            # Evaluate global model
            loss, accuracy = self.global_model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
            self.rounds_loss.append(loss)
            self.rounds_accuracy.append(accuracy)
            print(f"Round {round+1}/{num_rounds} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        print("Federated Learning completed.")
        print("Final Global Model Results:")
        print(f"Loss: {self.rounds_loss[-1]:.4f}, Accuracy: {self.rounds_accuracy[-1]:.4f}")
        
        # Save the global model's performance plots
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.rounds_loss, label='Testing Loss')
        plt.plot(self.rounds_accuracy, label='Testing Accuracy')
        plt.title('Global Model Performance')
        plt.xlabel('Round')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), 'global_model_performance.png'))
        plt.close()

@app.route('/')
def home():
    return 'Welcome to the Federated Learning Server!'

@app.route('/global_model', methods=['GET'])
def get_global_model():
    global_weights = server.global_model.get_weights()
    return jsonify({'weights': [w.tolist() for w in global_weights]})

@app.route('/update_model', methods=['POST'])
def update_model():
    client_weights = request.json['weights']
    client_weights = [np.array(w) for w in client_weights]
    server.update_global_model(client_weights)
    return 'Model updated successfully'

if __name__ == "__main__":
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_data()

    # Create clients and distribute data
    num_clients = 100
    clients = []
    server_url = 'http://localhost:5000'  # Update with the appropriate server URL
    for i in range(num_clients):
        client_data = (x_train[i*500:(i+1)*500], y_train[i*500:(i+1)*500])
        clients.append(Client(i, client_data, server_url))

    # Create server
    server = Server(clients, (x_test, y_test))

    # Run FedAvg
    epochs_per_round = 5
    batch_size = 50
    num_rounds = 10  # Reduced the number of rounds for faster execution
    fraction_fit = 0.1
    server.federated_averaging(epochs_per_round, batch_size, num_rounds, fraction_fit)

    # Run the Flask server
    app.run(debug=True)