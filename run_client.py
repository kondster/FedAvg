import requests
from client import Client
from helper import load_data

def main():
    server_url = 'http://localhost:5000'  # Update with the appropriate server URL
    client_id = 0  # Update with the appropriate client ID

    # Load and preprocess data for the client
    x_train, y_train, _, _ = load_data()
    client_data = (x_train[client_id*500:(client_id+1)*500], y_train[client_id*500:(client_id+1)*500])

    # Create a client instance
    client = Client(client_id, client_data, server_url)

    # Simulate federated learning rounds
    num_rounds = 10
    epochs_per_round = 5
    batch_size = 50

    for round in range(num_rounds):
        print(f"Round {round+1}/{num_rounds}")

        # Get the global model weights from the server
        client.get_global_model()

        # Train the client's model locally
        client.train(epochs=epochs_per_round, batch_size=batch_size)

        # Send the updated model weights to the server
        client.send_model_update()

    print("Federated Learning completed.")

if __name__ == "__main__":
    main()