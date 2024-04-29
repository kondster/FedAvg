Federated Averaging (FedAvg) on CIFAR-10
=======================================

This project demonstrates the implementation of the Federated Averaging (FedAvg) algorithm on the CIFAR-10 dataset using Python and TensorFlow. The code is split into client-side and server-side components, which communicate using a REST API.

Project Structure
-----------------
- `helper.py`: Contains helper functions for loading and preprocessing the CIFAR-10 dataset and creating the model architecture.
- `client.py`: Defines the `Client` class, which represents a client in the federated learning setup.
- `server.py`: Defines the `Server` class, which coordinates the federated learning process and contains the server-side logic.

Code Explanation
----------------
1. `helper.py`:
   - `load_data()`: Loads the CIFAR-10 dataset, preprocesses the data, and returns the training and testing sets.
   - `create_model()`: Creates the CNN model architecture using the Sequential API from Keras.

2. `client.py`:
   - `Client` class:
     - `__init__(self, client_id, data, server_url)`: Initializes the client with an ID, local dataset, and server URL.
     - `get_global_model(self)`: Retrieves the global model weights from the server.
     - `train(self, global_weights, epochs, batch_size)`: Performs local training on the client's dataset using the global model weights.
     - `send_model_update(self)`: Sends the updated model weights back to the server.

3. `server.py`:
   - `Server` class:
     - `__init__(self, clients, test_data)`: Initializes the server with a list of clients and the test dataset.
     - `federated_averaging(self, epochs, batch_size, num_rounds, fraction_fit)`: Coordinates the federated learning process by selecting clients, broadcasting the global model, aggregating client updates, and evaluating the model.
   - Flask routes:
     - `@app.route('/')`: Handles the root URL and returns a welcome message.
     - `@app.route('/global_model', methods=['GET'])`: Endpoint for clients to retrieve the global model weights.
     - `@app.route('/update_model', methods=['POST'])`: Endpoint for clients to send their updated model weights.
   - Main execution:
     - Loads and preprocesses the CIFAR-10 dataset.
     - Creates clients and distributes the data among them.
     - Creates the server instance with the clients and test dataset.
     - Runs the federated learning process using the `federated_averaging` method.
     - Starts the Flask server to handle client requests.

Setup and Execution
-------------------
1. Make sure you have Python 3.x, TensorFlow 2.x, Flask, and NumPy installed. You can install the required libraries using the following command:
   ```
   pip install tensorflow flask numpy
   ```

2. Save the `helper.py`, `client.py`, and `server.py` files in the same directory.

3. Open a terminal or command prompt and navigate to the directory containing the project files.

4. Run the server-side code by executing the following command:
   ```
   python server.py
   ```
   This will start the Flask server and initiate the federated learning process.

5. The server will run the federated learning process for the specified number of rounds, selecting a fraction of clients in each round, broadcasting the global model, aggregating client updates, and evaluating the model's performance on the test set.

6. The server will output the loss and accuracy of the global model after each round of communication.

7. Once the federated learning process is complete, the server will continue running and listening for client requests.

8. You can access the server's welcome message by opening a web browser and navigating to `http://localhost:5000`.

Note: The current implementation assumes a balanced distribution of data among the clients. If you have a different data distribution, you may need to modify the code accordingly.

Additional Notes
----------------
- The communication between the clients and the server is not secure in this basic setup. In a production environment, you should implement secure communication protocols and authentication mechanisms.
- The code provided is a simplified implementation for demonstration purposes. In a real-world scenario, you may need to handle additional aspects such as client failures, asynchronous communication, and more robust error handling.
- Feel free to experiment with different model architectures, hyperparameters, and dataset distributions to explore the behavior of the federated learning system.
```