# main.py
import numpy as np
import pandas as pd
from alpaca_api import api
from yahoo_stock_data import all_data, high_values_list, low_values_list, open_values_list, close_values_list, volume_values_list


# Implementation of LTSM class 
class LTSM:
    def __init__ (self, input_data, hidden_state):
        self.input_data = input_data
        self.hidden_state = hidden_state

        # Initialize 4 weights and biases for each of the 4 gates in the LTSM modle. (forget, input, output, cell)
        self.weight_forget = np.random.randn(hidden_state, input_data + hidden_state)
        self.weight_input = np.random.randn(hidden_state, input_data + hidden_state)
        self.weight_output = np.random.randn(hidden_state, input_data + hidden_state)
        self.weight_cell = np.random.randn(hidden_state, input_data + hidden_state)
        # Biases re-write
        self.bias_forget = np.zeros((hidden_state, 1))
        self.bias_input = np.zeros((hidden_state, 1))
        self.bias_output = np.zeros((hidden_state, 1))
        self.bias_cell = np.zeros((hidden_state, 1))

    def z_score_normalization(self, data):
        # Call data from Yahoo stock data
        mean = np.mean(data)
        std_dev = np.std(data)
        normalized_data = [(x - mean) / std_dev for x in data]
        print(normalized_data)
        return normalized_data
    
    def sigmoid(self, x):
        # Sigmoid function
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        # Tanh function
        return (np.tanh(x))   
    
    # Forget gate 
    def forget_gate(self, inputs, previous_hidden_state):
        # Concatenate previous hidden state and current inputs
        concatenate_input = previous_hidden_state + inputs
        # Create an array the size of the hidden 
        # Retrived: https://blog.finxter.com/how-to-create-a-python-list-of-size-n/#:~:text=How%20to%20initialize%20a%20list%20with%20n%20placeholder,%3D%205%20lst%20%3D%20%5BNone%5D%20%2A%20n%20print%28lst%29
        # Had to change none to 0 because you cant you cant add to a NoneType
        forget_gate_input = [0] * self.hidden_state
        #print(forget_gate_activation)
        for i in range(self.hidden_state):
            for j in range(len(concatenate_input)):
                forget_gate_input[i] += self.weight_forget[i][j] * concatenate_input[j]
            forget_gate_input[i] += self.bias_forget[i]
        
        # Apply sigmoid activation function manually
        forget_gate_activation = [self.sigmoid(x) for x in forget_gate_input]
        
        return forget_gate_activation
    
    # Input gate 
    def input_gate(self, inputs, previous_hidden_state):
        concatenate_input = previous_hidden_state + inputs

        # Compute the input to the output gate manually
        input_gate_input = [0] * self.hidden_state
        for i in range(self.hidden_state):
            for j in range(len(concatenate_input)):
                input_gate_input[i] += self.weight_input[i][j] * concatenate_input[j]
            input_gate_input[i] += self.bias_input[i]
        
        # Apply sigmoid activation function manually
        input_gate_activation = [self.sigmoid(x) for x in input_gate_input]
        
        return input_gate_activation
    

    def output_gate(self, inputs, previous_hidden_state):
        concatenate_input = previous_hidden_state + inputs

        # Compute the input to the output gate manually
        output_gate_input = [0] * self.hidden_state
        for i in range(self.hidden_state):
            for j in range(len(concatenate_input)):
                output_gate_input[i] += self.weight_output[i][j] * concatenate_input[j]
            output_gate_input[i] += self.bias_output[i]
        
        # Apply sigmoid activation function manually
        output_gate_activation = [self.sigmoid(x) for x in output_gate_input]
        
        return output_gate_activation
    

    def cell_gate(self, inputs, previous_hidden_state):
        concatenate_input = previous_hidden_state + inputs

        # Compute the input to the output gate manually
        cell_gate_input = [0] * self.hidden_state
        for i in range(self.hidden_state):
            for j in range(len(concatenate_input)):
                cell_gate_input[i] += self.weight_cell[i][j] * concatenate_input[j]
            # account for bias
            cell_gate_input[i] += self.bias_cell[i]
        
        # Apply sigmoid activation function manually
        cell_gate_activation = [self.tanh(x) for x in cell_gate_input]

        return cell_gate_activation
        
    def forward(self, inputs, prev_hidden_state, prev_cell_state):
    # Compute gate activations
        forget_val = np.array(self.forget_gate(inputs, prev_hidden_state))
        input_val = np.array(self.input_gate(inputs, prev_hidden_state))
        output_val = np.array(self.output_gate(inputs, prev_hidden_state))
        cell_val = np.array(self.cell_gate(inputs, prev_hidden_state))
    
    # Compute current cell state (C_t) and hidden state (h_t)
        cell_state = forget_val * prev_cell_state + input_val * cell_val
        hidden_state = output_val * self.tanh(cell_state)
    
        return hidden_state, cell_state
    

    def train(self, epochs, learning_rate=0.001, print_loss=False):
        # Initialize previous hidden state and cell state
        prev_hidden_state = np.zeros((self.hidden_state, 1))
        prev_cell_state = np.zeros((self.hidden_state, 1))

        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for i in range(len(all_normalized_data) - 1):
                # Prepare input and target data for this time step
                inputs = all_normalized_data[i]
                target = all_normalized_data[i + 1]

                # Forward pass
                hidden_state, cell_state = self.forward(inputs, prev_hidden_state, prev_cell_state)

                # Calculate loss (mean squared error)
                loss = np.mean((hidden_state - target) ** 2)
                total_loss += loss

                # Backpropagation through time (BPTT)
                # Compute gradients
                dL_dhidden = 2 * (hidden_state - target)
                dL_doutput = dL_dhidden * self.tanh(cell_state)
                dL_dcell = dL_dhidden * hidden_state * (1 - self.tanh(cell_state) ** 2)

                dL_dforget = dL_dcell * prev_cell_state
                dL_dinput = dL_dcell * cell_state
                dL_doutput_gate = dL_dhidden * self.tanh(cell_state)

                # Update weights and biases
                # Forget gate
                dL_dforget_reshaped = dL_dforget.reshape(self.hidden_state)  # Flatten dL_dforget
                prev_hidden_state_flattened = prev_hidden_state.flatten()  # Flatten prev_hidden_state
                forget_gate_input = np.concatenate([prev_hidden_state_flattened, inputs])
                self.weight_forget -= learning_rate * np.outer(dL_dforget_reshaped, forget_gate_input)
                self.bias_forget -= learning_rate * dL_dforget_reshaped.reshape(self.hidden_state, 1)

                # Input gate
                dL_dinput_reshaped = dL_dinput.reshape(self.hidden_state)  # Flatten dL_dinput
                input_gate_input = np.concatenate([prev_hidden_state_flattened, inputs])
                self.weight_input -= learning_rate * np.outer(dL_dinput_reshaped, input_gate_input)
                self.bias_input -= learning_rate * dL_dinput_reshaped.reshape(self.hidden_state, 1)

                # Output gate
                dL_doutput_gate_reshaped = dL_doutput_gate.reshape(self.hidden_state)  # Flatten dL_doutput_gate
                output_gate_input = np.concatenate([prev_hidden_state_flattened, inputs])
                self.weight_output -= learning_rate * np.outer(dL_doutput_gate_reshaped, output_gate_input)
                self.bias_output -= learning_rate * dL_doutput_gate_reshaped.reshape(self.hidden_state, 1)

                # Cell gate
                dL_dcell_reshaped = dL_dcell.reshape(self.hidden_state)  # Flatten dL_dcell
                cell_gate_input = np.concatenate([prev_hidden_state_flattened, inputs])
                self.weight_cell -= learning_rate * np.outer(dL_dcell_reshaped, cell_gate_input)
                self.bias_cell -= learning_rate * dL_dcell_reshaped.reshape(self.hidden_state, 1)

                # Update states for next iteration
                prev_hidden_state = hidden_state
                prev_cell_state = cell_state

                # Calculate accuracy (optional)
                predicted_next_state = self.forward(inputs, prev_hidden_state, prev_cell_state)[0]
                total_predictions += 1
                if np.array_equal(predicted_next_state, target):
                    correct_predictions += 1

            # Calculate average loss per epoch
            avg_loss = total_loss / len(all_normalized_data)
            if print_loss:
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

            # Calculate and print accuracy
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            print(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}")

# Create an instance of LSTM model
run = LTSM(5, 256)

prev_hidden_state = np.zeros((run.hidden_state, 1))
prev_cell_state = np.zeros((run.hidden_state, 1))

# Normalize and prepare data
normalized_high_values = run.z_score_normalization(high_values_list)
normalized_low_values = run.z_score_normalization(low_values_list)
normalized_open_values = run.z_score_normalization(open_values_list)
normalized_close_values = run.z_score_normalization(close_values_list)
normalized_volume_values = run.z_score_normalization(volume_values_list)

# Combine normalized data into a single array
all_normalized_data = np.column_stack((
    normalized_high_values,
    normalized_low_values,
    normalized_open_values,
    normalized_close_values,
    normalized_volume_values
))

run.train(epochs=10, learning_rate=0.001, print_loss=True)