import torch
import lltm_cpp  # This is your compiled C++ extension

# Input sizes
batch_size = 4
input_features = 5
hidden_features = 10

# Dummy inputs
input = torch.randn(batch_size, input_features, dtype=torch.float32)
old_h = torch.randn(batch_size, hidden_features, dtype=torch.float32)
old_cell = torch.randn(batch_size, hidden_features, dtype=torch.float32)

# Combined input (input + hidden)
X_combined = input_features + hidden_features

# Weights and bias
weights = torch.randn(3 * hidden_features, X_combined, dtype=torch.float32)
bias = torch.randn(3 * hidden_features, dtype=torch.float32)

# Call C++ forward function
new_h, new_cell, input_gate, output_gate, candidate_cell, X, gate_weights = lltm_cpp.lltm_forward(
    input, weights, bias, old_h, old_cell
)

# Display outputs
print("New hidden state:\n", new_h)
print("New cell state:\n", new_cell)
