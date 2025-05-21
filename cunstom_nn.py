import torch
import torch.nn.functional as F

# Input (batch size 1, input size 3)
input = torch.tensor([[1.0, 2.0, 3.0]])  # Shape: (1, 3)

# Repeat TX 3 times to create input shape (3, 3)
input = input.repeat(3, 1)  # Shape: (3, 3)

# Target output
y_true = torch.tensor([[1.0]])  # Shape: (1, 1)

# Initialize weights (3x3) with requires_grad=True for autograd
w1 = torch.full((3, 3), 1.0, requires_grad=True)
w2 = torch.full((3, 3), 1.0, requires_grad=True)
w3 = torch.full((3, 3), 1.0, requires_grad=True)
w4 = torch.full((3, 3), 1.0, requires_grad=True)

# Bias vectors for activations (shape (3,))
ba1 = torch.full((3,), 1.0, requires_grad=True)
ba2 = torch.full((3,), 1.0, requires_grad=True)
ba3 = torch.full((3,), 1.0, requires_grad=True)
ba4 = torch.full((3,), 1.0, requires_grad=True)

# Training parameters
lr = 0.001
epochs = 3000

for epoch in range(epochs):
    # Forward pass
    x = input
    z1 = torch.sign(x) * (torch.abs(x) ** w1)    # Shape: (3, 3)
    a1 = z1.sum(dim=1) + ba1                      # Shape: (3,)
    x = a1.unsqueeze(1).repeat(1, 3)   # Shape: (3, 3)

    z2 = torch.sign(x) * (torch.abs(x) ** w2)
    a2 = z2.sum(dim=1) + ba2
    x = a2.unsqueeze(1).repeat(1, 3)

    z3 = torch.sign(x) * (torch.abs(x) ** w3)
    a3 = z3.sum(dim=1) + ba3
    x = a3.unsqueeze(1).repeat(1, 3)

    z4 = torch.sign(x) * (torch.abs(x) ** w4)
    a4 = z4.sum(dim=1) + ba4                      # Shape: (3,)
    x = a4
    # Final prediction: sum elements of a4 to get scalar output
    y_pred = x.sum().reshape(1, 1)

    # Loss calculation
    loss = F.mse_loss(y_pred, y_true)

    # Backward pass
    loss.backward()

    # Update weights and biases
    with torch.no_grad():
        for param in [w1, w2, w3, w4, ba1, ba2, ba3, ba4]:
            param -= lr * param.grad
            param.grad.zero_()

    # Print loss occasionally
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

print("\nFinal prediction:", y_pred.detach().numpy())
