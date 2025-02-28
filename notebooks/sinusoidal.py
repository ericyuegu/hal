# %%
import matplotlib.pyplot as plt
import torch


def sinusoidal_positional_encoding_1d(seq_len: int, d_model: int, device: torch.device | None = None) -> torch.Tensor:
    """
    :param d_model: dimension of the model
    :param seq_len: length of positions
    :return: seq_len*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with " "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(torch.tensor(10000.0).log() / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


# Generate positional encoding
d_model = 512  # Dimension of the model
seq_len = 256  # Length of positions
pe = sinusoidal_positional_encoding_1d(d_model, seq_len)

# Convert to numpy for plotting
pe_np = pe.numpy()

# Create a figure with two subplots
plt.figure(figsize=(15, 10))

# Plot 1: Heatmap of the positional encoding
plt.subplot(2, 1, 1)
plt.imshow(pe_np, cmap="viridis", aspect="auto")
plt.colorbar(label="Value")
plt.xlabel("Dimension")
plt.ylabel("Position")
plt.title("Sinusoidal Positional Encoding Heatmap")

# Plot 2: Line plot of each position vector
plt.subplot(2, 1, 2)
for i in range(min(20, length)):  # Plot first 10 positions for clarity
    plt.plot(pe_np[i], label=f"Position {i}")
plt.xlabel("Dimension")
plt.ylabel("Value")
plt.title("Sinusoidal Positional Encoding Vectors")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
pe[1]


# %%
def fourier_positional_encoding1d(dim, length):
    frequencies = 1024 * torch.linspace(0, -torch.tensor(10000.0).log(), dim // 2).exp()
    emb = normalized.view(-1, 1) * frequencies
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
