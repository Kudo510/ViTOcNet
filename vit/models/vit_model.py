import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST


# How does ViT Work? - forward function
# 1. patchify method:
#   a. Each input image is cut into sub-images equally sized. Ex: imagine we have 28x28 image. if n_patches = 7 then we have 7x7 patch where each patch is 4x4
#   b. Each sub-image goes through a linear embedding meaning that each sub-image turns into a one-dimensional vector
#   as a result imagine that we had N input images consisting of 1 channel, H=28, W=28 (N, 1, 28, 28) - they become -> (N, 49, 16)(#number of batches, due to number of patches generated, due to flattened patch) array
# 2. Linear Mapping (an extra hidden layer of matrix multiplication): Map the vector corresponding to each patch to the hidden size dimension.
# 3. Add classification token to each patch: (N, 49, 8) tokens tensor becomes an (N, 50, 8) tensor (we add the special token to each sequence)
# 4. get_positional_embeddings method:
#   a. Since the position of patches is important, A positional embedding is added to these vectors (tokens).
#      The positional embedding allows the network to know where each sub-image is positioned originally in the image.
# 5. These tokens are then passed (together with a special classification token) to the transformer encoders blocks 
# Each Transformer Block (Refer to "class ViTBlock") =
#   3.1. A Layer Normalization (LN)
#   3.2. followed by a Multi-head Self Attention (MSA) and a residual connection.
#   3.3. Then a second Layer Normalization, a Multi-Layer Perceptron (MLP), and again a residual connection. These blocks are connected back-to-back.
# 6. (!THIS LAYER CAN BE REMOVED FOR OUR PROJECT)
#    After consecutive transformer encoders
#    A classification Multi Layer Perceptron block is used for the final classification task only on the special classification token,
#    which by the end of this process has global information about the picture.
#For referance please refer to "https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c"
class ViT(nn.Module):
    #divides images into #n patches
    def patchify(self, images, n_patches):
        n, c, h, w = images.shape

        assert h == w, "Patchify method is implemented for square images only"

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches

    # Value to be added to the i-th tensor in its j-th coordinate
    def get_positional_embeddings(self, sequence_length, d):
      result = torch.ones(sequence_length, d)
      for i in range(sequence_length):
          for j in range(d):
              result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
      return result

    #n_patches = defines how many patches one the weight and the height will be divided into. n_patches=4 means the image will be divided into 4x4 patches
    #n_blocks = number of consecutive Transformer blocks that will be used.
    #hidden_d = value of hidden dimension is not certain. it can be anything. If we have a tensor of shape (N, 49, 16), it becomes (N, 49, 8) as the last vector replaced by an 8-dimensional vector that is the result of the matrix multiplication
    #n_heads = The number of heads is a hyperparameter and can be adjusted based on the specific task and the amount of data available.
    #          Increasing the number of heads allows the model to capture more complex patterns, but also increases the computational cost and the amount of data needed to train the model effectivel
    #out_d = important for classification task only.
    #mlp_ratio = The MLP is composed of two layers, where the hidden layer typically is four times as big (this is a parameter)
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10, mlp_ratio=4):
        # Super constructor
        super(ViT, self).__init__()
        
        # Attributes
        self.chw = chw # ( Channel , Height , Width )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        self.mlp_ratio = mlp_ratio
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 3) Positional embedding
        self.register_buffer('positional_embeddings', self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)
        
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads, mlp_ratio) for _ in range(n_blocks)])
        
        # 5) Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d * (n_patches**2 + 1) , out_d),
            #nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = self.patchify(images, self.n_patches).to(self.positional_embeddings.device)
        
        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)
        
        # Adding classification token to the tokens
        # (N, 49, 8) tokens tensor becomes an (N, 50, 8) tensor (we add the special token to each sequence)
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        
        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        
        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
            
        # Getting the classification token only
        #print(f'outshape: {out.shape}')
        #out = out[:, 0]
        
        return self.mlp(out.view(out.shape[0], -1)) # Map to output dimension, output category distribution

#Multihead Self Attention 
class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class ViTBlock(nn.Module):
  def __init__(self, hidden_d, n_heads, mlp_ratio=4):
      super(ViTBlock, self).__init__()
      self.hidden_d = hidden_d
      self.n_heads = n_heads

      self.norm1 = nn.LayerNorm(hidden_d)
      self.mhsa = MSA(hidden_d, n_heads)
      self.norm2 = nn.LayerNorm(hidden_d)
      self.mlp = nn.Sequential(
          nn.Linear(hidden_d, mlp_ratio * hidden_d),
          nn.GELU(),
          nn.Linear(mlp_ratio * hidden_d, hidden_d)
      )

  def forward(self, x):
      out = x + self.mhsa(self.norm1(x))
      out = out + self.mlp(self.norm2(out))
      return out
