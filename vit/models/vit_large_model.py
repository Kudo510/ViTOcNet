import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import numpy as np
import math
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
from transformers import ViTFeatureExtractor, ViTForImageClassification

class ViTModel(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10, mlp_ratio=4, batch_size = 128):
        super(ViTModel, self).__init__()

        self.chw = chw  # (Channel, Height, Width)
        self.n_patches = n_patches
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.out_d = out_d
        self.mlp_ratio = mlp_ratio
        self.n_blocks = n_blocks

        #self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224')
        #self.pretrained_model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')

        self.embeddings = ViTEmbeddings(chw, n_patches, hidden_d)
        self.encoder = ViTEncoder(hidden_d, n_heads, mlp_ratio, n_blocks)
        self.layernorm = nn.LayerNorm((hidden_d,), eps=1e-12, elementwise_affine=True)
        # 1000: the output tensor length of pretrained model
        self.classifier = nn.Linear(hidden_d* (n_patches**2), 1000, bias=True)
        self.hidden_bridge = nn.Linear(1000 , out_d, bias=True)

    def forward(self, images):
        """
        embedding_output = self.embeddings(images)
        encoded_output = self.encoder(embedding_output)
        layernorm_output = self.layernorm(encoded_output)
        logits_hidden = self.classifier(layernorm_output[:, 0])
        """
        
        """ Pretrained
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs = self.feature_extractor(images=images, return_tensors="pt").to(device) 
        outputs = self.pretrained_model(**inputs)        
        logits_hidden = outputs.logits
        logits = self.hidden_bridge(logits_hidden)
        return logits
        """
        embedding_output = self.embeddings(images)
        encoder_output = self.encoder(embedding_output)
        layernorm_output = self.layernorm(encoder_output)
        logits_hidden = self.classifier(layernorm_output.view(layernorm_output.shape[0], -1))
        logits = self.hidden_bridge(logits_hidden)
        return logits


class ViTEmbeddings(nn.Module):
    def __init__(self, chw, n_patches, hidden_d):
        super(ViTEmbeddings, self).__init__()

        self.patch_embeddings = ViTPatchEmbeddings(chw, n_patches, hidden_d)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, images):
        x = self.patch_embeddings(images)
        return self.dropout(x)

class ViTPatchEmbeddings(nn.Module):
    def __init__(self, chw, n_patches, hidden_d):
        super(ViTPatchEmbeddings, self).__init__()
        patch_size = chw[1] // n_patches
        self.projection = nn.Conv2d(chw[0], hidden_d, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))

    def forward(self, images):
        patches = self.projection(images)
        return patches.flatten(2).transpose(1, 2)


class ViTEncoder(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio, n_blocks):
        super(ViTEncoder, self).__init__()

        self.layer = nn.ModuleList([ViTLayer(hidden_d, n_heads, mlp_ratio) for _ in range(n_blocks)])

    def forward(self, hidden_states):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)

        return hidden_states

class ViTLayer(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio):
        super(ViTLayer, self).__init__()
        self.layernorm_before = nn.LayerNorm((hidden_d,), eps=1e-12, elementwise_affine=True)
        self.attention = ViTAttention(hidden_d, n_heads)
        self.intermediate = ViTIntermediate(hidden_d, mlp_ratio)
        self.output = ViTOutput(hidden_d, mlp_ratio)
        self.layernorm_after = nn.LayerNorm((hidden_d,), eps=1e-12, elementwise_affine=True)
    
    def forward(self, hidden_states):
        attention_output = self.attention(self.layernorm_before(hidden_states))
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        layernorm_output = self.layernorm_after(layer_output)
        return layernorm_output


class ViTAttention(nn.Module):
    def __init__(self, hidden_d, n_heads):
        super(ViTAttention, self).__init__()

        self.attention = ViTSelfAttention(hidden_d, n_heads)
        self.output = ViTSelfOutput(hidden_d)

    def forward(self, input_tensor):
        self_output = self.attention(input_tensor)
        attention_output = self.output(self_output, input_tensor)

        return attention_output


class ViTSelfAttention(nn.Module):
    def __init__(self, hidden_d, n_heads):
        super(ViTSelfAttention, self).__init__()

        self.query = nn.Linear(hidden_d, hidden_d)
        self.key = nn.Linear(hidden_d, hidden_d)
        self.value = nn.Linear(hidden_d, hidden_d)
        self.dropout = nn.Dropout(p=0.1)
        self.n_heads = n_heads

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, x.size(-1) // self.n_heads)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.n_heads)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.n_heads * (hidden_states.size(-1) // self.n_heads),)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class ViTSelfOutput(nn.Module):
    def __init__(self, hidden_d):
        super(ViTSelfOutput, self).__init__()

        self.dense = nn.Linear(hidden_d, hidden_d)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTIntermediate(nn.Module):
    def __init__(self, hidden_d, mlp_ratio):
        super(ViTIntermediate, self).__init__()

        self.dense = nn.Linear(hidden_d, hidden_d * mlp_ratio)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ViTOutput(nn.Module):
    def __init__(self, hidden_d, mlp_ratio):
        super(ViTOutput, self).__init__()

        self.dense = nn.Linear(hidden_d * mlp_ratio, hidden_d)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
