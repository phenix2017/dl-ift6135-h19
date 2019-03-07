import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
    # A helper function for producing N identical layers
    # (each with their own parameters).
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Problem 1
class RNN(nn.Module):
    # Implement a stacked vanilla RNN with Tanh nonlinearities
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size,
                 num_layers, dp_keep_prob):
        """
        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary
                      (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers
                      at each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the 
                      non-recurrent connections.
        """
        super(RNN, self).__init__()

        # Initialization of the parameters of the recurrent and fc layers.

        # Params
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

        # Input Embedding layer
        self.emb_layer = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.emb_size)

        # Hidden layers as a MduleList
        self.hidden_layers = nn.ModuleList()

        # Hidden layer module as a Sequential of linear and tanh
        def one_hidden_layer(in_features, out_features):
            return nn.Sequential(nn.Linear(in_features=in_features,
                                           out_features=out_features,
                                           bias=True),
                                 nn.Tanh())

        # First hidden layer
        self.hidden_layers.append(one_hidden_layer(
                                 in_features=(self.emb_size + self.hidden_size),
                                 out_features=self.hidden_size))

        # More hidden layers
        for _ in range(num_layers-1):
            self.hidden_layers.append(one_hidden_layer(
                                               in_features=(2*self.hidden_size),
                                               out_features=self.hidden_size))

        # Out layer
        self.out_layer = nn.Sequential(nn.Linear(in_features=self.hidden_size,
                                                 out_features=self.vocab_size,
                                                 bias=True),
                                       nn.Tanh())

        # Dropouts
        self.dropouts = nn.ModuleList()
        # At each time step,
        # one for input of each hidden layer, and one more for out layer
        for _ in range(seq_len):
            self.dropouts.append(nn.ModuleList())
            for _ in range(num_layers + 1):
                self.dropouts[-1].append(nn.Dropout(p=(1 - dp_keep_prob)))

        # Initialize all weights
        self.init_weights_uniform()

    def init_weights_uniform(self):
        # Initialize all the weights uniformly in the range [-0.1, 0.1]
        # and all the biases to 0 (in place)
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.uniform_(m.weight.data, a=-0.1, b=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0.)

    def init_hidden(self):
        # initialize the hidden states to zero
        """
        This is used for the first mini-batch in an epoch, only.
        """
        # return a parameter tensor of shape
        # (self.num_layers, self.batch_size, self.hidden_size)
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        # Compute the forward pass, using nested python for loops.
        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that
                      represent the index of the current token(s) in the
                      vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked
                      RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
            - Logits for the softmax over output tokens at every time-step.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
        """

        # Input to hidden layer - embedding of input
        emb_input = self.emb_layer(inputs)    # (seq_len, batch_size, emb_size)

        # To save outputs at each time step
        logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size],
                             device=inputs.device)

        # For each time step
        for t in range(self.seq_len):

            # Input at this time step to the first layer
            input_l = emb_input[t]        # (batch_size, emb_size)

            # For each layer
            for l, h_layer in enumerate(self.hidden_layers):

                # Concatenate dropout(input), and hidden state at this layer
                input_cat = torch.cat((self.dropouts[t][l](input_l),
                                       hidden[l]),
                                      dim=1)

                # Get hidden layer output
                h_layer_out_t = h_layer(input_cat)

                # Input for next layer
                input_l = h_layer_out_t

                # Hidden state for next time step
                hidden[l] = h_layer_out_t.clone()

            # Get output at this time step
            logits[t] = self.out_layer(self.dropouts[t][-1](h_layer_out_t))

        # Return logits: (seq_len, batch_size, vocab_size),
        #        hidden: (num_layers, batch_size, hidden_size)
        return logits, hidden

    def generate(self, input, hidden, generated_seq_len):
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        # 
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output
        # distribution, as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation
        # function here in order to compute the parameters of the categorical 
        # distributions to be sampled from at each time-step.

        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked
                      RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used 
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """

        # Input to hidden layer - embedding of input
        samples = input.view(1, -1)         # (1, batch_size)
        h_input = self.emb_layer(samples)   # (1, batch_size, emb_size)

        # For each time step
        for t in range(generated_seq_len):

            # Input at this time step
            input_t = h_input[0]        # (batch_size, emb_size)

            # To save hidden states for next time step
            hidden_next = []

            # For each layer
            for l, h_layer in enumerate(self.hidden_layers):

                # Hidden state at this layer
                hidden_l = hidden[l]    # (batch_size, hidden_size)

                # Concatenate input and hidden state
                input_cat = torch.cat((input_t, hidden_l), dim=1)

                # Get layer output
                h_layer_out_t = h_layer(input_cat)

                # Input for next layer
                input_t = h_layer_out_t

                # Hidden state for next time step
                hidden_next.append(h_layer_out_t)

            # Make hidden for next time step
            hidden = torch.stack(hidden_next)
            # (num_layers, batch_size, hidden_size)

            # Get output at this time step
            logits = self.out_layer(h_layer_out_t)    # (batch_size, vocab_size)
            token_out = torch.argmax(nn.Softmax()(logits), dim=1) # (batch_size)
            token_out = token_out.view(1, -1)       # (1, batch_size)
            # token_out = token_out.detach().view(1, -1)       # (1, batch_size)

            # Make input to next time step
            h_input = self.emb_layer(token_out)     # (1, batch_size, emb_size)

            # Append to samples
            samples = torch.cat((samples, token_out), dim=0)

        return samples


# Problem 2
class GRU(nn.Module): # Implement a stacked GRU RNN
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size,
                 num_layers, dp_keep_prob):
        super(GRU, self).__init__()

        # Params
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

        # Input Embedding layer
        self.emb_layer = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_size)

        # Transformations
        def transformation(in_features, out_features, bias=True):
            return nn.Sequential(nn.Linear(in_features=in_features,
                                           out_features=out_features,
                                           bias=bias),
                                 nn.Tanh())

        # Reset
        self.reset_tx = nn.ModuleList()
        for i in range(num_layers):
            # Input dimension
            input_dim = emb_size if i == 0 else hidden_size
            # Reset transformation
            self.reset_tx.append(transformation(
                                    in_features=input_dim+hidden_size,
                                    out_features=hidden_size)
                                )

        # Forget
        self.forget_tx = nn.ModuleList()
        for i in range(num_layers):
            # Input dimension
            input_dim = emb_size if i == 0 else hidden_size
            # Reset transformation
            self.forget_tx.append(transformation(
                                    in_features=input_dim+hidden_size,
                                    out_features=hidden_size)
                                 )

        # Reset gate
        self.reset_g = nn.ModuleList()
        for _ in range(num_layers):
            # Input dimension
            input_dim = emb_size if i == 0 else hidden_size
            # Reset transformation
            self.reset_g.append(transformation(
                                    in_features=input_dim+hidden_size,
                                    out_features=hidden_size)
                               )

        # Output layer
        self.out_layer = transformation(in_features=hidden_size,
                                        out_features=vocab_size)

        # Dropouts
        self.dropouts = nn.ModuleList()
        # At each time step,
        # one for input of each hidden layer, and one more for out layer
        for _ in range(seq_len):
            self.dropouts.append(nn.ModuleList())
            for _ in range(num_layers + 1):
                self.dropouts[-1].append(nn.Dropout(p=(1 - dp_keep_prob)))

        # Initialize all weights
        self.init_weights_uniform()

    def init_weights_uniform(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.uniform_(m.weight.data, a=-0.1, b=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0.)

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):

        # Input to hidden layer - embedding of input
        h_input = self.emb_layer(inputs)    # (seq_len, batch_size, emb_size)

        # To save outputs at each time step
        logits = []

        # For each time step
        for t in range(self.seq_len):

            # Input at this time step at the first layer
            input_l = h_input[t]        # (batch_size, emb_size)

            # To save hidden states for next time step
            hidden_next_t = []

            # For each layer
            for l, (reset_tx_l, forget_tx_l, reset_gate_l) in enumerate(zip(
                                                                self.reset_tx,
                                                                self.forget_tx,
                                                                self.reset_g
                                                               )):

                # dropout(input) at this layer
                input_l_dropout = self.dropouts[t][l](input_l)

                # Concatenate dropout(input), and hidden state at this layer
                input_cat = torch.cat((input_l_dropout, hidden[l]), dim=1)

                # Get reset
                r_t = reset_tx_l(input_cat)

                # Get forget
                z_t = forget_tx_l(input_cat)

                # Concatenate input and reset (.) hidden state for reset gate
                reset_in = torch.cat((input_l_dropout, r_t*hidden[l]),
                                     dim=1)

                # Get reset gate output
                h_tild_t = reset_gate_l(reset_in)

                # Get output
                h_layer_out_t = (1 - z_t)*hidden[l] + z_t*h_tild_t

                # Input for next layer
                input_l = h_layer_out_t

                # Hidden state for next time step
                hidden_next_t.append(h_layer_out_t)

            # Make hidden for next time step
            hidden = torch.stack(hidden_next_t)

            # Get output at this time step
            logits.append(self.out_layer(self.dropouts[t][-1](h_layer_out_t)))

        # Return logits: (seq_len, batch_size, vocab_size),
        #        hidden: (num_layers, batch_size, hidden_size)
        return torch.stack(logits), hidden

    def generate(self, input, hidden, generated_seq_len):

        # Input to hidden layer - embedding of input
        samples = input.view(1, -1)         # (1, batch_size)
        h_input = self.emb_layer(samples)   # (1, batch_size, emb_size)

        # For each time step
        for t in range(generated_seq_len):

            # Input at this time step at the first layer
            input_l = h_input[0]        # (batch_size, emb_size)

            # To save hidden states for next time step
            hidden_next_t = []

            # For each layer
            for l, (reset_tx_l, forget_tx_l, reset_gate_l) in enumerate(zip(
                                                                self.reset_tx,
                                                                self.forget_tx,
                                                                self.reset_g
                                                               )):

                # dropout(input) at this layer
                input_l_dropout = self.dropouts[t][l](input_l)

                # Concatenate dropout(input), and hidden state at this layer
                input_cat = torch.cat((input_l_dropout, hidden[l]), dim=1)

                # Get reset
                r_t = reset_tx_l(input_cat)

                # Get forget
                z_t = forget_tx_l(input_cat)

                # Concatenate input and reset (.) hidden state for reset gate
                reset_in = torch.cat((input_l_dropout, r_t*hidden[l]),
                                     dim=1)

                # Get reset gate output
                h_tild_t = reset_gate_l(reset_in)

                # Get output
                h_layer_out_t = (1 - z_t)*hidden[l] + z_t*h_tild_t

                # Input for next layer
                input_l = h_layer_out_t

                # Hidden state for next time step
                hidden_next_t.append(h_layer_out_t)

            # Make hidden for next time step
            hidden = torch.stack(hidden_next_t)

            # Get output at this time step
            logits = self.out_layer(self.dropouts[t][-1](h_layer_out_t))
            token_out = torch.argmax(nn.Softmax()(logits), dim=1) # (batch_size)
            token_out = token_out.view(1, -1)       # (1, batch_size)

            # Make input to next time step
            h_input = self.emb_layer(token_out)     # (1, batch_size, emb_size)

            # Append to samples
            samples = torch.cat((samples, token_out), dim=0)

        return samples      # (generated_seq_len, batch_size)


# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned
WordEmbedding and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code
that identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#-------------------------------------------------------------------------------

# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units 

        # TODO: create/initialize any necessary parameters or layers
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout
        
    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value all have size:
        # (batch_size, seq_len, self.n_units, self.d_k),
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax 
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.

        return # size: (batch_size, seq_len, self.n_units)


#-------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



#-------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size,
                                                                   dropout), 2)
 
    def forward(self, x, mask):
        # apply the self-attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # apply the position-wise MLP
        return self.sublayer[1](x, self.feed_forward)


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks):
        # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#-------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#-------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

