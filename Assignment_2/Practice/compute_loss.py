import collections
import numpy as np
import os
import time
import torch
import torch.nn as nn
import tqdm

from torch.autograd import Variable

from models import RNN, GRU 
from models import make_model as TRANSFORMER

###############################################################################
#
# DATA LOADING & PROCESSING
#
###############################################################################

# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())
    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")
    # Hmm
    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word

# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    # Hmm
    epoch_size = (batch_len - 1) // num_steps
    # Hmm
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    # Hmm
    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)


class Batch:
    "Data processing for the transformer. This class adds a mask to the data."
    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# LOAD DATA
print('Loading data from '+'data')
raw_data = ptb_raw_data(data_path='data')
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))

###############################################################################
#
# MAKE MODELS
#
###############################################################################


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    
    This prevents Pytorch from trying to backpropagate into previous input 
    sequences when we use the final hidden states from one mini-batch as the 
    initial hidden states for the next mini-batch.
    
    Using the final hidden states in this way makes sense when the elements of 
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)


def make_my_model(model_name, device, seq_len=35, batch_size=20, pt=None):
    #          --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
    #          --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
    #          --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best
    if model_name == 'RNN':
        model = RNN(emb_size=200, hidden_size=1500, 
                    seq_len=seq_len, batch_size=batch_size,
                    vocab_size=vocab_size, num_layers=2, 
                    dp_keep_prob=0.35) 
    elif model_name == 'GRU':
        model = GRU(emb_size=200, hidden_size=1500, 
                    seq_len=seq_len, batch_size=batch_size,
                    vocab_size=vocab_size, num_layers=2, 
                    dp_keep_prob=0.35)
    elif model_name == 'TRANSFORMER':
        model = TRANSFORMER(vocab_size=vocab_size, n_units=512, 
                            n_blocks=6, dropout=1.-0.9) 
        # these 3 attributes don't affect the Transformer's computations;
        # they are only used in run_epoch
        model.batch_size=128
        model.seq_len=35
        model.vocab_size=vocab_size
    else:
      print("ERROR: Model type not recognized.")
      return
    # Model to device
    model = model.to(device)
    # Load pt
    if pt is not None:
        model.load_state_dict(torch.load(pt, map_location=device))
    return model


# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda") 
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")

# Make models
# exp_dir = '/home/voletiv/GitHubRepos/dl-ift6135-h19/Assignment_2/Practice/'
exp_dir = '/home/voletivi/GitHubRepos/dl-ift6135-h19/Assignment_2/Practice'

model_names = ['RNN', 'GRU', 'TRANSFORMER']
model_state_dict_paths = [os.path.join(exp_dir, 'RNN', 'best_params.pt'),
                          os.path.join(exp_dir, 'GRU', 'best_params.pt'),
                          os.path.join(exp_dir, 'TX', 'best_params.pt')]

# models = []
# for model_name, model_state_dict_path in zip(model_names, model_state_dict_paths):
#     models.append(make_my_model(model_name, device, model_state_dict_path))


###############################################################################
#
# LOSS COMPUTATION
#
###############################################################################

loss_fn = nn.CrossEntropyLoss()


def compute_loss_at_every_t(model_name, model_state_dict_path, data, loss_fn):
    """
    One epoch of training/validation (depending on flag is_train).
    """
    print(model_name)
    model = make_my_model(model_name, device, model_state_dict_path)
    model.eval()
    losses = []
    epoch_size = ((len(data) // model.batch_size) - 1) // model.seq_len
    start_time = time.time()
    if model_name != 'TRANSFORMER':
        hidden = model.init_hidden().to(device)
    costs = 0.
    iters = 0
    losses = np.empty((0, 35))
    # LOOP THROUGH MINIBATCHES
    # import pdb; pdb.set_trace()
    for step, (x, y) in tqdm.tqdm(enumerate(ptb_iterator(data, model.batch_size, model.seq_len)),
                                  total=(len(data)//model.batch_size - 1)//model.seq_len):
        if model_name == 'TRANSFORMER':
            batch = Batch(torch.from_numpy(x).long().to(device))
            model.zero_grad()
            outputs = model.forward(batch.data, batch.mask).transpose(1,0)
            #print ("outputs.shape", outputs.shape)
        else:
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
            model.zero_grad()
            # hidden = repackage_hidden(hidden)
            hidden = model.init_hidden().to(device)
            outputs, hidden = model(inputs, hidden)
        # Target
        targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous()
        # LOSS COMPUTATION
        # This line currently averages across all the sequences in a mini-batch 
        # and all time-steps of the sequences.
        # For problem 5.3, you will (instead) need to compute the average loss 
        #at each time-step separately.
        outputs = outputs.detach().cpu().contiguous()
        loss = np.array([loss_fn(o, t).item() for o, t in zip(outputs, targets)])
        losses = np.vstack((losses, loss))
    # Return
    return np.mean(losses, 0)


model_losses = []
for model_name, model_state_dict_path in zip(model_names, model_state_dict_paths):
    model_losses.append(compute_loss_at_every_t(model_name, model_state_dict_path, valid_data, loss_fn))

np.save('model_losses.npy', model_losses)

# PLOT

model_losses = np.load('model_losses.npy')

# times = [len(m) for m in model_losses]
# minindex = np.argmin(times)
times = np.arange(35) + 1

for i in range(len(model_losses)):
    plt.plot(times, model_losses[i], '-o', alpha=0.7, label=model_names[i])

plt.legend()
# plt.title("Average validation loss at each time step")
plt.ylabel("Validation loss")
plt.xlabel("Time step in sequence")
# plt.show()
plt.savefig('5.1_loss_per_time_step.png', bbox_inches='tight', pad_inches=0.2)
plt.clf()
plt.close()


###############################################################################
#
# GENERATE SAMPLES
#
###############################################################################


def generate_samples(model_name, model_state_dict_path, generated_seq_len, num_of_samples):
    """
    One epoch of training/validation (depending on flag is_train).
    """
    print(model_name)
    x = np.random.choice(vocab_size, (1, num_of_samples))
    # x = np.array([9970]*num_of_samples).reshape((1, num_of_samples))
    inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)
    model = make_my_model(model_name, device, seq_len=generated_seq_len, batch_size=num_of_samples, pt=model_state_dict_path)
    model.eval()
    model.zero_grad()
    hidden = model.init_hidden().to(device)
    samples = model.generate(inputs, hidden, generated_seq_len-1) # (seq_len, batch_size)
    sample_words = [' '.join([id_2_word[t] for t in seq]) for seq in samples.detach().cpu().numpy().T]
    return sample_words

# RNN
model_name = model_names[0]
model_state_dict_path = model_state_dict_paths[0]
# 1
num_of_samples = 10
generated_seq_len = 35
RNN_samples_1 = generate_samples(model_name, model_state_dict_path, generated_seq_len, num_of_samples)
for i in RNN_samples_1: print(i+"\n")
# 2
num_of_samples = 10
generated_seq_len = 70
RNN_samples_2 = generate_samples(model_name, model_state_dict_path, generated_seq_len, num_of_samples)
for i in RNN_samples_2: print(i+"\n")

# GRU
model_name = model_names[1]
model_state_dict_path = model_state_dict_paths[1]
# 1
num_of_samples = 10
generated_seq_len = 35
GRU_samples_1 = generate_samples(model_name, model_state_dict_path, generated_seq_len, num_of_samples)
for i in GRU_samples_1: print(i+"\n")
# 2
num_of_samples = 10
generated_seq_len = 70
GRU_samples_2 = generate_samples(model_name, model_state_dict_path, generated_seq_len, num_of_samples)
for i in GRU_samples_2: print(i+"\n")


