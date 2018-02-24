from tqdm import tqdm

from model import *
from generate import *
from util import *
from config import *

import pdb

# generate training set randomly
def get_train_set(file, vocab, seq_length, batch_size):
    # two tensors: (batch_size * seq_length)
    input = torch.LongTensor(batch_size, seq_length)
    target = torch.LongTensor(batch_size, seq_length)

    file_length = len(file)
    for i in range(batch_size): # for every sample (row) in this batch
        start_idx = random.randint(0, file_length - seq_length) # a random number
        end_idx = start_idx + seq_length
        seq = file[start_idx: end_idx]  # len: seq_length
        # input: (a1, a2, ..., an-1, an) -> target: (a2, ..., an-1, an, a1)
        input[i] = text_to_tensor(seq, vocab)
        target[i][0: -1] = input[i][1:].clone()
        target[i][-1] = input[i][0]

    input = Variable(input)
    target = Variable(target)

    # ship to gpu variable
    if torch.cuda.is_available() and config.cuda:
        input = input.cuda()
        target = target.cuda()

    return input, target    # both of them: (batch_size, seq_length)

def train(train_set, val_set, config):

    # initialize a model
    char_rnn = CharRNN(len(char_to_int), config.hidden_size, len(char_to_int),
                       model = config.model, n_layers = config.n_layers)

    # wrap them in Variable
    # train_input_set : (train_batches, batch_size, seq_length); the rest are similar
    train_input_set, train_target_set = Variable(train_set[0]), Variable(train_set[0])
    val_input_set, val_target_set = Variable(val_set[0]), Variable(val_set[0])

    # ship to gpu
    if torch.cuda.is_available() and config.cuda:
        char_rnn.cuda()
        train_input_set, train_target_set = train_input_set.cuda(), train_target_set.cuda()
        val_input_set, val_target_set = val_input_set.cuda(), val_target_set.cuda()

    # compute initial hidden states
    init_hidden = char_rnn.init_hidden(config.batch_size)  # (n_layers * n_directions, batch_size, hidden_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(char_rnn.parameters(), lr = config.learning_rate)

    best_val_loss = sys.float_info.max
    try:
        for epoch_idx in tqdm(range(1, config.max_epochs + 1)): # for evary epoch
            print("Training for %d epochs..." % epoch_idx)
            running_loss = 0.0
            for batch_idx in range(1, train_input_set.shape[0] + 1):   # for every batch
                # for every batch
                input = train_input_set[batch_idx - 1]
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                output, _ = char_rnn(input, init_hidden)
                # compute loss for this batch
                loss = 0
                for i in range(config.batch_size):  # for every sequence in this batch
                    loss += criterion(output[i], train_target_set[batch_idx - 1][i])

                # backward
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if batch_idx % config.print_interval == 0:  # print_interval batches
                    print('[%d, %4d] loss: %.3f' % (epoch_idx, batch_idx, running_loss / config.print_interval))
                    running_loss = 0.0

                    # validate model
                    val_loss = 0
                    for val_batch_idx in range(1, val_input_set.shape[0] + 1):
                        # for every batch
                        val_input = val_input_set[val_batch_idx - 1]
                        val_output, _ = char_rnn(val_input, init_hidden)
                        for i in range(config.batch_size):  # for every sequence in this batch
                            val_loss += criterion(val_output[i], val_target_set[val_batch_idx - 1][i])
                    val_loss /= val_input_set.shape[0]  # loss per batch
                    print('Validation loss: %.3f' % val_loss)
                    if val_loss.data[0] < best_val_loss:
                        print('Saving model [%d, %4d]...' % (epoch_idx, batch_idx))
                        torch.save(char_rnn.state_dict(), path.join(config.model_dir, config.model + '.pth'))
                        # to load: char_rnn = CharRNN(*args, **kwargs), char_rnn.load_state_dict(torch.load(PATH))
                        best_val_loss = val_loss.data[0]

    except KeyboardInterrupt:
        print("Saving before abnormal quit...")
        torch.save(char_rnn.state_dict(), os.path.join(config.model_dir, config.model + '.pth'))

if __name__ == '__main__':

    config = get_config()   # get configuration parameters

    # train_set = (input_set, target_set), the shape of input_set: (nbatches, batch_size, seq_length)
    # val_set and test_set are similar to train_set
    train_set, val_set, test_set, (char_to_int, int_to_char) = create_dataset(config)

    train(train_set, val_set, config)

