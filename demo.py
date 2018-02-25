from tqdm import tqdm
import numpy as np

from model import *
from util import *
from config import *

import pdb

# build and train a model
def train(train_set, val_set, config):

    # initialize a model
    char_rnn = CharRNN(len(char_to_int), config.hidden_size, len(char_to_int),
                       model = config.model, n_layers = config.n_layers)

    # wrap them in Variable
    # train_input_set : (train_batches, batch_size, seq_length); the rest are similar
    train_input_set, train_target_set = Variable(train_set[0]), Variable(train_set[0])
    val_input_set, val_target_set = Variable(val_set[0]), Variable(val_set[0])

    # compute initial hidden states
    init_hidden = char_rnn.init_hidden(config.batch_size)  # (n_layers * n_directions, batch_size, hidden_size)

    # ship to gpu if possible
    if torch.cuda.is_available() and config.cuda:
        char_rnn.cuda()
        init_hidden = (init_hidden[0].cuda(), init_hidden[1].cuda())
        train_input_set, train_target_set = train_input_set.cuda(), train_target_set.cuda()
        val_input_set, val_target_set = val_input_set.cuda(), val_target_set.cuda()

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
        torch.save(char_rnn.state_dict(), path.join(config.model_dir, config.model + '.pth'))

# use the trained model to make prediction
def pred(test_set, train_set, val_set, int_to_char, config):
    # not existing trained model, train a new one
    if not path.exists(path.join(config.model_dir, config.model + '.pth')):
        train(train_set, val_set, config)

    # load a trained model
    char_rnn = CharRNN(len(char_to_int), config.hidden_size, len(char_to_int),
                       model = config.model, n_layers = config.n_layers)
    char_rnn.load_state_dict(torch.load(path.join(config.model_dir, config.model + '.pth')))
    char_rnn.eval()

    # prepare test data, and # no need to convert test_target_set into Variable
    test_input_set, test_target_set = Variable(test_set[0]), test_set[1]

    # ship to gpu if possible
    if torch.cuda.is_available() and config.cuda:
        char_rnn.cuda()
        test_input_set = test_input_set.cuda()  # no need to ship test_target_set to gpu Variable

    # randomly choose a sequence in train_set to warm up the network
    train_input_set, _ = train_set[0], train_set[1] # train_set: (train_batches, batch_size, seq_length)
    train_batch_idx = np.random.choice(train_input_set.shape[0])
    train_seq_idx = np.random.choice(config.batch_size)
    warmup_seq = train_input_set[train_batch_idx][train_seq_idx].unsqueeze(0)

    init_hidden = (char_rnn.init_hidden(1)[0], char_rnn.init_hidden(1)[1])
    # ship to gpu if possible
    if torch.cuda.is_available() and config.cuda:
        warmup_seq = Variable(warmup_seq).cuda()
        # compute initial hidden states for warmup sequence
        init_hidden = (init_hidden[0].cuda(), init_hidden[1].cuda())

    # get final hidden state
    _, hidden = char_rnn(warmup_seq, init_hidden)

    target_text = []
    pred_text = []
    for test_batch_idx in range(1, test_input_set.shape[0] + 1):
        # for every batch
        test_input = test_input_set[test_batch_idx - 1]
        # for every sequence in this batch
        for test_seq_idx in range(1, config.batch_size + 1):
            pred = []
            for ch in test_input[test_seq_idx - 1]: # for every character in this sequence
                output, init_hidden = char_rnn(ch.view(1, -1), init_hidden)
                # choose the one with the highest value
                prob, idx = torch.topk(output.data, 1, dim = 2)
                if idx.is_cuda:
                    pred.append(idx.cpu().squeeze()[0])
                else:
                    pred.append(idx.squeeze()[0])

            # calculate prediction accuracy
            pred = np.array(pred)
            target = test_target_set[test_batch_idx - 1][test_seq_idx - 1].numpy()
            accuracy = float(np.where(pred == target)[0].size) / pred.size
            print('Accuracy of [batch %2d, seq %2d] is ' % (test_batch_idx, test_seq_idx) + '{:.2%}'.format(accuracy))

            # convert target and pred from int to character
            for idx in target:
                target_text.append(int_to_char[target[idx]])
                pred_text.append(int_to_char[pred[idx]])

    # display target text and predicted text
    print('Target ----------------------------')
    print(''.join(target_text)) # convert from array to string
    print('Predicted -------------------------')
    print(''.join(pred_text))

if __name__ == '__main__':

    config = get_config()   # get configuration parameters

    # train_set = (input_set, target_set), the shape of input_set: (nbatches, batch_size, seq_length)
    # val_set and test_set are similar to train_set
    train_set, val_set, test_set, (char_to_int, int_to_char) = create_dataset(config)

    # train(train_set, val_set, config)

    pred(test_set, train_set, val_set, int_to_char, config)

