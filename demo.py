import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import *
from util import *
from config import *

import pdb

# build and train a model
def train(train_set, val_set, vocab_size, config):

    # initialize a model
    char_rnn = CharRNN(vocab_size, config.hidden_size, vocab_size, model = config.model, n_layers = config.n_layers)
    # ship to gpu if possible
    if torch.cuda.is_available() and config.cuda:
        char_rnn.cuda()

    # train_input_set : (train_batches, batch_size, seq_length); the rest are similar
    train_input_set, train_target_set = train_set[0], train_set[1]
    val_input_set, val_target_set = val_set[0], val_set[1]

    criterion = nn.CrossEntropyLoss()   # include softmax
    optimizer = torch.optim.Adam(char_rnn.parameters(), lr = config.learning_rate)
    # learning rate decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.95)

    best_val_loss = sys.float_info.max
    try:
        for epoch_idx in tqdm(range(1, config.max_epochs + 1)): # for evary epoch
            print("Training for %d epochs..." % epoch_idx)
            running_loss = 0.0
            scheduler.step()    # lr decay

            # initialize hidden states for every epoch
            hidden = char_rnn.init_hidden(config.batch_size)  # (n_layers * n_directions, batch_size, hidden_size)
            # ship to gpu if possible
            if torch.cuda.is_available() and config.cuda:
                hidden = tuple([x.cuda() for x in hidden])

            for batch_idx in range(1, train_input_set.shape[0] + 1):   # for every batch
                # for every batch
                optimizer.zero_grad()   # zero the parameter gradients

                train_input = train_input_set[batch_idx - 1]

                # ship to gpu if possible
                if torch.cuda.is_available() and config.cuda:
                    train_input = train_input.cuda()

                # compute loss for this batch
                loss = 0
                for i in range(config.seq_length):  # for every time step in this batch
                    # forward pass
                    train_output, hidden = char_rnn(Variable(train_input[:, i]), hidden)
                    # add up loss at each time step
                    loss += criterion(train_output.view(config.batch_size, -1).cpu(),
                                      Variable(train_target_set[batch_idx - 1][:, i]))

                # detach hidden state from current computational graph for back-prop
                for x in hidden:
                    x.detach_()

                # backward
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if batch_idx % config.print_interval == 0:  # print_interval batches
                    print('[%d, %4d] loss: %.3f' % (epoch_idx, batch_idx, running_loss / config.print_interval))
                    running_loss = 0.0

                    '''
                    # validate model
                    val_loss = 0
                    # for every batch
                    for val_batch_idx in range(1, val_input_set.shape[0] + 1):

                        val_input = val_input_set[val_batch_idx - 1]

                        # ship to gpu if possible
                        if torch.cuda.is_available() and config.cuda:
                            val_input = val_input.cuda()

                        for i in range(config.seq_length):  # for every time step in this batch
                            # forward pass
                            val_output, _ = char_rnn(Variable(val_input[:, i]), hidden)
                            # add up loss at each time step
                            val_loss += criterion(val_output.view(config.batch_size, -1).cpu(),
                                                  Variable(val_target_set[val_batch_idx - 1][:, i]))

                    val_loss /= val_input_set.shape[0]  # loss per batch
                    print('Validation loss: %.3f' % val_loss)

                    # save the best model sofar
                    if val_loss.data[0] < best_val_loss:
                        print('Saving model [%d, %4d]...' % (epoch_idx, batch_idx))
                        torch.save(char_rnn.state_dict(), path.join(config.model_dir, config.model + '.pth'))
                        # to load a saved model: char_rnn = CharRNN(*args, **kwargs), char_rnn.load_state_dict(torch.load(PATH))
                        best_val_loss = val_loss.data[0]
                    '''
        torch.save(char_rnn.state_dict(), path.join(config.model_dir, config.model + '.pth'))

    except KeyboardInterrupt:
        print("Saving before abnormal quit...")
        torch.save(char_rnn.state_dict(), path.join(config.model_dir, config.model + '.pth'))

# use the trained model to make prediction
def pred(test_set, train_set, val_set, int_to_char, vocab_size, config):
    # not existing trained model, train a new one
    if not path.exists(path.join(config.model_dir, config.model + '.pth')):
        train(train_set, val_set, vocab_size, config)

    # load a trained model
    char_rnn = CharRNN(vocab_size, config.hidden_size, vocab_size, model = config.model, n_layers = config.n_layers)
    char_rnn.load_state_dict(torch.load(path.join(config.model_dir, config.model + '.pth')))
    char_rnn.eval()

    # ship to gpu if possible
    if torch.cuda.is_available() and config.cuda:
        char_rnn.cuda()

    # prepare test data
    test_input_set, test_target_set = test_set[0], test_set[1]

    # randomly choose a sequence in train_set to warm up the network
    train_input_set, _ = train_set[0], train_set[1] # train_input_set: (train_batches, batch_size, seq_length)
    # random batch index
    train_batch_idx = np.random.choice(train_input_set.shape[0])
    # random sequence index
    train_seq_idx = np.random.choice(config.batch_size)
    # random sequence
    warmup_seq = train_input_set[train_batch_idx][train_seq_idx].unsqueeze(0)

    # initialize hidden state
    hidden = char_rnn.init_hidden(1)   # here, batch_size = 1

    # ship to gpu if possible
    if torch.cuda.is_available() and config.cuda:
        warmup_seq = warmup_seq.cuda()
        hidden = tuple([x.cuda() for x in hidden])

    for i in range(config.seq_length):
        # get final hidden state
        _, hidden = char_rnn(Variable(warmup_seq[:, i]), hidden)

    for test_batch_idx in range(1, test_input_set.shape[0] + 1):
        # for every batch
        test_batch = test_input_set[test_batch_idx - 1]
        # for every sequence in this batch
        for test_seq_idx in range(1, config.batch_size + 1):
            # predicted int result (not character)
            pred = []

            # current sequence
            test_seq = test_batch[test_seq_idx - 1]

            # first character in current sequence
            idx = torch.LongTensor([test_seq[0]]).view(1, -1)

            # # ship to gpu if possible
            if torch.cuda.is_available() and config.cuda:
                idx = idx.cuda()

            # forward pass
            output, hidden = char_rnn(Variable(idx), hidden)  # idx: (1, 1, input_size)

            # choose the one with the highest value
            prob, idx = torch.topk(output.data, 1, dim = 2)
            # add predicted value
            pred.append(idx.cpu().squeeze()[0])

            # predict every remaining character in this sequence
            for i in range(1, len(test_seq)):
                # ship to gpu if possible
                if torch.cuda.is_available() and config.cuda:
                    idx = idx.cuda()

                # forward pass
                output, hidden = char_rnn(Variable(idx.view(1, -1)), hidden)

                # choose the one with the highest value
                prob, idx = torch.topk(output.data, 1, dim = 2)
                # add predicted value
                pred.append(idx.cpu().squeeze()[0])

            # calculate prediction accuracy
            pred = np.array(pred)
            target = test_target_set[test_batch_idx - 1][test_seq_idx - 1].numpy()

            '''
            accuracy = float(np.where(pred == target)[0].size) / pred.size
            print('Accuracy of [batch %2d, seq %2d] is ' % (test_batch_idx, test_seq_idx) + '{:.2%}'.format(accuracy))
            '''

            # convert target and pred from int to character
            target_text = []
            pred_text = []
            for i in range(len(target)):
                target_text.append(int_to_char[target[i]])
                pred_text.append(int_to_char[pred[i]])

            # display target text and predicted text
            print('Target ----------------------------------------------------------')
            print(''.join(target_text))  # convert from array to string
            print('Predicted -------------------------------------------------------')
            print(''.join(pred_text))

if __name__ == '__main__':

    config = get_config()   # get configuration parameters

    # train_set = (input_set, target_set), the shape of input_set: (nbatches, batch_size, seq_length)
    train_set, val_set, test_set, (char_to_int, int_to_char) = create_dataset(config)   # val_set and test_set are similar to train_set

    # train(train_set, val_set, len(char_to_int), config)

    pred(test_set, train_set, val_set, int_to_char, len(char_to_int), config)

