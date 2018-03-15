from config import *
from utils import *
from vis import *

if __name__ == '__main__':

    config = get_config()  # get configuration parameters

    _, _, test_set, (char_to_int, int_to_char) = create_dataset(config)

    # visualize cell state values
    # vis_cell(test_set, int_to_char, len(char_to_int), config)

    # visualize three gates: input, forget, output
    vis_gate(test_set, len(char_to_int), config)