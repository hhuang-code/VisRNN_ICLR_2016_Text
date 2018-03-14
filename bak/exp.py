from config import *
from utils import *
from vis import *

if __name__ == '__main__':

    config = get_config()  # get configuration parameters

    _, _, test_set, (char_to_int, int_to_char) = create_dataset(config)

    vis_cell(test_set, char_to_int, int_to_char, config, mode = 'keepone')