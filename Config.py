import argparse

import pdb

"""
set configuration arguments as class attributes
"""
class Config(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


"""
get configuration arguments
"""
def get_config(**kwargs):

    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data_dir', type = str, default = '/localdisk/hh/dataset/text/shakespeare',
                        help = 'data directory. Should contain the file input.txt with input data')

    # optimization
    parser.add_argument('--seq_length', type = int, default = 50, help = 'number of timesteps to unroll for')
    parser.add_argument('--batch_size', type = int, default = 50, help = 'number of sequences to train on in parallel')
    parser.add_argument('--train_frac', type = float, default = 0.95, help = 'fraction of data that goes into train set')
    parser.add_argument('--val_frac', type = float, default = 0.05, help = 'fraction of data that goes into validation set')

    args = parser.parse_args()

    # namespace -> dictionary
    args = vars(args)
    args.update(kwargs)

    return Config(**args)

    # # lstm parameters
    # parser.add_argument('--input_size', type=int, default=512)
    # parser.add_argument('--hidden_size', type=int, default=512)
    # parser.add_argument('--num_layers', type=int, default=1)
    #
    # # training parameters
    # parser.add_argument('--max_epoch_num', type=int, default=1)
    # parser.add_argument('--learning_rate', type=float, default=0.0001)
    # parser.add_argument('--weight_decay', type=float, default=0.005)
    #
    # # dataset path
    # # video_ds_dir: downsampled-video
    # # combined dir: data combining video and ground-truth
    # parser.add_argument('--video_dir_tvsum', type=str, default=Path('/localdisk/videosm/dataset/video/TVSum'))
    # parser.add_argument('--video_ds_dir_tvsum', type=str, default=Path('/localdisk/videosm/dataset/videods/TVSum'))
    # parser.add_argument('--gt_dir_tvsum', type=str, default=Path('/localdisk/videosm/dataset/gt/TVSum'))
    # parser.add_argument('--combined_dir_tvsum', type=str, default=Path('/localdisk/videosm/dataset/combined/TVSum'))
    #
    # parser.add_argument('--video_dir_summe', type=str, default=Path('/localdisk/videosm/dataset/video/SumMe'))
    # parser.add_argument('--video_ds_dir_summe', type=str, default=Path('/localdisk/videosm/dataset/videods/SumMe'))
    # parser.add_argument('--gt_dir_summe', type=str, default=Path('/localdisk/videosm/dataset/gt/SumMe'))
    # parser.add_argument('--combined_dir_summe', type=str, default=Path('/localdisk/videosm/dataset/combined/SumMe'))
    #
    # parser.add_argument('--video_dir_youtube', type=str, default=Path('/localdisk/videosm/dataset/video/Youtube'))
    # parser.add_argument('--video_ds_dir_youtube', type=str, default=Path('/localdisk/videosm/dataset/videods/Youtube'))
    # parser.add_argument('--gt_dir_youtube', type=str, default=Path('/localdisk/videosm/dataset/gt/Youtube'))
    # parser.add_argument('--combined_dir_youtube', type=str, default=Path('/localdisk/videosm/dataset/combined/Youtube'))
    #
    # parser.add_argument('--video_dir_openvideo', type=str, default=Path('/localdisk/videosm/dataset/video/OpenVideo'))
    # parser.add_argument('--video_ds_dir_openvideo', type=str,
    #                     default=Path('/localdisk/videosm/dataset/videods/OpenVideo'))
    # parser.add_argument('--gt_dir_openvideo', type=str, default=Path('/localdisk/videosm/dataset/gt/OpenVideo'))
    # parser.add_argument('--combined_dir_openvideo', type=str,
    #                     default=Path('/localdisk/videosm/dataset/combined/OpenVideo'))
    #
    # parser.add_argument('--dataset_dir', type=str, default=Path('/localdisk/videosm/dataset'))
    # parser.add_argument('--gt_dir', type=str, default=Path('/localdisk/videosm/dataset/gt'))
    # parser.add_argument('--combined_dir', type=str, default=Path('/localdisk/videosm/dataset/combined'))
    #
    # # model path
    # parser.add_argument('--c3d_model_dir', type=str, default=Path('/localdisk/videosm/model/'))
    # parser.add_argument('--videosm_model_dir', type=str, default=Path('/localdisk/videosm/model/'))