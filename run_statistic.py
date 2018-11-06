import argparse
from Network import Network

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default=[524*100*3, 572],
        type=list)
    parser.add_argument(
        "--labels_path",
        default='labels',
        type=str)
    parser.add_argument(
        "--pretrained_path",
        default='checkpoints/pass_0.ckpt',
        type=str)
    parser.add_argument(
        "--max_data_len",
        default=524,
        type=int)
    parser.add_argument(
        "--test_data_path",
        default='test.pickle',
        type=str)
    parser.add_argument(
        "--output_path",
        default='./',
        type=str)
    parser.add_argument(
        "--use_gpu",
        default=True,
        type=bool)
    args = parser.parse_args()

    nn = Network(shapes=args.sizes,
                 label_path=args.labels_path,
                 pretrained_path=args.pretrained_path,
                 max_data_len=args.max_data_len,
                 use_gpu=args.use_gpu)
    statisctic_report = nn.get_statistics(args.test_data_path)
    print(statisctic_report)
