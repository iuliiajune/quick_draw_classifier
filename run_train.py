import argparse
from Network import Network

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning_rate",
        default=0.1,
        type=float)
    parser.add_argument(
        "--batch_size",
        default=200,
        type=int)
    parser.add_argument(
        "--epochs",
        default=10,
        type=int)
    parser.add_argument(
        "--sizes",
        default=(286*200*3, 572, 10),
        type=list)
    parser.add_argument(
        "--train_data_path",
        default='validate.pickle',
        type=str)
    parser.add_argument(
        "--validate_data_path",
        default='validate.pickle',
        type=str)
    parser.add_argument(
        "--labels_path",
        default='labels',
        type=str)
    parser.add_argument(
        "--pretrained_path",
        default=None,
        type=str)
    parser.add_argument(
        "--max_data_len",
        default=286,
        type=int)
    parser.add_argument(
        "--save_path",
        default='checkpoints',
        type=str)
    args = parser.parse_args()
    nn = Network(shapes=args.sizes,
                 label_path=args.labels_path,
                 pretrained_path=args.pretrained_path,
                 max_data_len=args.max_data_len)
    nn.train(train_data_path=args.train_data_path,
             validate_data_path=args.validate_data_path,
             save_path=args.save_path,
             batch_size=args.batch_size,
             learning_rate=args.learning_rate,
             epochs=args.epochs)
