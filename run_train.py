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
        default=(524*200*2, 572, 10),
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
        "--num_classes",
        default=10,
        type=int)
    parser.add_argument(
        "--pretrained_path",
        default=None,
        type=str)
    parser.add_argument(
        "--max_data_len",
        default=524,
        type=int)
    parser.add_argument(
        "--save_path",
        default='checkpoints',
        type=str)
    args = parser.parse_args()

    nn = Network(shapes=args.sizes,
                 num_classes=args.num_classes,
                 pretrained_path=args.pretrained_path,
                 max_data_len=args.max_data_len)
    nn.train(train_data_path=args.train_data_path,
             validate_data_path=args.validate_data_path,
             label_path=args.labels_path,
             save_path=args.save_path,
             batch_size=args.batch_size,
             learning_rate=args.learning_rate,
             epochs=args.epochs)
