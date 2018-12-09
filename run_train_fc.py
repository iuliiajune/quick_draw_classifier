import argparse
from FCNetwork import FCNetwork

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float)
    parser.add_argument(
        "--use_gpu",
        default=True,
        type=bool)
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int)
    parser.add_argument(
        "--epochs",
        default=100,
        type=int)
    parser.add_argument(
        "--sizes",
        default=[3*28*28,
                 2*28*28,
                 14*28],
        type=list)
    parser.add_argument(
        "--train_data_path",
        default='./5_classes/train',
        type=str)
    parser.add_argument(
        "--validate_data_path",
        default='./5_classes/validate',
        type=str)
    parser.add_argument(
        "--labels_path",
        default='./5_classes/labels',
        type=str)
    parser.add_argument(
        "--pretrained_path",
        default=None,
        type=str)
    parser.add_argument(
        "--save_path",
        default='5_classes/checkpoints',
        type=str)
    parser.add_argument(
        "--log_dir",
        default='5_classes/log_dir',
        type=str)
    parser.add_argument(
        "--img_size",
        default=28,
        type=int)
    args = parser.parse_args()
    nn = FCNetwork(label_path=args.labels_path,
                   pretrained_path=args.pretrained_path,
                   use_gpu=args.use_gpu,
                   log_dir=args.log_dir,
                   img_size=args.img_size)
    nn.create(args.sizes)
    nn.train(train_data_path=args.train_data_path,
             validate_data_path=args.validate_data_path,
             save_path=args.save_path,
             batch_size=args.batch_size,
             learning_rate=args.learning_rate,
             epochs=args.epochs)
