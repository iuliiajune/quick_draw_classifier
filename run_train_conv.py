import argparse
from ConvNetwork import ConvNetwork

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning_rate",
        default=0.002,
        type=float)
    parser.add_argument(
        "--use_gpu",
        default=True,
        type=bool)
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int)
    parser.add_argument(
        "--epochs",
        default=100,
        type=int)
    parser.add_argument(
        "--sizes",
        default=[(16, 3, 2, 2),
                 (16, 3, 2, 2),
                 (32, 3, 2, 2)],
        type=list)
    parser.add_argument(
        "--train_data_path",
        default='20_classes/train',
        type=str)
    parser.add_argument(
        "--validate_data_path",
        default='20_classes/validate',
        type=str)
    parser.add_argument(
        "--labels_path",
        default='20_classes/labels',
        type=str)
    parser.add_argument(
        "--pretrained_path",
        default=None,
        type=str)
    parser.add_argument(
        "--save_path",
        default='20_classes/checkpoints_conv',
        type=str)
    parser.add_argument(
        "--log_dir",
        default='20_classes/log_dir_conv',
        type=str)
    parser.add_argument(
        "--img_size",
        default=32,
        type=int)
    args = parser.parse_args()
    nn = ConvNetwork(label_path=args.labels_path,
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
