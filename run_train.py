import argparse
from Network import Network

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
        default=128,
        type=int)
    parser.add_argument(
        "--epochs",
        default=100,
        type=int)
    parser.add_argument(
        "--sizes",
        default=[345*4*10, 345*3*10, 345*3*10, 345*2*10,
                 345*3*2,  345*3*2,  345*3,    345*3,   345*2],
        type=list)
    parser.add_argument(
        "--train_data_path",
        default='./100_classes/train',
        type=str)
    parser.add_argument(
        "--validate_data_path",
        default='./100_classes/validate',
        type=str)
    parser.add_argument(
        "--labels_path",
        default='./100_classes/labels',
        type=str)
    parser.add_argument(
        "--pretrained_path",
        default=None,
        type=str)
    parser.add_argument(
        "--max_data_len",
        default=834,
        type=int)
    parser.add_argument(
        "--save_path",
        default='checkpoints',
        type=str)
    parser.add_argument(
        "--log_dir",
        default='/mount/export0/log_dir/3',
        type=str)
    parser.add_argument(
        "--img_size",
        default=128,
        type=int)
    parser.add_argument(
        "--is_conv",
        default=True,
        type=bool)
    args = parser.parse_args()
    nn = Network(shapes=args.sizes,
                 label_path=args.labels_path,
                 pretrained_path=args.pretrained_path,
                 max_data_len=args.max_data_len,
                 use_gpu=args.use_gpu,
                 log_dir=args.log_dir,
                 img_size=args.img_size,
                 is_conv_nn=args.is_conv)
    nn.train(train_data_path=args.train_data_path,
             validate_data_path=args.validate_data_path,
             save_path=args.save_path,
             batch_size=args.batch_size,
             learning_rate=args.learning_rate,
             epochs=args.epochs)
