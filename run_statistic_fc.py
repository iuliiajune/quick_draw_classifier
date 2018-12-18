import argparse
from FCNetwork import FCNetwork

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default=[
            # 3*28*28,
            #      2*28*28,
                 14*28],
        type=list)
    parser.add_argument(
        "--labels_path",
        default='5_classes/labels',
        type=str)
    parser.add_argument(
        "--pretrained_path",
        default='5_classes/checkpoints_fc_3/pass_25.ckpt',
        type=str)
    parser.add_argument(
        "--test_data_path",
        default='5_classes/test',
        type=str)
    parser.add_argument(
        "--output_path",
        default='./',
        type=str)
    parser.add_argument(
        "--use_gpu",
        default=True,
        type=bool)
    parser.add_argument(
        "--img_size",
        default=28,
        type=int)
    args = parser.parse_args()

    nn = FCNetwork(label_path=args.labels_path,
                     pretrained_path=args.pretrained_path,
                     use_gpu=args.use_gpu,
                     img_size=args.img_size)
    nn.create(args.sizes)
    statisctic_report = nn.get_statistics(args.test_data_path)
    print(statisctic_report)
