import argparse
from parse_data import create_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary_path",
        default="binary",
        type=str)
    parser.add_argument(
        "--max_items_per_class",
        default=5000,
        type=int)
    parser.add_argument(
        "--num_classes",
        default=5,
        type=int)
    parser.add_argument(
        "--output_path",
        default='./5_classes',
        type=str)
    parser.add_argument(
        "--dataset_proportion",
        default='0.7,0.2,0.1',
        help='train,test,validate',
        type=str)
    args = parser.parse_args()
    train_part, test_part, validate_part = args.dataset_proportion.split(',')
    create_dataset(args.binary_path,
                   save_path=args.output_path,
                   train_part=float(train_part.strip()),
                   test_part=float(test_part.strip()),
                   validate_part=float(validate_part.strip()),
                   max_items_per_class=args.max_items_per_class,
                   num_classes=args.num_classes)
