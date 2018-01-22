import sys
import argparse

def str2bool(v):
    is_true = v.lower() in ("yes", "true", "t", "1")
    is_false = v.lower() in ("no", "false", "f", "0")

    if (not is_true) and (not is_false):
        raise Exception("Invalid str for bool cast: {}".format(v))

    return is_true

# Custom argument parser for file reading.
class ArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()

    def parse_args(self, *args, **kwargs):
        parser = super().parse_args(*args, **kwargs)

        # Check None.
        for key, val in parser.__dict__.items():
            if val is None:
                raise Exception("The value of \"{}\" is not set.".format(key))

        return parser


def make_arg_parser():
    parser = ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--extension", type=str)
    parser.add_argument("--z_size", type=int)
    parser.add_argument("--voxel_size", type=int)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--move", type=str2bool)
    parser.add_argument("--rotate", type=str2bool)
    parser.add_argument("--invert", type=str2bool)
    parser.add_argument("--energy_limit", type=float, nargs=2)
    parser.add_argument("--energy_scale", type=float, nargs=2)
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--bottom_size", type=int)
    parser.add_argument("--bottom_filters", type=int)
    parser.add_argument("--top_size", type=int)
    parser.add_argument("--filter_unit", type=int)
    parser.add_argument("--g_learning_rate", type=float)
    parser.add_argument("--d_learning_rate", type=float)
    parser.add_argument("--minibatch", type=str2bool)
    parser.add_argument("--minibatch_kernel_size", type=int)
    parser.add_argument("--minibatch_dim_per_kernel", type=int)
    parser.add_argument("--l2_loss", type=str2bool)
    parser.add_argument("--train_gen_per_disc", type=int)
    parser.add_argument("--device", type=str)

    return parser


def write_config_log(args, date):
    logdir = args.logdir

    items = sorted(list(args.__dict__.items()))
    with open("{}/config-{}".format(logdir, date), "w") as f:
        for key, val in items:
            f.write("{} {}\n".format(key, val))


def make_args_from_config(config):
    """Convert config file to args that can be used for argparser."""
    # Get file name (not a path)
    #config_name = config.split("/")[-1]
    # Get parent path
    #config_folder = "/".join(config.split("/")[:-1])

    args = list()
    with open(config, "r") as f:
        for line in f:
            # Remove trash chars
            line = line.replace("[", "")
            line = line.replace(",", "")
            line = line.replace("]", "")

            line = "--" + line

            args += line.split()

    #args += ["--logdir", config_folder]

    #print(args)

    return args


def _test():
    parser = make_arg_parser()
    args = parser.parse_args(sys.argv[1:])

    print(args.__dict__)


if __name__ == "__main__":
    _test()
