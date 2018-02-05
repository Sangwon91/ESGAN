import os
import sys
import glob
import shutil
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


def make_arg_parser():
    parser = ArgumentParser(fromfile_prefix_chars='@')

    # Required arguments
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--extension", type=str, required=True)
    parser.add_argument("--z_size", type=int, required=True)
    parser.add_argument("--voxel_size", type=int, required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--move", type=str2bool, required=True)
    parser.add_argument("--rotate", type=str2bool, required=True)
    parser.add_argument("--invert", type=str2bool, required=True)
    parser.add_argument("--energy_limit", type=float, nargs=2, required=True)
    parser.add_argument("--energy_scale", type=float, nargs=2, required=True)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--save_every", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--bottom_size", type=int, required=True)
    parser.add_argument("--bottom_filters", type=int, required=True)
    parser.add_argument("--top_size", type=int, required=True)
    parser.add_argument("--filter_unit", type=int, required=True)
    parser.add_argument("--g_learning_rate", type=float, required=True)
    parser.add_argument("--d_learning_rate", type=float, required=True)
    parser.add_argument("--minibatch", type=str2bool, required=True)
    parser.add_argument("--minibatch_kernel_size", type=int, required=True)
    parser.add_argument("--minibatch_dim_per_kernel", type=int, required=True)
    parser.add_argument("--l2_loss", type=str2bool, required=True)
    parser.add_argument("--train_gen_per_disc", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)

    # Optional arguments
    parser.add_argument("--restore_config", type=str)

    return parser


def make_frac2cell_arg_parser():
    parser = ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--validset_path", type=str, required=True)
    parser.add_argument("--extension", type=str, required=True)
    parser.add_argument("--voxel_size", type=int, required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--rotate", type=str2bool, required=True)
    parser.add_argument("--move", type=str2bool, required=True)
    parser.add_argument("--invert", type=str2bool, required=True)
    parser.add_argument("--energy_limit", type=float, nargs=2, required=True)
    parser.add_argument("--energy_scale", type=float, nargs=2, required=True)
    parser.add_argument("--cell_length_scale",
                            type=float, nargs=2, required=True)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--save_every", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--top_size", type=int, required=True)
    parser.add_argument("--filter_unit", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--scale_factor", type=float, required=True)
    parser.add_argument("--device", type=str, required=True)

    # Optional arguments
    parser.add_argument("--restore_config", type=str)

    return parser


def write_config_log(args, date):
    logdir = args.logdir

    # Try to make parent folder.
    try:
        os.makedirs(logdir)
    except Exception as e:
        print("Error:", e, "but keep going.")

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
            # Neglect restore_ckpt...
            if "restore" in line:
                continue
            # Remove trash chars
            line = line.replace("[", "")
            line = line.replace(",", "")
            line = line.replace("]", "")

            line = "--" + line

            args += line.split()

    return args


def cache_ckpt_from_config(*, cache_folder, config):
    # Get file name (not a path)
    config_name = config.split("/")[-1]
    # Get parent path
    config_folder = "/".join(config.split("/")[:-1])
    # Extract path
    date = "-".join(config_name.split("-")[1:])

    expression = "{}/save-{}-*".format(config_folder, date)
    ckpts = glob.glob(expression)

    try:
        for f in ckpts:
            shutil.copy2(f, cache_folder)
    except Exception as e:
        raise Exception(str(e) + ", Terminate program")

    ckpt = ".".join(ckpts[0].split(".")[:-1])
    ckpt = ckpt.split("/")[-1]
    ckpt = "{}/{}".format(cache_folder, ckpt)

    return ckpt


def _test():
    """
    parser = make_arg_parser()
    args = parser.parse_args(sys.argv[1:])

    print(args.__dict__)
    """

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        ckpt = cache_ckpt_from_config(cache_folder=temp_dir, config=sys.argv[1])
        print(ckpt)

if __name__ == "__main__":
    _test()
