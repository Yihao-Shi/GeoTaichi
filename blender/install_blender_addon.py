import argparse
import os
import shutil
import time


ADDON_NAME = 'GeoTaichi'
COPIED_SUFFIXES = {'.py', '.toml', '.json', '.md'}


def copy_file(src_dir, out_dir, file):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    source_file_path = os.path.join(src_dir, file)
    shutil.copy(source_file_path, os.path.join(out_dir, file))


def copy_files(src_dir, out_dir):
    for file in os.listdir(src_dir):
        if file.startswith('._'):
            continue
        if os.path.isdir(os.path.join(src_dir, file)):
            src_subdir = os.path.join(src_dir, file)
            out_subdir = os.path.join(out_dir, file)
            copy_files(src_subdir, out_subdir)
        elif os.path.splitext(file)[1].lower() in COPIED_SUFFIXES:
            copy_file(src_dir, out_dir, file)


def addons_path(value):
    resolved = os.path.abspath(os.path.expanduser(value))
    if (
        os.path.basename(resolved) != 'addons'
        or os.path.basename(os.path.dirname(resolved)) != 'scripts'
    ):
        raise argparse.ArgumentTypeError('path must end with scripts/addons')
    return resolved


def install(destination):
    print("Installing...")
    addon_out_path = os.path.join(destination, ADDON_NAME)
    if os.path.exists(addon_out_path):    # delete the old addon
        shutil.rmtree(addon_out_path)

    addon_input_path = os.path.dirname(os.path.abspath(__file__))
    copy_files(addon_input_path, addon_out_path)

    print("Done.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Install the current Blender addon.')
    parser.add_argument(
        '--addon-path',
        required=True,
        type=addons_path,
        help='Blender user scripts/addons directory',
    )
    parser.add_argument('-k', '--keep-refreshing', action='store_true')
    parser.add_argument('-y', '--yes', action='store_true', help='skip the replacement prompt')
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_args(argv)
    addon_path = os.path.join(arguments.addon_path, ADDON_NAME)
    if arguments.keep_refreshing:
        while True:
            time.sleep(1)
            install(arguments.addon_path)
    else:
        print(f"This will remove everything under {addon_path}.")
        if not arguments.yes:
            print("Are you sure? [y/N]")
            if input() != 'y':
                print("exiting")
                return
        install(arguments.addon_path)


if __name__ == '__main__':
    main()
