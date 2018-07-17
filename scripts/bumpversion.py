#!/usr/bin/env python3

import argparse
import subprocess

def extract(git_version):
    version_array = git_version.split('.')
    patch_array = version_array[2].split('-')
    major = int(version_array[0][1:])
    minor = int(version_array[1])
    patch = int(patch_array[0])
    postfix = ""

    if len(patch_array) == 3:
        postfix = patch_array[2]

    return major, minor, patch, postfix


def tag(major, minor, patch):
    version = "v{}.{}.{}".format(major, minor, patch)
    subprocess.getoutput("git tag -a {}".format(version))


def main(argv):
    git_version = subprocess.getoutput("git describe --always")
    major, minor, patch, postfix = extract(git_version)

    if argv.major:
        major += 1
        minor = 0
        patch = 0
        postfix = ""
        tag(major, minor, patch)
    elif argv.minor:
        minor += 1
        patch = 0
        postfix = ""
        tag(major, minor, patch)
    elif argv.patch:
        patch += 1
        postfix = ""
        tag(major, minor, patch)
    elif postfix != "":
        patch += 1

    print('{} {} {} {}'.format(major, minor, patch, postfix))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Set or read current version.')
    argparser.add_argument('--major', action='store_true',
            help='bumps major version')
    argparser.add_argument('--minor', action='store_true',
            help='bumps minor version')
    argparser.add_argument('--patch', action='store_true',
            help='bumps patch')
    argv = argparser.parse_args()
    main(argv)

