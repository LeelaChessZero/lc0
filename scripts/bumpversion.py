#!/usr/bin/env python3

import argparse
import os


VERSION_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../src/version.inc")
VERSION_CONTENT = """
#define LC0_VERSION_MAJOR {}
#define LC0_VERSION_MINOR {}
#define LC0_VERSION_PATCH {}
#define LC0_VERSION_POSTFIX "{}"
"""
VERSION_CONTENT = VERSION_CONTENT.strip()


def get_version():
    with open(VERSION_FILE, 'r') as f:
        major = int(f.readline().split()[2])
        minor = int(f.readline().split()[2])
        patch = int(f.readline().split()[2])
        postfix = f.readline().split()[2]

    postfix = postfix.replace('"', '')
    return major, minor, patch, postfix


def set_version(major, minor, patch, postfix=""):
    version_inc = VERSION_CONTENT.format(major, minor, patch, postfix)

    with open(VERSION_FILE, 'w') as f:
        f.write(version_inc)


def update(major, minor, patch, postfix=""):
    set_version(major, minor, patch, postfix)


def main(argv):
    major, minor, patch, postfix = get_version()

    if argv.major:
        major += 1
        minor = 0
        patch = 0
        postfix = ""
        update(major, minor, patch)
    if argv.minor:
        minor += 1
        patch = 0
        postfix = ""
        update(major, minor, patch)
    if argv.patch:
        patch += 1
        postfix = ""
        update(major, minor, patch)
    if argv.postfix and len(argv.postfix) > 0:
        postfix = argv.postfix
        update(major, minor, patch, postfix)

    if len(postfix) == 0:
        print('v{}.{}.{}'.format(major, minor, patch))
    else:
        print('v{}.{}.{}-{}'.format(major, minor, patch, postfix))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Set or read current version.')
    argparser.add_argument('--major', action='store_true',
            help='bumps major version')
    argparser.add_argument('--minor', action='store_true',
            help='bumps minor version')
    argparser.add_argument('--patch', action='store_true',
            help='bumps patch')
    argparser.add_argument('--postfix', type=str,
            help='set postfix')
    argv = argparser.parse_args()
    main(argv)

