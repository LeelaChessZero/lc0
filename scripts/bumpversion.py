#!/usr/bin/env python3

import argparse
import tempfile
import textwrap
import os
from string import Template
from subprocess import call

VERSION_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../src/version.inc")
VERSION_CONTENT = """
#define LC0_VERSION_MAJOR $major 
#define LC0_VERSION_MINOR $minor
#define LC0_VERSION_PATCH $patch
#define LC0_VERSION_POSTFIX $postfix
"""
VERSION_CONTENT = textwrap.dedent(VERSION_CONTENT).strip()
EDITOR = os.environ.get("EDITOR", "vim")


def get_current_version():
    with open(VERSION_FILE, 'r') as f:
        major = int(f.readline().split()[2])
        minor = int(f.readline().split()[2])
        patch = int(f.readline().split()[2])
        postfix = f.readline().split()[2]

    return major, minor, patch, postfix


def set_current_version(major, minor, patch, postfix="\"\""):
    tmp = Template(VERSION_CONTENT)
    version_inc = tmp.substitute(major=major, minor=minor, patch=patch, postfix=postfix)

    with open(VERSION_FILE, 'w') as f:
        f.write(version_inc)


def tag(major, minor, patch):
    version = "v{}.{}.{}".format(major, minor, patch)
    call(["git", "-m", version, version])


def editor(init_msg):
    with tempfile.NamedTemporaryFile() as f:
        f.write(init_msg.encode('utf-8'))
        f.flush()
        call([EDITOR, f.name])
        f.seek(0)
        return f.read().decode('utf-8')


def update(major, minor, patch):
    set_current_version(major, minor, patch)
    # edit changelot
    # git commit
    # git tag


def main(argv):
    major, minor, patch, postfix = get_current_version()

    if argv.major:
        major += 1
        minor = 0
        patch = 0
        postfix = ""
        update(major, minor, patch)
    elif argv.minor:
        minor += 1
        patch = 0
        postfix = ""
        update(major, minor, patch)
    elif argv.patch:
        patch += 1
        postfix = ""
        update(major, minor, patch)

    print(editor("# type message here"))

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

