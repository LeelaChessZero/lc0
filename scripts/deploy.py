#!/usr/bin/env python3

import os
import os.path
import json
import click
import hashlib
import traceback
import sys
import subprocess
import re
import ast

OWN_FILENAME = os.path.realpath(__file__)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(OWN_FILENAME), '..'))
VERSION_PATH = os.path.join(REPO_ROOT, 'src/version.inc')
CHANGELOG_PATH = os.path.join(REPO_ROOT, 'changelog.txt')


# Root command line parsing object
@click.group()
@click.option(
    '--start', '-s', default=1, type=int, help='Start with this step.')
@click.option('--list', '-l', is_flag=True, help='Only show list of steps.')
@click.option(
    '--confirm', is_flag=True, help='Ask confirmation before every step.')
@click.pass_context
def cli(ctx, start, list, confirm):
    p = ctx.obj['pipeline']
    p.start = start
    p.list_only = list
    p.step_by_step = confirm


############################################################################
# Commands.
############################################################################
@cli.command()
@click.pass_context
def new_minor_release(ctx):
    p = ctx.obj['pipeline']

    # version.inc update strategy:
    # 1. Merge everything from master to release (version file is not changed
    #    because it was already merged at step 3 of previous iteration)
    # 2. Change version.inc in release branch to new "dev"
    # 3. Merge release to master (so that master has new "dev" too)
    # 4. Change version.inc in release branch to new release version.

    # Then do the push, so all that merging back and forth is kind of atomic
    # from github point of view).

    # Without that (e.g. modifying version.inc independently rather than having
    # a sequence), all merges will have a collision in version.inc and that's
    # painful to resolve.

    CheckOwnIntegrityAndFetch(p)
    p.AddStep(RunCmdStep('git checkout release', doc='Checkout release'))
    p.AddStep(ReadCurrentVerionFromFile)
    p.AddStep(GetNextVersion)
    p.AddStep(RunCmdStep('git merge master', doc='Merge master into release'))
    UpdateChangeLog(p)
    p.AddStep(WriteAndCommitVersion('dev-version', add_tag=False))
    p.AddStep(RunCmdStep('git checkout master', doc='Checkout master'))
    p.AddStep(RunCmdStep('git merge release', doc='Merge release into master'))
    p.AddStep(RunCmdStep('git checkout release', doc='Checkout release'))
    p.AddStep(WriteAndCommitVersion('new-version', add_tag=True))
    p.AddStep(RunCmdStep('git checkout master', doc='Checkout master'))
    p.AddStep(
        RunCmdStep(
            'git push --all origin', doc='Push release changes to github'))
    p.AddStep(RunCmdStep('git push --tags origin', doc='Push tags to github'))
    p.Run('new-release')


@cli.command()
@click.pass_context
def cherrypick(ctx):
    p = ctx.obj['pipeline']

    CheckOwnIntegrityAndFetch(p)
    p.AddStep(RunCmdStep('git checkout release', doc='Checkout release'))
    p.AddStep(SelectCommitToCherrypick(jump_on_end=3))
    p.AddStep(CherryPick)
    p.AddStep(RunJump(-2))
    p.AddStep(RunCmdStep('git checkout master', doc='Checkout master'))
    p.Run('cherrypick')


@cli.command()
@click.pass_context
def new_patch_release(ctx):
    p = ctx.obj['pipeline']
    CheckOwnIntegrityAndFetch(p)
    p.AddStep(RunCmdStep('git checkout release', doc='Checkout release'))
    p.AddStep(ReadCurrentVerionFromFile)
    p.AddStep(GetNextPatchVersion)
    UpdateChangeLog(p)
    p.AddStep(WriteAndCommitVersion('new-version', add_tag=True))
    p.AddStep(
        RunCmdStep(
            'git push --all origin', doc='Push release changes to github'))
    p.AddStep(RunCmdStep('git push --tags origin', doc='Push tags to github'))
    p.AddStep(RunCmdStep('git checkout master', doc='Checkout master'))
    p.Run('new-patch')


def CheckOwnIntegrityAndFetch(p):
    p.AddStep(RunStoreFileSha(OWN_FILENAME))
    p.AddStep(
        RunCmdStep(
            'git diff-index --quiet HEAD --',
            doc="Check that git doesn't have uncommited changes"))
    p.AddStep(RunCmdStep('git checkout release', doc='Checkout release'))
    p.AddStep(RunCmdStep('git pull', doc='Pull release from upstream'))
    p.AddStep(RunCmdStep('git checkout master', doc='Checkout master'))
    p.AddStep(RunCmdStep('git pull', doc='Pull master from upstream'))
    p.AddStep(CompareFileSha(OWN_FILENAME, require_equal=True))


def UpdateChangeLog(p):
    p.AddStep(SaveGitLog)
    p.AddStep(PrependGitLogToChangeLog)
    p.AddStep(RunStoreFileSha(CHANGELOG_PATH))
    p.AddStep(
        RunCmdStep(
            '{0} {1}'.format(os.getenv('EDITOR', 'vim'), CHANGELOG_PATH),
            doc='Edit changelog'))
    p.AddStep(CompareFileSha(CHANGELOG_PATH, require_equal=False))
    p.AddStep(RunCmdStep(f'git add {CHANGELOG_PATH}'))


############################################################################
# Pipeline machinery
############################################################################


def RunJump(offset):
    def f(ctx):
        raise Jump(offset)

    if offset > 0:
        f.__doc__ = f'Go {offset} steps forward'
    else:
        f.__doc__ = f'Go {-offset} steps backwards'
    return f


def FormatVersion(version_dict):
    res = '{MAJOR}.{MINOR}.{PATCH}'.format(*version_dict)
    if version_dict['POSTFIX']:
        res += '-{0}'.format(version_dict['POSTFIX'])
    return res


def SelectionPrompt(caption, choices, *, zero_choice=None):
    while True:
        click.secho(caption)
        if zero_choice:
            click.secho(f'  0. {zero_choice}', fg='yellow')
        for i, n in enumerate(choices):
            click.secho(f'{i+1:3}. {n[0]}', fg='yellow')
        r = click.prompt('', prompt_suffix='>>>>>>> ')
        try:
            r = int(r) - 1
            if zero_choice and r == -1:
                return None
            if 0 <= r < len(choices):
                return choices[r][1]
        except ValueError:
            pass


def SelectCommitToCherrypick(jump_on_end):
    def f(ctx):
        proc = subprocess.Popen(
            ['git', 'log', 'HEAD..master', '--pretty=format:%h %s'],
            stdout=subprocess.PIPE)
        commits = proc.communicate()[0].decode('utf-8').split('\n')
        if proc.returncode != 0:
            click.secho('Git returned non-zero exit code!', fg='red')
            return False
        ctx['cherrypick-commit'] = SelectionPrompt(
            'Select commit to cherrypick.', [(x, x.split()[0])
                                             for x in commits],
            zero_choice='[Select 0 to finish.]')
        if ctx['cherrypick-commit'] is None:
            raise Jump(jump_on_end)
        return True

    f.__doc__ = 'Choose which commit to cherrypick'
    return f


def CherryPick(ctx):
    """Cherry-pick a commit"""
    return RunCmdStep(f'git cherry-pick {ctx["cherrypick-commit"]}')(ctx)


def WriteVersionToFile(version):
    with open(VERSION_PATH, 'w') as f:
        f.write('// This file is automatically generated by deploy.py. '
                'Do not edit this file.\n')
        for k, v in version.items():
            # Value may be string or number. JSON formatting of those data
            # types is conveniently the same as in C++, so outputting JSON.
            f.write(f'#define LC0_VERSION_{k} {json.dumps(v)}\n')
    return True


def WriteAndCommitVersion(version_key, *, add_tag):
    def f(ctx):
        v = ctx[version_key]
        v_str = FormatVersion(v)
        WriteVersionToFile(v)
        if not RunCmdStep('git add ' + VERSION_PATH)(ctx):
            return False
        if not RunCmdStep(f'git commit -m "Change version.inc to v{v_str}."')(
                ctx):
            return False
        if add_tag:
            if not RunCmdStep(f'git tag -a v{v_str} -m "Adding tag v{v_str}"')(
                    ctx):
                return False
        return True

    f.__doc__ = 'Write version into version.inc, commit and tag.'
    return f


def PrependGitLogToChangeLog(ctx):
    """Prepend git log to a changelog"""
    with open(CHANGELOG_PATH, 'r') as f:
        contents = f.read()

    contents = 'v%s\n%s\n\nPLEASE EDIT THE FOLLOWING LOG!\n\n{0}\n'.format(
        FormatVersion(ctx['new-version']),
        '~' * (1 + len(FormatVersion(ctx['new-version']))),
        ctx['git-log'],
    ) + contents

    with open(CHANGELOG_PATH, 'w') as f:
        contents = f.write(contents)

    return True


def SaveGitLog(ctx):
    """Store git log for appending to changelog.txt"""
    proc = subprocess.Popen(
        ['git', 'log', f'v{FormatVersion(ctx["old-version"])}..HEAD'],
        stdout=subprocess.PIPE)
    ctx['git-log'] = proc.communicate()[0].decode('utf-8')
    if proc.returncode != 0:
        click.secho('Git returned non-zero exit code!', fg='red')
        return False

    return True


def GetNextVersion(ctx):
    """Select new major/minor version."""
    v = ctx['old-version']
    variants = [
        {
            'MAJOR': v['MAJOR'],
            'MINOR': v['MINOR'] + 1,
            'PATCH': 0,
            'POSTFIX': 'rc1',
        },
        {
            'MAJOR': v['MAJOR'] + 1,
            'MINOR': 0,
            'PATCH': 0,
            'POSTFIX': 'rc1',
        },
    ]
    ctx['new-version'] = SelectionPrompt(
        f'Current version is v{FormatVersion(v)}. What will be the new one?',
        [(FormatVersion(x), x) for x in variants])
    ctx['dev-version'] = {
        'MAJOR': ctx['new-version']['MAJOR'],
        'MINOR': ctx['new-version']['MINOR'] + 1,
        'PATCH': 0,
        'POSTFIX': 'dev',
    }
    return True


def GetNextPatchVersion(ctx):
    """Select new patch version."""
    v = ctx['old-version']
    m = re.search(r'rc(\d+)', v['POSTFIX'])
    if m:
        variants = [
            {
                'MAJOR': v['MAJOR'],
                'MINOR': v['MINOR'],
                'PATCH': v['PATCH'],
                'POSTFIX': f'rc{int(m.group(1)) + 1}',
            },
            {
                'MAJOR': v['MAJOR'],
                'MINOR': v['MINOR'],
                'PATCH': v['PATCH'],
                'POSTFIX': '',
            },
        ]
    else:
        variants = [
            {
                'MAJOR': v['MAJOR'],
                'MINOR': v['MINOR'],
                'PATCH': v['PATCH'] + 1,
                'POSTFIX': '',
            },
            {
                'MAJOR': v['MAJOR'],
                'MINOR': v['MINOR'],
                'PATCH': v['PATCH'] + 1,
                'POSTFIX': 'rc1',
            },
        ]
    ctx['new-version'] = SelectionPrompt(
        f'Current version is v{FormatVersion(v)}. What will be the new one?',
        [(FormatVersion(x), x) for x in variants])
    return True


def ReadCurrentVerionFromFile(ctx):
    """Read current version from version.inc file"""
    version = {}
    macro_re = re.compile(r'#define LC0_VERSION_(\S+) (.*)')
    with open(VERSION_PATH) as f:
        for line in f:
            m = macro_re.match(line)
            if m:
                version[m.group(1)] = ast.literal_eval(m.group(2))
    for x in ['MAJOR', 'MINOR', 'PATCH', 'POSTFIX']:
        if x not in version:
            click.secho(
                f'Unable to parse version from {VERSION_PATH}: missing {x}',
                fg='red')
            return None
    click.secho(
        f'Found current version v{FormatVersion(version)}', fg='yellow')
    ctx['old-version'] = version
    return True


def ComputeSha256(filename):
    BUF_SIZE = 65536
    sha = hashlib.sha256()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                return sha.hexdigest()
            sha.update(data)


def RunStoreFileSha(filename):
    def f(ctx):
        key = f'self-sha-{filename}'
        ctx[key] = ComputeSha256(filename)
        click.secho(f'SHA256 of {filename} is {ctx[key]}')
        return True

    f.__doc__ = f'Store SHA of {filename}'
    return f


def CompareFileSha(filename, *, require_equal):
    def f(ctx):
        sha = ComputeSha256(filename)
        click.secho(f'SHA256 of {filename} is {sha}')

        if require_equal and sha != ctx['self-sha-' + filename]:
            click.secho(f'File {filename} changed.', fg='red')
            return False
        if not require_equal and sha == ctx['self-sha-' + filename]:
            click.secho(f'File {filename} not changed.', fg='red')
            return False
        return True

    f.__doc__ = f'Compare SHA of {filename}'
    return f


# A step which runs exernal command.
def RunCmdStep(cmd_line, doc=None):
    def f(context):
        click.secho(f'$ {cmd_line}', fg='yellow')
        r = os.system(cmd_line)
        if r != 0:
            click.echo(
                f'Return code {r}, '
                f'The command was: {click.style(cmd_line, fg="yellow")}')
            return False
        return True

    if doc:
        f.__doc__ = doc
    else:
        f.__doc__ = f'Execution of {cmd_line} '
    return f


def RetryPrompt():
    while True:
        click.secho('(type one of: retry/ignore/abort)', fg='cyan')
        value = click.prompt('', prompt_suffix='>>>>>>> ')
        if value == 'retry':
            return True
        if value == 'ignore':
            return False
        if value == 'abort':
            sys.exit(1)


# Exception when step decides to jump to another step.
class Jump(BaseException):
    def __init__(self, offset):
        self.offset = offset


class Pipeline:
    def __init__(self):
        self.steps = []  # List of steps in a pipeline.
        self.start = 1  # From which step to start, 1-based.
        self.end = None  # At which step to end.
        self.list_only = False  # Only list steps, do not execute.
        self.step_by_step = False  # Ask confirmation before every step.
        self.context = {}
        self.cmd_name = 'unknown'

    def AddStep(self, func):
        self.steps.append(func)

    def StateFileName(self):
        return os.path.join('/tmp', f'lc0_{self.cmd_name}_state.json')

    def MaybeLoadState(self):
        if os.path.isfile(self.StateFileName()):
            if click.confirm('Forgotten state found. Restore?'):
                with open(self.StateFileName()) as f:
                    self.context = json.loads(f.read())
                    if 'chdir' in self.context:
                        os.chdir(self.context['chdir'])

    def StoreState(self):
        with open(self.StateFileName(), 'w') as f:
            f.write(json.dumps(self.context))

    def Run(self, cmd_name):
        self.cmd_name = cmd_name
        if self.list_only:
            for i, n in enumerate(self.steps):
                click.secho(f'{i+2:2}. {n.__doc__}')
            return
        click.clear()

        if self.end is None:
            end = len(self.steps)
        self.context['idx'] = self.start - 1

        self.MaybeLoadState()

        while self.start <= (self.context['idx'] + 1) <= end:
            self.StoreState()

            task_f = self.steps[self.context['idx']]
            try:
                click.echo(
                    click.style(
                        '[%2d/%2d]' %
                        (self.context['idx'] + 1, len(self.steps)),
                        fg='green') + f' {task_f.__doc__}...')
                if self.step_by_step:
                    if not click.confirm('Should I run it?'):
                        raise click.Abort
                if task_f(self.context):
                    self.context['idx'] += 1
                    continue
            except click.Abort:
                raise
            except Jump as jmp:
                self.context['idx'] += jmp.offset
                click.secho(
                    f'[ JMP ] Jumping to {self.context["idx"] + 1} '
                    f'({self.steps[self.context["idx"]].__doc__})',
                    fg='green',
                    bold=True)
                continue
            except:
                click.secho(traceback.format_exc(), fg='red')

            click.secho('[ FAIL ]', fg='red', bold=True)
            if not RetryPrompt():
                self.context['idx'] += 1
        click.secho('The pipeline has finished.', fg='green', bold=True)
        os.remove(self.StateFileName())


if __name__ == '__main__':
    p = Pipeline()
    cli(obj={'pipeline': p})