# Releasing a new version

To release a new version, `deploy.py` is used.

### Prerequisites

`deploy.py` should be run from the repo, directly cloned from
`LeelaChessZero/lc0` (not fork), and upstreams from `master` and `release`
branches should be configured to point to that repo.

It's recommended to have a separate repository cloned which you will use
exclusively for releases, and won't do other development on that repo.

To push version changes, scripts needs write access to this repo.


### General script information

There are three pipelines implemented in this script:
- `new_minor_release`
- `cherrypick`
- `new_patch_release`

Every pipeline consists of series of steps.

To start pipeline, run it like this:
```bash
$ ./deploy.py new_patch_release
```

To just get a list of steps in a pipeline, do that:
```bash
$ ./deploy.py --list new_patch_release
```

To start pipeline from particular step (5 in this example), do that:
```bash
$ ./deploy.py --start 5 new_patch_release
```

To make pipeline ask confirmation before executing every step, run like this:
```bash
$ ./deploy.py --steps new_patch_release
```

The state of the pipeline is stored in file `/tmp/<pipeline-name>_state.json`, 
so if pipeline crashes (or you Ctrl+C it), when you restart pipeline with
the same parameters, it will offer you to continue from where you stopped.

That's the recommended way of handling stuff when something is wrong:
Ctrl+C, then fix stuff manually, then restart and continue.


## Pipelines

### New minor/major release

When issuing new minor/major release, entire contents of `master` branch is
merged into `release` branch, new minor version change happens, and `-rc1`
suffix is added.

The version in `master` branch is after that updated to a next minor version,
with `dev` suffix added.

Examples:

- `v0.13.1 -> v0.14-rc1`, master branch becomes `v0.15-dev`
- `v0.13.1 -> v1.0-rc1`, master branch becomes `v1.1-dev`

To do that, just run:
```bash
$ ./deploy new_minor_release
```

(in one of the steps it will ask whether you prefer minor or major verion update)

You will also be prompted to edit changelog.txt



### New patch release

During patch release, contents of `master` is NOT merged into `release`.
Instead, you run `cherrypick` pipeline to pick individual changes to merge into.

During patch release, only patch version is updated. Examples:
- `v0.13-rc1` -> `v0.13-rc2`
- `v0.13-rc1` -> `v0.13.0`
- `v0.13.4` -> `v0.13.5`


#### Typical workflow:

##### 1. Cherrypick changes

 It's safe to run multiple times. Also cherrypicking of multiple commits within
 one pipeline run is supported.

```bash
$ ./deploy.py cherrypick
```
The cherrypick pipeline commits cherrypicks, but does not push them (as
patch release pipeline will do that).

Sometimes (e.g. latest RC was bugless and you just want to promote it
to .0 version) you don't have to run it at all.

##### 2. Flip patch version

```bash
$ ./deploy new_patch_release
```
In one of the steps it will ask whether you want to bump RC number or patch.
You will also be prompted to edit changelog.txt.




## Example runs:

In terminal it's all colorful and nicer.

New minor release:

```
$ ./deploy.py new_minor_release

[ 1/23] Store SHA of /home/crem/dev/test/scripts/deploy.py...
SHA256 of /home/crem/dev/test/scripts/deploy.py is c04e3279dedcc9b2a5526a1eca3914e6a6529c266507546a42e0f0a135825883

[ 2/23] Check that git doesn't have uncommited changes...
$ git diff-index --quiet HEAD --

[ 3/23] Fetch fresh version from upstream...
$ git fetch

[ 4/23] Checkout master...
$ git checkout master
Already on 'master'
Your branch is up to date with 'origin/master'.

[ 5/23] Compare SHA of /home/crem/dev/test/scripts/deploy.py...
SHA256 of /home/crem/dev/test/scripts/deploy.py is c04e3279dedcc9b2a5526a1eca3914e6a6529c266507546a42e0f0a135825883

[ 6/23] Checkout release...
$ git checkout release
Switched to branch 'release'
Your branch is up to date with 'origin/release'.

[ 7/23] Read current version from version.inc file...
Found current version v1.3.1

[ 8/23] Select new major/minor version....
Current version is v1.3.1. What will be the new one?
  1. 1.4-rc1
  2. 2.0-rc1
>>>>>>> 1

[ 9/23] Merge master into release...
$ git merge master
Already up to date.

[10/23] Store git log for appending to changelog.txt...

[11/23] Prepend git log to a changelog...

[12/23] Store SHA of /home/crem/dev/test/changelog.txt...
SHA256 of /home/crem/dev/test/changelog.txt is 7e77a949b543dce2fa324d047599ab0927107bc9a92b6077a8b0f52ff3f55d37

[13/23] Edit changelog...
$ vim /home/crem/dev/test/changelog.txt

[14/23] Compare SHA of /home/crem/dev/test/changelog.txt...
SHA256 of /home/crem/dev/test/changelog.txt is 5cb0aab523f19b7387a362a145e29983386399b50a7cffcdad1dbf8b981c736d

[15/23] Execution of git add /home/crem/dev/test/changelog.txt ...
$ git add /home/crem/dev/test/changelog.txt

[16/23] Write version into version.inc, commit and tag....
$ git add /home/crem/dev/test/src/version.inc
$ git commit -m "Change version.inc to v1.5-dev."
[release 70e12c6] Change version.inc to v1.5-dev.
 2 files changed, 8 insertions(+), 3 deletions(-)

[17/23] Checkout release...
$ git checkout master
Switched to branch 'master'
Your branch is up to date with 'origin/master'.

[18/23] Merge release into master...
$ git merge release
Updating 232136d..70e12c6
Fast-forward
 changelog.txt   | 25 +++++++++++++++++++++++++
 src/version.inc |  2 +-
 2 files changed, 26 insertions(+), 1 deletion(-)

[19/23] Checkout release...
$ git checkout release
Switched to branch 'release'
Your branch is ahead of 'origin/release' by 1 commit.
  (use "git push" to publish your local commits)

[20/23] Write version into version.inc, commit and tag....
$ git add /home/crem/dev/test/src/version.inc
$ git commit -m "Change version.inc to v1.4-rc1."
[release 2b515ea] Change version.inc to v1.4-rc1.
 1 file changed, 2 insertions(+), 2 deletions(-)
$ git tag -a v1.4-rc1 -m "Adding tag v1.4-rc1"

[21/23] Checkout master...
$ git checkout master
Switched to branch 'master'
Your branch is ahead of 'origin/master' by 6 commits.
  (use "git push" to publish your local commits)

[22/23] Push release changes to github...
$ git push --all origin
Enumerating objects: 13, done.
Counting objects: 100% (13/13), done.
Delta compression using up to 4 threads.
Compressing objects: 100% (7/7), done.
Writing objects: 100% (9/9), 776 bytes | 776.00 KiB/s, done.
Total 9 (delta 4), reused 0 (delta 0)
To bitbucket.org:mooskagh/test.git
   232136d..70e12c6  master -> master
   5af7fba..2b515ea  release -> release

[23/23] Push tags to github...
$ git push --tags origin
Enumerating objects: 1, done.
Counting objects: 100% (1/1), done.
Writing objects: 100% (1/1), 171 bytes | 171.00 KiB/s, done.
Total 1 (delta 0), reused 0 (delta 0)
To bitbucket.org:mooskagh/test.git
 * [new tag]         v1.4-rc1 -> v1.4-rc1

The pipeline has finished.
```

Cherrypick:
```
$ ./deploy.py cherrypick

[ 1/10] Store SHA of /home/crem/dev/test/scripts/deploy.py...
SHA256 of /home/crem/dev/test/scripts/deploy.py is c1835b71d8555a830c919e32933bd34fb83e4df3ccbc0f079aaf921ec2238a13

[ 2/10] Check that git doesn't have uncommited changes...
$ git diff-index --quiet HEAD --

[ 3/10] Fetch fresh version from upstream...
$ git fetch

[ 4/10] Checkout master...
$ git checkout master
Already on 'master'
Your branch is ahead of 'origin/master' by 1 commit.
  (use "git push" to publish your local commits)

[ 5/10] Compare SHA of /home/crem/dev/test/scripts/deploy.py...
SHA256 of /home/crem/dev/test/scripts/deploy.py is c1835b71d8555a830c919e32933bd34fb83e4df3ccbc0f079aaf921ec2238a13

[ 6/10] Checkout release...
$ git checkout release
Switched to branch 'release'
Your branch is up to date with 'origin/release'.

[ 7/10] Choose which commit to cherrypick...
Select commit to cherrypick.
  0. [Select 0 to finish.]
  1. 2d880f8 Bugfix!
  2. 8c351a1 New untested feature
>>>>>>> 1

[ 8/10] Cherry-pick a commit...
$ git cherry-pick 2d880f8
[release 69e45fd] Added test.txt
 Date: Sun Jul 22 12:49:58 2018 +0200
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 scripts/e.txt

[ 9/10] Go 2 steps backwards...
[ JMP ] Jumping to 7 (Choose which commit to cherrypick)

[ 7/10] Choose which commit to cherrypick...
Select commit to cherrypick.
  0. [Select 0 to finish.]
  1. 2d880f8 Bugfix!
  2. 8c351a1 New untested feature
>>>>>>> 0
[ JMP ] Jumping to 10 (Checkout master)

[10/10] Checkout master...
$ git checkout master
Switched to branch 'master'
Your branch is ahead of 'origin/master' by 1 commit.
  (use "git push" to publish your local commits)
The pipeline has finished.
```


New patch release:

```
$ ./deploy.py new_patch_release

[ 1/18] Store SHA of /home/crem/dev/test/scripts/deploy.py...
SHA256 of /home/crem/dev/test/scripts/deploy.py is c1835b71d8555a830c919e32933bd34fb83e4df3ccbc0f079aaf921ec2238a13

[ 2/18] Check that git doesn't have uncommited changes...
$ git diff-index --quiet HEAD --

[ 3/18] Fetch fresh version from upstream...
$ git fetch

[ 4/18] Checkout master...
$ git checkout master
Already on 'master'
Your branch is ahead of 'origin/master' by 1 commit.
  (use "git push" to publish your local commits)

[ 5/18] Compare SHA of /home/crem/dev/test/scripts/deploy.py...
SHA256 of /home/crem/dev/test/scripts/deploy.py is c1835b71d8555a830c919e32933bd34fb83e4df3ccbc0f079aaf921ec2238a13

[ 6/18] Checkout release...
$ git checkout release
Switched to branch 'release'
Your branch is up to date with 'origin/release'.

[ 7/18] Read current version from version.inc file...
Found current version v1.4-rc1

[ 8/18] Select new patch version....
Current version is v1.4-rc1. What will be the new one?
  1. 1.4-rc2
  2. 1.4.0
>>>>>>> 2

[ 9/18] Store git log for appending to changelog.txt...

[10/18] Prepend git log to a changelog...

[11/18] Store SHA of /home/crem/dev/test/changelog.txt...
SHA256 of /home/crem/dev/test/changelog.txt is 9f1ed218e3e443412266aa09c9504a0a36039de0e5e8d6e925b11cf909412c8c

[12/18] Edit changelog...
$ vim /home/crem/dev/test/changelog.txt

[13/18] Compare SHA of /home/crem/dev/test/changelog.txt...
SHA256 of /home/crem/dev/test/changelog.txt is 555a4e9899d627f49e161902f538b6ea6d4c389ea29c979e727b38f740c83dac

[14/18] Execution of git add /home/crem/dev/test/changelog.txt ...
$ git add /home/crem/dev/test/changelog.txt

[15/18] Write version into version.inc, commit and tag....
$ git add /home/crem/dev/test/src/version.inc
$ git commit -m "Change version.inc to v1.4.0."
[release 30b714a] Change version.inc to v1.4.0.
 2 files changed, 7 insertions(+), 2 deletions(-)
$ git tag -a v1.4.0 -m "Adding tag v1.4.0"

[16/18] Push release changes to github...
$ git push --all origin
Enumerating objects: 21, done.
Counting objects: 100% (18/18), done.
Delta compression using up to 4 threads.
Compressing objects: 100% (8/8), done.
Writing objects: 100% (9/9), 832 bytes | 832.00 KiB/s, done.
Total 9 (delta 4), reused 0 (delta 0)
To bitbucket.org:mooskagh/test.git
   70e12c6..8c351a1  master -> master
   2b515ea..30b714a  release -> release

[17/18] Push tags to github...
$ git push --tags origin
Enumerating objects: 1, done.
Counting objects: 100% (1/1), done.
Writing objects: 100% (1/1), 169 bytes | 169.00 KiB/s, done.
Total 1 (delta 0), reused 0 (delta 0)
To bitbucket.org:mooskagh/test.git
 * [new tag]         v1.4.0 -> v1.4.0

[18/18] Checkout master...
$ git checkout master
Switched to branch 'master'
Your branch is up to date with 'origin/master'.
The pipeline has finished.
```