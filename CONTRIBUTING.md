# Contributing to lc0

There are several project guidelines we prefer that contributors stick to.

This codebase uses Semantic Versioning. This codebase uses Semantic Versioning. A release is the final commit for that version number, and all subsequent commits are development for the next version. Updating the version (i.e. making a release) should be done only with the automated tool for it, `scripts/bumpversion.py`. Any commit that changes the version must consist solely of that version change; any commit which changes anything but the version cannot also change the version.

`master` is the default branch, and the active development branch. `release` will always point to the latest tag. `master` (when it differs from `release`, i.e. almost always) will have a version which includes a postfix attached to the patch section of the version. `release` does not have a postfix attached to the version.

### Style

The most obvious set of guidelines is code style. This codebase strictly complies with the Google C++ style guide, as [enforced by Clang](https://clang.llvm.org/docs/ClangFormatStyleOptions.html).

The guide does not specify a pointer/reference type declaration style, other than that it must be consistent; this codebase uses left-aligned pointer and reference types, i.e., attached to the type, not the variable name (and not dangling either).

Comments must be full sentences, i.e. capitalized and ending in a period. (Sentences with elided subjects are fine.)

Braces are K&R style. Single-statement `if` statements must be on one line; all others must use fully expanded braces.

Indentation is two spaces; \t characters are disallowed.

Code line length is strictly capped at 80 characters.

### Git history

We require Pull Requests to be merged by squashing. This means all commits will be merged into one commit applied onto master, so PRs should stick to *one* topic only. If you think changes deserve separate commits, make separate PRs for each commit.
