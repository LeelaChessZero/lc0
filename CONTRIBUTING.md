# Contributing to Leela Chess Zero (Lc0)

Last updated: June 2025

Thank you for your interest in contributing to LCZero! This document provides
guidelines for contributing to the codebase.

## Before you start

* All contributors are encouraged to join our Discord server at
  <https://lc0.org/chat>.
* Refer to [README.md](README.md) for building and running instructions.
  * The protobufs that are shared with the training code are located in a
    separate repository. Don't forget to run `git submodule update --init` to
    fetch them.
* Familiarize yourself with the developer documentation at
  <https://lczero.org/dev/>.

We use the [Meson](https://mesonbuild.com/) build system.

* In Linux, using `builddir/` is recommended for development
  (`meson setup builddir/`), as VSCode recognizes it and all the development and
  debugging tools work there (ask in Discord if you have issues).
  * In the `builddir/`, run `ninja lc0` (which is faster than just `ninja`). Run
    `ninja test` to run the tests.
* In Windows, `meson setup build/debug` generates a Visual Studio solution that
  can be used for development.
* Check `meson_options.txt` for the available build options (to use them, pass
  `-D<name>=<value>` to `meson setup`).

## Sending Pull Requests

* We use GitHub for managing contributions. Please fork the repository and
  create a new branch for your changes.
* It's encouraged to discuss your changes in the Discord server before
  starting work.
  * Small bug fixes are fine to submit without prior discussion.
  * Large changes that add code rather than modifying existing code (e.g. new
    backends, new search algorithms) are generally fine as well. Use your best
    judgement on whether your change may be controversial.
  * Changes that modify existing code (e.g. search algorithm tweaks, API
    changes) should be discussed first.

Changes that may affect playing strength **must** be tested.

* Unfortunately, we don't have a robust strength testing framework yet (working
  on it), so ask in the #testing-discuss channel on Discord for help with
  testing.
* Even for Elo-positive changes, we need to balance the strength and
  maintainability of the code. If your change is Elo-positive but makes the code
  more complex, please discuss it first. Recently, we added an option to
  clone the search algorithm in extreme cases.
* Elo-neutral simplifications are always welcome.

Pull Requests are squashed when merged. This means all commits in the branch
will be squashed into one commit applied onto master. This makes it tricky to
reuse the branch and continue to work on it after the PR is merged.

**Note:** This section only applies if you have dependent branches that were
built on top of your merged PR branch. If you only had one branch, you can
simply delete it after merging.

**Example scenario:** You had a branch `add-feature` that got merged, and you
have another branch `extend-feature` that was based on `add-feature`. After
`add-feature` is merged, you need to rebase `extend-feature` onto the new
master. Here's what to do after your PR is merged:

```shell
git fetch upstream  # Update your local master
git checkout extend-feature  # Switch to your dependent branch
git rebase --update-refs --onto upstream/master add-feature # Rebase onto the updated master
```

The `--update-refs` flag will also update any branches between your leaf branch
and the merged branch if you have any.

## C++ Standard and Libraries

* We use most C++20 features. However, supported compilers are GCC 10 and clang
  10, so some features may not be available.
* We use protocol buffers. However, we don't use any external library for it,
  but rather generate the code from `.proto` files using the script in
  `scripts/`.
* Since v0.32, we use Abseil (`absl::`).
* Use `CERR` for logging (goes to stderr and log), or `LOGFILE` (goes to log
  file only).
* Writing tests is encouraged. We use `gtest`/`gmock` for unit tests. Tests are
  located in the same directory as the code they test, in a file with the same
  name but ending with `_test.cc`.

## Style Guidelines

We follow the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with
these modifications:

* **Header guards**: Use `#pragma once` instead of traditional header guards.
* **Exceptions**: are allowed, but only one: `lczero::Exception`.
* **References vs Pointers**: Non-const reference parameters are neither
  encouraged nor discouraged over pointers (in the Google style guide, they used
  to be discouraged, and now they are encouraged).
* **Formatting**: Run `clang-format` on all code before committing.
* **RValue references**: Rvalue reference function parameters are allowed, not
  only for constructors and assignment operators. However, use them only if they
  benefit readability or have significant performance gain.
* Every new file should contain a GPLv3 banner with an additional exception
  under GNU GPL version 3 section 7 regarding NVIDIA libraries (see examples in
  existing files).
  * It's allowed to write new backends without the NVIDIA exception, but that
    means that we won't be able to include NVIDIA libraries if we link with that
    code.

## AI-Assisted Development

AI tools and coding assistants are allowed for contributing to Leela Chess Zero,
provided you follow these guidelines:

* **Disclose AI usage**: Clearly mention in your PR description if you used AI
  tools, LLMs, or agentic coding approaches (beyond simple code completion).
* **Maintain code ownership**: You must thoroughly read, understand, and review
  all AI-generated code in detail before submitting.
* **Ensure quality**: Take all appropriate steps to verify the code quality,
  correctness, and adherence to project standards.
* **Submit only what you could write yourself**: Use AI as a productivity
  booster, not as a replacement for proper programming skills and domain
  knowledge.
* **Core vs. auxiliary code**: While AI assistance works well for auxiliary code
  (website, documentation, simple tools), it has failed so far on core lc0
  engine code. We discourage using agentic coding for core engine components but
  welcome it for simpler projects within the Leela Chess Zero organization.

Remember: You are responsible for the quality and correctness of all code you
submit, regardless of how it was generated.

Thank you for helping make Leela Chess Zero better!
