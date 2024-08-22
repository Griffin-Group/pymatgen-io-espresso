# Contributing to `pymatgen.io.espresso`

First off, thanks for taking the time to contribute!

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier me as the maintainer, and smooth out the experience for all involved. I look forward to your contributions.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Your First Code Contribution](#your-first-code-contribution)
- [Improving The Documentation](#improving-the-documentation)
- [Styleguides](#styleguides)
- [General Guidelines](#general-guidelines)
- [Editor Config and Tools](#editor-config-and-tools)


## Code of Conduct

This project and everyone participating in it is governed by the
`pymatgen.io.espresso` [Code of Conduct](blob/master/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Please report unacceptable behavior
to @oashour.


## I Have a Question

> If you want to ask a question, I assume that you have read the available [Documentation]().

Before you ask a question, it is best to search for existing [Issues](/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue. It is also advisable to search the internet for answers first, issues with `python` itself do not belong here.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (`python`, `pymatgen`, etc.), depending on what seems relevant.

I will then address the issue as soon as possible.

## I Want To Contribute

> ### Legal Notice 
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project license.

### Reporting Bugs

I appreciate bug reports and want to fix them as soon as possible. But we also need your help to identify and fix them. Please follow the steps below to report a bug.

#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more information. Therefore, I ask you to investigate carefully, collect information and describe the issue in detail in your report. Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest release.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions (Make sure that you have read the [documentation]().) 
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](issues?q=label%3Abug).
- Also make sure to search the internet (including Stack Overflow) to see if users outside of the GitHub community have discussed the issue.
- Collect information about the bug:
- Stack trace (Traceback)
- OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
- Version of `python`, `pymatgen`, and Quantum ESPRESSO, you're using (e.g., `conda` or `pip`), depending on what seems relevant.
- Possibly your input and the output
- Can you reliably reproduce the issue? And can you also reproduce it with older versions?


#### How Do I Submit a Good Bug Report?

`pymatgen.io.espresso` uses GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [Issue](/issues/new). (Since we can't be sure at this point whether it is a bug or not, we ask you not to talk about a bug yet and not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the *reproduction steps* that someone else can follow to recreate the issue on their own. This usually includes your code. **For good bug reports you should isolate the problem and create a reduced test case, a minimum working example (MWE).**
- Provide the information you collected in the previous section.

Once it's filed:

- I will label the issue as appropriate (e.g., `bug`).
- I will try to reproduce the issue with your provided steps. If there are no reproduction steps or no obvious way to reproduce the issue, we will ask you for those steps and mark the issue as `needs-repro`. Bugs with the `needs-repro` tag will not be addressed until they are reproduced.
- If I am able to reproduce the issue, it will be marked `needs-fix`, as well as possibly other tags (such as `critical`), and the issue will be left to be [implemented by someone](#your-first-code-contribution).

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for CONTRIBUTING.md, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.


#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation]() carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an independent package that uses `pymatgen` and `pymatgen.io.espresso` as dependencies.

#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why. At this point you can also tell which alternatives do not work for you.
- You may want to **include screenshots and animated GIFs** which help you demonstrate the steps or point out the part which the suggestion is related to. You can use [this tool](https://www.cockos.com/licecap/) to record GIFs on macOS and Windows, and [this tool](https://github.com/colinkeenan/silentcast) or [this tool](https://github.com/GNOME/byzanz) on Linux. 
- **Explain why this enhancement would be useful** to most `pymatgen.io.espresso` users. You may also want to point out the other projects that solved it better and which could serve as inspiration.


### Improving The Documentation
You are welcome to improve the documentation, and add examples if they seem relevant. If you are planning large changes, please open an issue first to discuss it with me.

## Styleguides
Naturally, [PEP 8](https://www.python.org/dev/peps/pep-0008/) is mandatory. This is enforced via pre-commit hooks that run `ruff` to lint and format the code. Following the [Google Python Style Guide](https://peps.python.org/pep-0008/) is also strongly recommended. If in doubt, follow the style of pre-existing code or ask the maintainer. Pay special attention to variable naming in the style guides, and please maintain consistency with the existing codebase.

People can have strong opinions about code style, so please be open to feedback and be willing to make changes. When contributing to open source projects, it is important to follow the style guide of the project you are contributing to and the maintainers' suggestions. This makes it easier for the maintainer to review your code and for others to understand it, and it helps maintain a consistent code style throughout the project.

### General Guidelines

Writing clear, well-maintained, well-documented scientific code is grueling, incredibly unrewarding, and in most cases does not further your scienific career. Making this code base public and maintaining it takes time away from my research. Please remember that the maintainer is human, and is doing this in his free time. I am not paid to maintain this project, I am not obligated to help you. I am mainly doing this because I enjoy it, and because maintaining good code is in the best interest of the scientific community as a whole. Please be respectful and patient, and keep an open mind when I make suggestions about your code.

* Before contributing any code, **please open an issue first to discuss it with the maintainer**. This way, I can help guide you in the right direction, prevent you from wasting time on something that might not get accepted, and help you do it in a way that fits with this project's goals and philosophy. 
* Please use clear and descriptive commit messages. Use standard practices, e.g., "Fix typo in README.md" or "Add tests for new feature". If your commit fixes a bug or resolves an issue, please include the issue number in the body. Note "Fix" and "Add," not "Fixed" or "Adding." This is a good post on [how to write a good commit message](https://chris.beams.io/posts/git-commit/). 
* Commit in small, logical chunks. This makes it easier to review and understand your changes. 
* If you have multiple changes, please split them into separate commits, and do not commit incomplete work or code that doesn't run. 
* Avoid unnecessary commits like "Reformatted code." Please squash these commits before opening a pull request. 
* Do not commit commented out code unless it's a line or two.
* If you have implemented any new features or changes, please add tests for them. I will not accept any new features or changes without tests to ensure a healthy codebase.
* If you have implemented any new features or changes, please update the documentation and write the necessary docstrings. I will not accept any new features or changes without documentation.
* Add the appropriate amount of comments. Read the rest of the code base to understand what the maintainer considers to be "appropriate." You don't need to explain every single thing you're doing. If you need more than 1-2 lines to explain, e.g., some oddity in QE's file formats, it might be better suited for the doc string or the documentation.

I will soon implement additional pre-commit hooks to enforce some of these rules. 

### Editor Config and Tools

I recommend installation in editable mode in a virtual environment, e.g., `pip install --editable '.[dev,docs]'`. This will also install the necessary development dependencies.

You are free to use any editor you like, but VSCode with the appropriate python and ruff extensions is recommended. This will catch many issues that may be missed by the pre-commit hooks and will often alert you to improvements you can make. I personally like [Sourcercy](https://sourcery.ai/), as well, and it is free if you are still a student. Sourcery will often catch improvements that I will ask for anyway in code reviews.

## Attribution
This guide is based on the **contributing.md**. [Make your own](https://contributing.md/)!
