# Contributing to CeRULEo

When contributing to CeRULEo, make sure that the changes you wish to make are in line with the project direction. If you are not sure about this, open an issue first, so we can discuss it.

We use github to host code, to track issues and feature requests, as well as accept pull requests.

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

* Reporting a bug
* Discussing the current state of the code
* Submitting a fix
* Proposing new features



# Report bugs 
We use GitHub issues to track public bugs. Report a bug by opening a new issue; it's that easy!

Write bug reports with detail, background, and sample code

Great Bug Reports tend to have:

1. A quick summary and/or background
2. Steps to reproduce
3. Be specific!
4. Give sample code if you can.
4. What you expected would happen
5. What actually happens
6. Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)


# Code submission policy
 
Pull requests are the best way to propose changes to the codebase (we use Github Flow). We actively welcome your pull requests:

1. Fork the repo and create your branch from main.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!


## Rules for submission policy

* Conform to the project coding style: We use black as the code formatter and isort for sorting the imports.
* Choose expressive variable, function and class names. Make it as obvious as possible what the code is doing.
* Split your changes into separate, atomic commits (i.e. A commit per feature or fix, where the build, tests and the system are all functioning).
* Wrap your commit messages at 72 characters.
* The first line of the commit message is the subject line, and must have the format "Category: Brief description of what's being changed". The 
* Write the commit message subject line in the imperative mood ("Foo: Change the way dates work", not "Foo: Changed the way dates work").
* Squash your commits when making revisions after a patch review.
* Add your personal copyright line to files when making substantive changes. (Optional but encouraged!)
* Check the spelling of your code, comments and commit messages.


## Versioning

We are using [Semantic Versioning](https://semver.org/) for keep tracking the development. When the Pull Request is ready, create a new commit for bumping the version following the rules of SymVer.

```bash
bump2version --tag --commit --allow-dirty --commit-args="-a" patch
```

# License

Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same MIT License that covers the project. Feel free to contact the maintainers if that's a concern.

