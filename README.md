# faktaark

## Setup
Install `uv`: https://docs.astral.sh/uv/getting-started/installation/

```bash
git init
uv sync --dev
git add .
git commit -m "Initial commit"
uv run pre-commit install # optional
```


### Run
To execute your software you have two options:

**Option 1: Direct execution**
```bash
uv run main.py
```

**Option 2: Run as installed package**
```bash
uvx --from . faktaark
```

### Development
Just run `uv run main.py` and you are good to go!

### Update from template
To update your project with the latest changes from the template, run:
```bash
uvx --with copier-template-extensions copier update --trust
```

You can keep your previous answers by using:
```bash
uvx --with copier-template-extensions copier update --trust --defaults
```

### (Optional) pre-commit
pre-commit is a set of tools that help you ensure code quality. It runs every time you make a commit.

First, install pre-commit:
```bash
uv tool install pre-commit
```

Then install pre-commit hooks:
```bash
pre-commit install
```

To run pre-commit on all files:
```bash
pre-commit run --all-files
```

### How to install a package
Run `uv add <package-name>` to install a package. For example:
```bash
uv add requests
```

#### Visual studio code
If you are using visual studio code install the recommended extensions


### Tools installed
- uv
- pre-commit (optional)

#### What is an environment variable? and why should I use them?
Environment variables are variables that are not populated in your code but rather in the environment
that you are running your code. This is extremely useful mainly for two reasons:
- security, you can share your code without sharing your passwords/credentials
- portability, you can avoid using hard-coded values like file-system paths or folder names

you can place your environment variables in a file called `.env`, the `main.py` will read from it. Remember to:
- NEVER commit your `.env`
- Keep a `.env.example` file updated with the variables that the software expects
