# Contributing to NEON Tree Classification

Thank you for your interest in contributing! We welcome contributions from researchers at all technical levels.

## 🚀 Quick Start for Contributors

### Option 1: UV Setup (Recommended)
UV is faster and ensures consistent dependencies across all users.

```bash
# 1. Fork the repository on GitHub (click the Fork button)
# 2. Clone your fork
git clone https://github.com/yourusername/NeonTreeClassification.git
cd NeonTreeClassification

# 3. Install UV if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: pip install uv

# 4. Install the package
uv sync
```

### Option 2: Pip Setup (Alternative)
If you prefer traditional pip or can't install UV:

```bash
# After cloning (steps 1-2 above)
pip install -e .
```

> **Note**: UV is preferred because it ensures all contributors have the same dependency versions using the `uv.lock` file.

## 🤝 How to Contribute

### 1. Found a Bug?
- Open an [issue](https://github.com/Ritesh313/NeonTreeClassification/issues) 
- Describe what you expected vs what happened
- Include any error messages

### 2. Have a Feature Idea?
- Open an [issue](https://github.com/Ritesh313/NeonTreeClassification/issues) to discuss it first
- We can help figure out the best approach

### 3. Want to Contribute Code?
1. **Create a branch**: `git checkout -b your-feature-name`
2. **Make changes**: Edit files, add features, fix bugs
3. **Test it works**: `uv run python -c "import neon_tree_classification; print('Works!')"`
4. **Submit**: Create a Pull Request on GitHub

> **For advanced testing**: Use `uv sync --all-extras` to install development tools, then run `uv run pytest tests/`

## 💡 Easy Ways to Help

**No coding required:**
- Report bugs or issues you encounter
- Suggest improvements to documentation
- Share example use cases
- Test the package with your data and report results

**Light coding:**
- Fix typos in documentation
- Add examples or tutorials
- Improve error messages

**More involved:**
- Add new model architectures
- Improve data processing tools
- Add visualization features

## 🆘 Need Help?

- **First time with Git/GitHub?** Check out [GitHub's guides](https://guides.github.com/)
- **Python packaging confusion?** Just ask in an issue - we're happy to help!
- **Not sure where to start?** Look for issues labeled "good first issue"

## 📝 Guidelines

- **Be descriptive** in commit messages and pull requests
- **Test your changes** - make sure imports still work
- **Ask questions** - we'd rather help than have you struggle alone

We appreciate all contributions, whether it's fixing a typo or adding a major feature!
