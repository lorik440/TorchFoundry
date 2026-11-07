# Changelog

Notable changes to TorchFoundry are documented in this file. Each version includes a summary of the changes made, including new features, bug fixes, and other improvements.

## [Unreleased]

### Added
- GitHub issue forms and templates for bug reports, feature requests, and general questions, plus guidance for reporting security issues privately.
- A pull request template covering descriptions, related issues, testing, and contributor checklists.
- Accessibility guidance, contributing guidelines, a security policy, and a Contributor Covenant Code of Conduct.
- A local environment-variable file and `.gitignore` rules to exclude `.env` files.
- Initial dataset, model, and training utility module scaffolding.
- GitHub Actions checks for Ruff, pytest, and distribution builds.
- Dataset helpers for resolving torchvision datasets, preparing train/test splits, and loading them through `DataLoader`.
- Public package exports for PyTorch utilities, dataset helpers, model helpers, and `trainModel()`.
- Separate model creation, checkpoint loading, and checkpoint saving helpers.
- A FashionMNIST example demonstrating dataset loading, model creation, training, and saving.

### Changed
- Clarified the accessibility guidelines, including contributor expectations, issue reporting, and known limitations.
- Clarified the security policy's supported versions and vulnerability reporting procedures.
- Renamed the project and Python package from TorchForge to TorchFoundry.
- Added package metadata, runtime dependencies, and development dependencies.
- Readme file updated to reflect the new project name and provide installation instructions.
- Updated dataset and training utilities to support configurable dataset inputs, batch/shuffle options, epoch validation, and optional checkpoint saving.
- Updated test dataset preparation to use the requested dataset split.
- Updated model layer type hints to accept sequences and corrected accelerator selection.
- Added checkpoint loading before training and checkpoint-based evaluation, and updated the FashionMNIST example to use the checkpoint.
- Updated the README with a notice clarifying that TorchFoundry is not affiliated with or endorsed by the PyTorch organization.
- Marked completed dataset, model, and training tasks in the project TODO list.

### Removed
- Removed the TensorBoard requirement.

### Fixed
- Fixed the `model.py` file to include the correct package name and metadata.
- Fixed the `loadModel()` for better model architectyre loading and initialization.
- Fixed the `` to include the correct package name and metadata.
- Fixed `test_load_model_runs_layers()` match the expected model architecture and layer count.
