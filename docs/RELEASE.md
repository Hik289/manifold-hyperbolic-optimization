# Release notes

This release includes all original experiment and plotting source files. The [source inventory](source_inventory.json) lists their paths and SHA-256 hashes, which can be used to check file contents. The repository also provides commands to run experiments, check saved results, rebuild figures and compile the paper.

## Changes for local use

- Data and output paths are relative to the project or supplied as command arguments.
- Training scripts and small test examples accept a device setting, such as `cpu` or `cuda`.
- File synchronization takes source and destination folders as arguments.
- Worker records use a generic label instead of a computer's hostname.
- The script for the historical budget change only shows the proposed changes by default. Applying them requires `--apply` and the expected original state.
- Public records exclude computer names, process IDs and connection details. Metric values and experiment settings are preserved.
- The Git history begins with the reviewed public files; it does not include the private development history.

These changes preserve the optimizer formulas, full training lengths, seed 42 and evaluation rules.

## Figures and paper

The README image illustrates the method for groups with separate Lorentz spaces. It does not show measured results. Python scripts and data for the paper's loss curves are included, so those curves can be rebuilt without training.

The repository contains LaTeX sources and PNG figures. It does not contain PDFs. Compile the paper or run the figure scripts to create PDFs locally; `.gitignore` keeps them out of commits.

## How to read the results

The quick-start example checks that the code runs. Full experiments, configurations deferred after the budget change, and earlier exploratory studies are recorded separately. See the [result guide](../results/README.md) for file locations and the [experiment guide](EXPERIMENTS.md) for the evaluation rules.
