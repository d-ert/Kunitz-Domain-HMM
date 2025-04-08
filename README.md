# Kunitz Domain (PF00014) HMM Pipeline

## Overview

This script automates a bioinformatics pipeline designed to:
1.  Identify candidate Kunitz domain (Pfam: `PF00014`) sequences and structures from an RCSB PDB report.
2.  Filter redundant sequences using CD-HIT.
3.  Split data into training and testing sets.
4.  Download full PDB structures and extract the specific Kunitz domain regions based on PDBe API boundary information.
5.  Perform multiple structure alignment on the training set Kunitz domains using MUSTANG.
6.  Build a Hidden Markov Model (HMM) from the structure-based alignment using HMMER.
7.  Validate the HMM performance on a test set containing known Kunitz domains and negative sequences.
8.  Use the validated HMM to search for potential Kunitz domains within the SwissProt database.
9.  Evaluate the SwissProt search results against known annotations.


## Prerequisites

*   **Python 3:** With libraries specified in the `import` statements 
*   **External Bioinformatics Tools:**
    *   [CD-HIT](https://github.com/weizhongli/cdhit)
    *   [MUSTANG](http://laskowskilab.org/mustang/) (Ensure the path in `run_mustang_alignment` is correct or it's in your PATH)
    *   [HMMER Suite](http://hmmer.org/) (`hmmbuild`, `hmmpress`, `hmmsearch`)
*   **WSL (for Windows users):** As this script was developed on Windows, it uses WSL to call the external tools. See the **Running on Linux/macOS** section below.

## How to Run

There are two primary ways to execute this pipeline:

**1. Using the Python Script (`kunitz_hmm.py`):**

This method runs the entire pipeline sequentially from start to finish.

1.  Ensure all prerequisites (Python libraries, external tools) are installed and accessible.
2.  Place the required input files in the correct directories (`data/input_data/`, `data/swiss_data/`). See *Directory Structure & Input Data*.
3.  Modify the MUSTANG executable path in the `run_mustang_alignment` function within the script if necessary.
4.  Execute the script from your terminal:
    ```bash
    python kunitz_pipeline.py
    ```

**2. Using the Jupyter Notebook (`Pipeline Notebook.ipynb`):**

This repository also includes a Jupyter Notebook which mirrors the steps in the Python script. The notebook format is ideal for:
*   Running the pipeline step-by-step.
*   Visualizing plots directly within the notebook interface.
*   Interactive analysis and easier modification of intermediate steps.

1.  Ensure prerequisites are met, including Jupyter Lab or Jupyter Notebook (`pip install jupyterlab` or `pip install notebook`).
2.  Place the required input files in the correct directories as described above.
3.  Modify the MUSTANG executable path within the relevant notebook cell if needed.
4.  Launch Jupyter from your terminal in the project's root directory:
    ```bash
    jupyter lab
    ```
    *or*
    ```bash
    jupyter notebook
    ```

## Running on Linux/macOS (WSL Note)

The following functions include a `wsl=True` argument and prepend `wsl` to the command:
*   `run_cd_hit`
*   `run_mustang_alignment`
*   `build_hmm`
*   `hmmpress`
*   `run_hmmsearch`

**If you are running this script on a native Linux or macOS environment** where these tools are installed and available in your system's PATH, you **must** modify the calls to these functions within the `main()` block. Change the `wsl=True` argument (or its default) to `wsl=False`.

 version:

*   **Example:**

    Modify the function calls in the `main()` section.

    *Change this type of call:*
    ```python
    # Call in main() section might look like this by default:
    run_cd_hit(input_fasta=cd_hit_input, output_fasta=cd_hit_output)
    # or explicitly:
    # run_cd_hit(input_fasta=..., output_fasta=..., wsl=True)
    ```

    *To this:*
    ```python
    # Add wsl=False:
    run_cd_hit(input_fasta=cd_hit_input, output_fasta=cd_hit_output, wsl=False)
    ```

    Apply this change to all calls of the listed functions (`run_cd_hit`, `run_mustang_alignment`, `build_hmm`, `hmmpress`, `run_hmmsearch`) within the `main()` part of the script.
    
## Output

The script will:
* Populate the directories under data/ with intermediate and final files (FASTA sequences, PDB structures, alignments, HMM files, search results).
* Generate HMM search output files (.domtblout, .txt) in data/test_data/ and data/swiss_data/.
* Save evaluation results, including confusion matrices and PR/ROC curve plots (.png), in data/test_data/ and data/swiss_data/.
* Create a CSV file (data/swiss_data/swissprot_kunitz_hmm_hits.csv) listing potential Kunitz domains found in SwissProt.
* Print progress and summary statistics to the console.
