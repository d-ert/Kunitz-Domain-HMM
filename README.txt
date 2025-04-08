# Kunitz Domain (PF00014) HMM Pipeline

## Overview

This script automates a bioinformatics pipeline designed to:
1.  Identify candidate Kunitz domain (Pfam: PF00014) sequences and structures from an RCSB PDB report.
2.  Filter redundant sequences using CD-HIT.
3.  Split data into training and testing sets.
4.  Download full PDB structures and extract the specific Kunitz domain regions based on PDBe API boundary information.
5.  Perform multiple structure alignment on the training set Kunitz domains using MUSTANG.
6.  Build a Hidden Markov Model (HMM) from the structure-based alignment using HMMER.
7.  Validate the HMM performance on a test set containing known Kunitz domains and negative sequences.
8.  Use the validated HMM to search for potential Kunitz domains within the SwissProt database.
9.  Evaluate the SwissProt search results against known annotations.

## Prerequisites

*   **Python 3:** With libraries specified in the `import` statements (e.g., `pandas`, `numpy`, `requests`, `BioPython`, `scikit-learn`, `matplotlib`, `seaborn`). Install using pip:
    ```bash
    pip install pandas numpy requests biopython scikit-learn matplotlib seaborn
    ```
*   **External Bioinformatics Tools:**
    *   [CD-HIT](https://github.com/weizhongli/cdhit)
    *   [MUSTANG](http://laskowskilab.org/mustang/) (Ensure the path in `run_mustang_alignment` is correct or it's in your PATH)
    *   [HMMER Suite](http://hmmer.org/) (`hmmbuild`, `hmmpress`, `hmmsearch`)
*   **WSL (for Windows users):** As this script was developed on Windows, it uses WSL to call the external tools. See the **Running on Linux/macOS** section below.

## How to Run

1.  Ensure all prerequisites (Python libraries, external tools) are installed and accessible.
2.  Place the required input files in the correct directories (`data/input_data/`, `data/swiss_data/`).
3.  Modify the MUSTANG executable path in the `run_mustang_alignment` function if necessary.
4.  Execute the script from your terminal:
    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of the Python script).

## Running on Linux/macOS (WSL Note)

The following functions include a `wsl=True` argument and prepend `wsl` to the command:
*   `run_cd_hit`
*   `run_mustang_alignment`
*   `build_hmm`
*   `hmmpress`
*   `run_hmmsearch`

**If you are running this script on a native Linux or macOS environment** where these tools are installed and available in your system's PATH, you **must** modify the calls to these functions within the `main()` block. Change the `wsl=True` argument (or its default) to `wsl=False`.

*Example:*
Change:
```python
# Call in main() section might look like this by default:
run_cd_hit(input_fasta=cd_hit_input, output_fasta=cd_hit_output)
# or explicitly:
# run_cd_hit(input_fasta=..., output_fasta=..., wsl=True)

to:

# Add wsl=False:
run_cd_hit(input_fasta=cd_hit_input, output_fasta=cd_hit_output, wsl=False)

Apply this change to all calls of the listed functions (run_cd_hit, run_mustang_alignment, build_hmm, hmmpress, run_hmmsearch) in the main() part of the script.


Output
The script will:
* Populate the directories under data/ with intermediate and final files (FASTA sequences, PDB structures, alignments, HMM files, search results).
* Generate HMM search output files (.domtblout, .txt) in data/test_data/ and data/swiss_data/.
* Save evaluation results, including confusion matrices and PR/ROC curve plots (.png), in data/test_data/ and data/swiss_data/.
* Create a CSV file (data/swiss_data/swissprot_kunitz_hmm_hits.csv) listing potential Kunitz domains found in SwissProt.
* Print progress and summary statistics to the console.