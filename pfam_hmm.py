import sys
import argparse
import pandas as pd
import numpy as np
import requests
import time
import os
import re 
import subprocess
import shutil
import random
from Bio import PDB
from Bio import SearchIO
from Bio import SeqIO 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import gc
import matplotlib.pyplot as plt
import seaborn as sns


### --- Configuration ---
BASE_DATA_DIR        = "data"

INPUT_DIR            = f"{BASE_DATA_DIR}/input_data"
FASTA_DIR            = f"{BASE_DATA_DIR}/fasta_full" # Directory for full FASTAs
PDB_DIR              = f"{BASE_DATA_DIR}/pdbs" # Directory for downloaded full PDBs
KUNITZ_PDB_TRAIN_DIR = f"{BASE_DATA_DIR}/kunitz_pdbs_train"
KUNITZ_PDB_TEST_DIR  = f"{BASE_DATA_DIR}/kunitz_pdbs_test"
HMM_DIR              = f"{BASE_DATA_DIR}/hmms"
TEST_DATA_DIR        = f"{BASE_DATA_DIR}/test_data" # For combined test sequences
SWISS_DIR            = f"{BASE_DATA_DIR}/swiss_data"

# Create directories if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(FASTA_DIR, exist_ok=True)
os.makedirs(PDB_DIR, exist_ok=True)
os.makedirs(KUNITZ_PDB_TRAIN_DIR, exist_ok=True)
os.makedirs(KUNITZ_PDB_TEST_DIR, exist_ok=True)
os.makedirs(HMM_DIR, exist_ok=True)
os.makedirs(TEST_DATA_DIR, exist_ok=True)
os.makedirs(SWISS_DIR, exist_ok=True)


### --- Functions ---
def extract_pfam_candidates(input_csv_path, output_fasta_dir, pfam_acc):
    """
    Extracts PDB IDs, Auth Asym IDs, and sequences for entries with
    "PF00014" annotation from a CSV report, saves them to a FASTA file,
    and returns a Pandas DataFrame of the filtered data.

    Args:
        input_csv_path (str): Path to the input CSV report file.
        output_fasta_dir (str): Directory to save the output FASTA file.

    Returns:
        pandas.DataFrame: DataFrame containing "PDB ID", "Auth Asym ID", and
                          "Sequence" for entries with "PF00014" annotation.
    """
    try:
        df_report = pd.read_csv(input_csv_path)

        # Restructure the dataframe if header is on the second row
        if df_report.columns[0] != "PDB ID":
            df_report.columns = df_report.iloc[0]
            df_report = df_report[1:]
            df_report.reset_index(drop=True, inplace=True)

        print("RCSB Report Loaded:")
        df_ids = df_report[df_report["Annotation Identifier"] == pfam_acc].copy()
        df_ids["PDB ID"].fillna(method="ffill", inplace=True) # Fill missing PDB IDs with the last valid PDB ID (forward fill)
        df_ids.reset_index(drop=True, inplace=True)

        initial_candidates = df_ids[["PDB ID", "Auth Asym ID", "Sequence"]].drop_duplicates().to_dict(orient="records")
        print(f"Found {len(initial_candidates)} initial PDB ID/Chain candidates from {len(df_ids['PDB ID'].unique())} unique PDB IDs.")
        if initial_candidates:
            print(f"Example candidate: {initial_candidates[0]}\n")

            # Define the output FASTA file path
            os.makedirs(output_fasta_dir, exist_ok=True)
            output_fasta_path = f"{output_fasta_dir}/initial_candidates_{pfam_acc}.fasta"
            # Clear existing file if it exists
            if os.path.exists(output_fasta_path):
                os.remove(output_fasta_path)

            
            with open(output_fasta_path, "w") as fasta_file:
                for candidate in initial_candidates:
                    pdb_id = candidate["PDB ID"]
                    chain_id = candidate["Auth Asym ID"]
                    sequence = candidate["Sequence"]
                    fasta_file.write(f">{pdb_id}_{chain_id}\n{sequence}\n")

            print(f"FASTA file created at: {output_fasta_path}")
        else:
            print(f"No candidates found for Annotation Identifier {pfam_acc}.")


    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return None
    except KeyError as e:
        print(f"Error: Column not found in the CSV file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def run_cd_hit(input_fasta, output_fasta, use_wsl, threshold=0.95, word_size=5):
    """Runs CD-HIT on the input FASTA file."""

    output_cluster_file = output_fasta + ".clstr" # CD-HIT also generates a cluster file
    
    command = [
        "cd-hit",
        "-i", input_fasta,
        "-o", output_fasta,
        "-c", str(threshold),
        "-n", str(word_size),
        "-d", "0", # prevent description truncation
        "-T", "0"  # use all available threads
    ]
    
    if use_wsl:
        command.insert(0, "wsl")
        
        

    print(f"\nRunning CD-HIT: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        #print("CD-HIT Output (stdout):")
        #print(result.stdout)
        #print("CD-HIT Output (stderr):")
        #print(result.stderr)
        print(f"CD-HIT filtering complete. Representative sequences saved to {output_fasta}")
        print(f"Cluster information saved to {output_cluster_file}")
        return 
    except FileNotFoundError:
        print("Error: 'wsl' or 'cd-hit' command not found. Make sure WSL and CD-HIT are installed and in your PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error during CD-HIT execution:")
        print(e.stderr)
        return None


def run_mustang_alignment(pdb_dir, pdb_paths, output_base, use_wsl, mustang_executable="./mustang_v3.2.4/bin/mustang-3.2.4"):
    """
    Runs MUSTANG for structure-based alignment on a set of PDB files.

    Args:
        pdb_dir (str): Path to directory containing PDB files (for -p).
        pdb_paths (list): List of full paths to individual PDB files.
        output_base (str): Base name for MUSTANG output files.
        mustang_executable (str): Path to the mustang binary.
        wsl (bool): Whether to run the command under WSL.

    Returns:
        str: Path to the aligned FASTA file.
    """

    output_fasta = f"{output_base}.afasta"
    pdb_filenames = [f"/{os.path.basename(p)}" for p in pdb_paths]
    

    command = [mustang_executable, '-p', pdb_dir, '-i'] + pdb_filenames + ['-o', output_base, '-F', 'fasta']
    
    if use_wsl:
        command.insert(0, "wsl")
        #input_fasta =f"/mnt/{input_fasta}"
        
    print("\n--- Running MUSTANG for Multiple Structure Alignment ---")
    print(f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
        if result.stderr:
            print("MUSTANG stderr (potential warnings/info):")
            print(result.stderr)
        print(f"Aligned FASTA saved to: {output_fasta}")
        return output_fasta
    except FileNotFoundError:
        print("Error: MUSTANG executable not found.")
        raise
    except subprocess.CalledProcessError as e:
        print("MUSTANG execution failed.")
        if e.stderr:
            print(e.stderr)
        raise


pfam_boundary_cache = {}
def get_pfam_boundaries_for_chains(pdb_id, chain_id, pfam_acc):
    """Gets PDB residue boundaries for a Pfam domain on a specific chain using Author Chain ID."""
    pdb_id_lower = pdb_id.lower()
    cache_key = (pdb_id_lower, pfam_acc)

    # Check cache first
    if cache_key in pfam_boundary_cache:
        all_boundaries = pfam_boundary_cache[cache_key]
    else:
        print(f"Querying PDBe API for boundaries: {pdb_id}")
        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/pfam/{pdb_id_lower}"
        print(url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if pdb_id_lower not in data or "Pfam" not in data[pdb_id_lower] or pfam_acc not in data[pdb_id_lower]["Pfam"]:
            print(f"  -> No Pfam {pfam_acc} mappings found for {pdb_id} via API.")
            pfam_boundary_cache[cache_key] = {} # Cache the negative result
            return None
        pfam_mapping = data[pdb_id_lower]["Pfam"][pfam_acc]["mappings"]
        pfam_boundary_cache[cache_key] = pfam_mapping # Store successful result in cache
        all_boundaries = pfam_mapping
        time.sleep(0.1) 


    # Now find the specific boundary for the requested author chain ID
    for mapping in all_boundaries:
        api_chain_id = mapping.get("chain_id") # This is likely the Auth Chain ID 
        api_auth_chain_id = mapping.get("author_chain_id", api_chain_id) # Fallback if author_chain_id field isn't present

        if api_auth_chain_id == chain_id:
            start = mapping["start"]["author_residue_number"]
            end = mapping["end"]["author_residue_number"]
            # Fallback to residue_number if author_residue_number is null
            if start is None:
                start = mapping["start"]["residue_number"]
            if end is None:
                end = mapping["end"]["residue_number"]

            if start is not None and end is not None:
                print(f"  -> Found Kunitz boundary for {pdb_id} Chain {chain_id}: Start={start}, End={end}")
                return {"start": int(start), "end": int(end)}
            else:
                print(f"  -> Found mapping for {pdb_id} Chain {chain_id}, but start/end numbers are missing.")
                return None # Boundary numbers missing

    print(f"  -> No specific mapping found for Author Chain {chain_id} in {pdb_id} (Pfam {pfam_acc}).")
    return None


def download_pdb(pdb_id, save_dir):
    """Downloads a PDB file from RCSB and saves it locally."""
    pdb_id_upper = pdb_id.upper()
    file_path = f"{save_dir}/{pdb_id_upper}.pdb"
    downloaded_pdb_files = set() # Keep track of downloaded PDBs


    if file_path in downloaded_pdb_files:
        print(f"PDB already downloaded: {file_path}")
        return file_path
    if os.path.exists(file_path):
        print(f"PDB file already exists locally: {file_path}")
        downloaded_pdb_files.add(file_path)
        return file_path

    url = f"https://files.rcsb.org/download/{pdb_id_upper}.pdb"
    print(f"Downloading PDB for {pdb_id_upper}...")
    
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    os.makedirs(save_dir, exist_ok=True)
    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f" -> Saved PDB to {file_path}")
    downloaded_pdb_files.add(file_path)
    time.sleep(0.1) 
    
    return file_path


def extract_kunitz_domain_pdb(pdb_id, chain_id, start, end, input_pdb_path, output_dir):
    """Creates a new PDB file containing only the Kunitz domain for a specific chain."""
    output_path = f"{output_dir}/{pdb_id}_{chain_id}_kunitz.pdb"

    if not os.path.exists(input_pdb_path):
        print(f"  -> PDB file {input_pdb_path} not found, skipping extraction for {pdb_id}_{chain_id}.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    extracted = False
    with open(input_pdb_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # PDB format: Chain ID is at index 21, Residue number is at indices 22-26
                line_chain = line[21].strip()
                if line_chain == chain_id:
                    res_num = int(line[22:26].strip())
                    if start <= res_num <= end:
                        outfile.write(line)
                        extracted = True

    if extracted:
            print(f"  -> Extracted Kunitz domain ({pdb_id} Chain {chain_id}, {start}-{end}) saved to {output_path}")
            return output_path
    else:
            print(f"  -> No ATOM/HETATM records matched criteria for {pdb_id} Chain {chain_id} ({start}-{end}) in {input_pdb_path}. Output file might be empty.")
            # Optionally remove empty file:
            if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
                os.remove(output_path)
            return None


def process_representatives(representatives, dataset_name, output_dir, pfam_acc):
    """
    Processes a list of protein structure representatives to extract Kunitz domains.

    Args:
        representatives (list): A list of dictionaries, where each dictionary
                                 contains 'pdb_id' and 'chain_id'.
        dataset_name (str): The name of the dataset being processed (e.g., "TRAINING", "TEST").
        output_dir (str): The directory to save the extracted Kunitz domain PDB files.

    Returns:
        list: A list of paths to the successfully extracted Kunitz domain PDB files.
    """
    print(f"\n--- Processing {dataset_name} Representatives ---")
    processed_kunitz_pdbs = []
    boundaries_cache = {} # Store boundaries for the current dataset

    for rep in representatives:
        pdb_id = rep["pdb_id"]
        chain_id = rep["chain_id"]
        print(f"Processing {dataset_name} Rep: {pdb_id} Chain {chain_id}")

        # 1. Get Boundaries
        boundaries = get_pfam_boundaries_for_chains(pdb_id, chain_id, pfam_acc)
        if not boundaries:
            print(f" -> Failed to get boundaries for {pdb_id} Chain {chain_id}. Skipping.")
            continue
        start, end = boundaries["start"], boundaries["end"]
        boundaries_cache[f"{pdb_id}_{chain_id}"] = boundaries # Store if needed

        # 2. Download PDB
        pdb_path = download_pdb(pdb_id, PDB_DIR)
        if not pdb_path:
            print(f" -> Failed to download PDB {pdb_id}. Skipping extraction.")
            continue

        # 3. Extract Kunitz Domain PDB
        kunitz_pdb_path = extract_kunitz_domain_pdb(pdb_id, chain_id, start, end, pdb_path, output_dir)
        if kunitz_pdb_path:
            processed_kunitz_pdbs.append(kunitz_pdb_path)

    print(f"\n--- Completed Processing {dataset_name} Representatives: {len(processed_kunitz_pdbs)} Kunitz domains extracted ---")
    return processed_kunitz_pdbs


def pdb_to_fasta_single_entry(pdb_filename):
    """Converts a single PDB file to a FASTA sequence string."""
    # Use QUIET=True if PDB parsing warnings are noisy
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_filename)
    ppb = PDB.PPBuilder()
    sequence = ""
    # Handle multi-model PDBs, process only the first model
    model = structure[0]
    for pp in ppb.build_peptides(model):
        sequence += pp.get_sequence()

    if not sequence:
        print(f"Warning: No polypeptide sequence found in {pdb_filename}")
        return None, None

    # Create a FASTA ID from the filename (e.g., 1BTI_A_kunitz)
    pdb_name = os.path.basename(pdb_filename).replace(".pdb", "")
    fasta_header = f">{pdb_name}"
    return fasta_header, str(sequence)


def build_hmm(hmm_name, fasta_file, use_wsl):
    """Builds HMM using hmmbuild."""
    
    command = ["hmmbuild", hmm_name, fasta_file]

    if use_wsl:
        command.insert(0, "wsl")

    print(f"Running hmmbuild: {' '.join(command)}")
    
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    
    print("hmmbuild Output (stdout):")
    print(result.stdout)
    
    if result.stderr:
        print("hmmbuild Output (stderr):")
        print(result.stderr)
    
    print(f"HMM built successfully: {hmm_name}")
    return hmm_name


def hmmpress(hmm_file, use_wsl):
    """Presses the HMM file using hmmpress."""

    command = ["hmmpress", hmm_file]

    if use_wsl:
        command.insert(0, "wsl")

    print(f"Running hmmpress: {' '.join(command)}")
    
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    
    if result.stderr:
        print("hmmpress Output (stderr):")
        print(result.stderr)
    
    print(f"HMM pressed successfully: {hmm_file}")
    return True


def run_hmmsearch(hmm_file, sequence_file, output_base, use_wsl, e_value=0.01, use_max=False, domtblout=True):
    """Runs hmmsearch and returns output file path."""
    
    domtblout_file = f"{output_base}.domtbl"
    txt_output_file = f"{output_base}.txt"

    command = ["hmmsearch"]
    
    if use_max:
        command.append("--max")  # Report scores for top hit only, not domains
    if domtblout:
        command.extend(["--domtblout", domtblout_file])
    
    command.extend(["--incE", str(e_value)])
    command.extend([hmm_file, sequence_file])  # HMM file and sequence DB

    if use_wsl:
        command.insert(0, "wsl")

    print(f"Running hmmsearch: {' '.join(command)} > {txt_output_file}")

    try:
        with open(txt_output_file, "w") as outfile:
            result = subprocess.run(command, check=True, text=True, stdout=outfile, stderr=subprocess.PIPE)

        if result.stderr:
            print("hmmsearch Output (stderr):")
            print(result.stderr)

        print(f"HMMER search completed. Domain results: {domtblout_file}, Text output: {txt_output_file}")
        return domtblout_file if domtblout else txt_output_file

    except FileNotFoundError:
        print("Error: 'wsl' or 'hmmsearch' command not found.")
        raise

    except subprocess.CalledProcessError as e:
        print("Error during hmmsearch execution:")
        if e.stderr:
            print(e.stderr)
        if os.path.exists(txt_output_file):
            print(f"Content of {txt_output_file}:")
            with open(txt_output_file, 'r') as f:
                print(f.read())
        raise


def parse_swissprot_dat(dat_file):
    """Parse SwissProt .dat file and extract primary accessions and Pfam domain IDs."""
    data, entry = [], {}
    count = 0

    print(f"Parsing SwissProt .dat file: {dat_file} ... (This may take time)")

    with open(dat_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('//'):
                if 'AC' in entry:
                    entry['AC'] = entry['AC'].split(';')[0].strip()
                    data.append(entry)
                    count += 1
                    if count % 10000 == 0:
                        print(f"  Processed {count} entries...")
                entry = {}
                continue

            if line.startswith('AC'):
                entry['AC'] = entry.get('AC', '') + line[5:]
            elif line.startswith('DR'):
                entry['DR'] = entry.get('DR', '') + '\n' + line[5:] if 'DR' in entry else line[5:]

    print(f"Finished parsing. Total entries processed: {count}")

    if not data:
        print("Warning: No data parsed from .dat file.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    print("Extracting Pfam domains from DR fields...")
    df['Pfam'] = df['DR'].apply(lambda x: re.findall(r"Pfam;\s*(PF\d+);", x) if isinstance(x, str) else [])
    del df['DR']
    gc.collect()
    return df[['AC', 'Pfam']]





def main(pfam_acc, input_csv_path, use_wsl):
    # Take the report from the RCSB PDB and extract the PF00014 candidates
    
    extract_pfam_candidates(
        input_csv_path= input_csv_path,
        output_fasta_dir= FASTA_DIR,
        pfam_acc=pfam_acc
    )

    # Run CD-HIT to filter the sequences, remove redundant sequences
    cd_hit_input = f"{FASTA_DIR}/initial_candidates_{pfam_acc}.fasta"
    cd_hit_output = f"{FASTA_DIR}/representatives_cdhit95_{pfam_acc}.fasta"
    cd_hit_threshold = 0.95
    cd_hit_word_size = 5 

    run_cd_hit(
        input_fasta=cd_hit_input,
        output_fasta=cd_hit_output,
        threshold=cd_hit_threshold,
        word_size=cd_hit_word_size,
        use_wsl=use_wsl
    )

    # Read the CD-HIT output file and parse headers
    print(f"\n--- Parsing representative IDs from {cd_hit_output} ---")
    representative_pdb_chain_pairs = []
    parsed_headers_count = 0
    with open(cd_hit_output, 'r') as f:
        for line in f:
            if line.startswith('>'):
                parsed_headers_count += 1
                header = line[1:-1].split("_")
                parsed_pairs = (header[0],header[1])
                if parsed_pairs:
                    representative_pdb_chain_pairs.append(parsed_pairs)

    # Remove duplicates
    representative_pdb_chain_pairs = list(set(representative_pdb_chain_pairs))

    print(f"Parsed {parsed_headers_count} headers from CD-HIT output.")
    print(f"Identified {len(representative_pdb_chain_pairs)} unique representative PDB ID / Author Chain ID pairs.")
    if representative_pdb_chain_pairs:
        print(f"Example representative pair: {representative_pdb_chain_pairs[0]}\n")


    # Train/Test Split
    if not representative_pdb_chain_pairs:
        raise ValueError("Cannot split representatives: The list is empty.")

    # Convert list of tuples to list of dicts for easier handling later
    representatives_info = [{"pdb_id": p, "chain_id": c} for p, c in representative_pdb_chain_pairs]

    # Split the representatives
    train_reps, test_reps = train_test_split(
        representatives_info,
        test_size=0.2,        # 20% for testing
        random_state=42       # for reproducibility
    )


    print(f"Split representatives into {len(train_reps)} training and {len(test_reps)} testing samples.\n")    

    # Process Training Representatives
    print(train_reps)
    processed_train_kunitz_pdbs = process_representatives(train_reps, "TRAINING", KUNITZ_PDB_TRAIN_DIR, pfam_acc)

    # Process Test Representatives 
    processed_test_kunitz_pdbs = process_representatives(test_reps, "TEST", KUNITZ_PDB_TEST_DIR, pfam_acc)

    # Check if we have training domains before proceeding
    if not processed_train_kunitz_pdbs:
        raise ValueError("No Kunitz domains were successfully extracted for the training set. Cannot proceed.")


    ### --- Sctructure based alignment with MUSTANG ---

    mustang_alignment_out_fasta = run_mustang_alignment(KUNITZ_PDB_TRAIN_DIR, processed_train_kunitz_pdbs, f"{KUNITZ_PDB_TRAIN_DIR}/mustang_aligned", use_wsl=use_wsl)


    ### --- HMM Construction ---
    hmm_file_path = f"{HMM_DIR}/kunitz_{pfam_acc}.hmm"
    aligned_fasta_for_hmm = mustang_alignment_out_fasta # Use the output from MUSTANG

    # Clean up old HMM files
    if os.path.exists(hmm_file_path):
        print(f"Removing existing HMM file: {hmm_file_path}")
        os.remove(hmm_file_path)
    for ext in ['.h3f', '.h3i', '.h3m', '.h3p']:
        if os.path.exists(hmm_file_path + ext):
            os.remove(hmm_file_path + ext)

    print("\n--- Building and Pressing HMM ---")
    built_hmm = build_hmm(hmm_file_path, aligned_fasta_for_hmm, use_wsl=use_wsl)
    if built_hmm:
        hmmpress(built_hmm, use_wsl=use_wsl)

    print("\n--- HMM Construction Complete ---")

    ### Prepare test set
    # --- Convert Positive Test Kunitz PDBs to FASTA ---
    print("\n--- Converting Positive Test Kunitz PDBs to FASTA ---")
    positive_test_fasta_path = f"{TEST_DATA_DIR}/positive_test_{pfam_acc}.fasta" 
    positive_test_ids = []

    # Clear existing file
    if os.path.exists(positive_test_fasta_path):
        os.remove(positive_test_fasta_path)

    with open(positive_test_fasta_path, "w") as outfile:
        for pdb_file in processed_test_kunitz_pdbs:
            header, sequence = pdb_to_fasta_single_entry(pdb_file)
            if header and sequence:
                outfile.write(f"{header}\n{sequence}\n")
                positive_test_ids.append(header[1:]) # Store ID without '>'

    print(f"Converted {len(positive_test_ids)} positive test Kunitz domains to {positive_test_fasta_path}")

    # --- Combine Positive Test + Negative Sequences ---
    negative_data_fasta = f"{INPUT_DIR}/uniprotkb_reviewed_true_AND_xref_pfam_P_2025_04_01.fasta"
    if not os.path.exists(negative_data_fasta):
        raise FileNotFoundError(f"Negative dataset FASTA file not found: {negative_data_fasta}")


    combined_test_fasta_path = f"{TEST_DATA_DIR}/combined_test_{pfam_acc}.fasta" 

    # Clear existing file
    if os.path.exists(combined_test_fasta_path):
        os.remove(combined_test_fasta_path)

    # Concatenate files
    with open(combined_test_fasta_path, 'wb') as wfd:
        # Add positive test sequences first
        if os.path.exists(positive_test_fasta_path):
            with open(positive_test_fasta_path, 'rb') as rfd:
                shutil.copyfileobj(rfd, wfd)
        else:
            print(f"Warning: Positive test FASTA {positive_test_fasta_path} not found for concatenation.")

        # Add negative sequences
        with open(negative_data_fasta, 'rb') as rfd:
            shutil.copyfileobj(rfd, wfd)

    print(f"\n--- Combined positive test and negative sequences into {combined_test_fasta_path} ---")



    ### Perform hmmsearch on the test set
    # Define paths for HMM search output
    hmm_search_output_base = f"{TEST_DATA_DIR}/test_set_search_{pfam_acc}"
    search_e_value = 0.001 # E-value threshold for reporting hits

    print("\n--- Running hmmsearch on Combined Test Set ---")
    hmm_validation_domtblout = run_hmmsearch(
        hmm_file_path,
        combined_test_fasta_path,
        hmm_search_output_base,
        e_value=search_e_value,
        use_max=True, # Get domain hits, not just best sequence hit
        domtblout=True,
        use_wsl=use_wsl
    )


    ### Analyze HMMER Results


    if not hmm_validation_domtblout or not os.path.exists(hmm_validation_domtblout):
        raise FileNotFoundError(f"HMM search domain output file not found: {hmm_validation_domtblout}")

    print("\n--- Parsing HMM Search Results and Evaluating ---")

    # Use SearchIO to parse the domtblout file
    qresults = list(SearchIO.parse(hmm_validation_domtblout, "hmmsearch3-domtab"))


    data = []
    hit_ids_found = set() # Keep track of which sequences had at least one significant hit

    for qresult in qresults: # Iterate through queries (our HMM)
        for hit in qresult.hits: # Iterate through sequence hits
            hit_id = hit.id # Get the ID of the sequence that was hit
            hit_ids_found.add(hit_id)
            is_positive_truth = hit_id in positive_test_ids # Check if it's one of our known positives
            best_hsp = min(hit.hsps, key=lambda hsp: hsp.evalue) # Get HSP with lowest E-value
            data.append({
                "hit_id": hit_id,
                "is_positive_truth": is_positive_truth,
                "evalue": best_hsp.evalue,
                "bitscore": best_hsp.bitscore,
            })

    # Get all sequence IDs from the combined test FASTA
    all_test_seq_ids = set()
    with open(combined_test_fasta_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            all_test_seq_ids.add(record.id)


    missed_ids = all_test_seq_ids - hit_ids_found
    print(f"Found {len(hit_ids_found)} sequences with hits. {len(missed_ids)} sequences had no hits passing threshold.")

    for missed_id in missed_ids:
        is_positive_truth = missed_id in positive_test_ids
        data.append({
            "hit_id": missed_id,
            "is_positive_truth": is_positive_truth,
            "evalue": float('inf'), # Assign infinite E-value for non-hits
            "bitscore": float('-inf'), # Assign very low bitscore
        })


    df_hmm_results = pd.DataFrame(data)


    # --- Calculate Confusion Matrix ---
    classification_threshold = 0.0001 

    df_hmm_results['predicted_positive'] = df_hmm_results['evalue'] <= classification_threshold

    y_true = df_hmm_results['is_positive_truth']
    y_pred = df_hmm_results['predicted_positive']

    cm = confusion_matrix(y_true, y_pred)

    print(f"\nConfusion Matrix on Test Set (Threshold E-value <= {classification_threshold}):")
    print(cm)

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0 
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("Test Set Evaluation Metrics:") 
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-score:    {f1_score:.4f}")

    # --- Create Plot ---
    group_names = ['True Neg (TN)', 'False Pos (FP)', 'False Neg (FN)', 'True Pos (TP)']
    group_counts = ["{:,}".format(value) for value in cm.flatten()]
    annot_labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    annot_labels = np.asarray(annot_labels).reshape(2, 2)

    # Create figure and axes explicitly
    fig, ax = plt.subplots(figsize=(7, 6)) # Adjust figsize if needed

    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', cbar=False, ax=ax, annot_kws={"size": 12}) # Pass ax
    ax.set_ylabel('Actual Label')
    ax.set_xlabel('Predicted Label')
    # Make title dynamic
    ax.set_title(f'HMM Validation on Test Set ({pfam_acc}, E-val ≤ {classification_threshold})')

    # --- Add Metrics Text Below Plot ---
    metrics_text = (
        f"Accuracy:    {accuracy:.4f}\n"
        f"Precision:   {precision:.4f}\n"
        f"Recall:      {recall:.4f}\n"
        f"Specificity: {specificity:.4f}\n"
        f"F1-score:    {f1_score:.4f}"
    )

    # Add text relative to the axes (ax).
    ax.text(0.5, -0.18, metrics_text, # Adjust Y position (-0.18) if needed
            ha='center', va='top', transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8)) # Optional box

    # Adjust layout to make room for the text
    plt.tight_layout(pad=2.0) # Increase padding if text gets cut off

    # --- Save Plot ---
    plot_filename = f"{TEST_DATA_DIR}/hmm_validation_confusion_matrix_{pfam_acc}.png"
    plt.savefig(plot_filename)
    print(f"Saved test set validation plot with metrics to: {plot_filename}")
    plt.close(fig) # Close the figure

    ### Annotate SwissProt
    

    # Define paths for SwissProt annotation
    swissprot_fasta_file = f"{SWISS_DIR}/swiss_all.fasta" # Assuming you have swiss_all.fasta or similar
    hmm_file_for_swiss = hmm_file_path # Use the HMM built earlier
    swiss_search_output_base = f"{SWISS_DIR}/hmmsearch_swissprot_{pfam_acc}"  
    swiss_search_e_value = 0.001 # E-value threshold for searching SwissProt

    # Check if SwissProt FASTA file exists
    if not os.path.exists(swissprot_fasta_file):
        raise FileNotFoundError(f"SwissProt FASTA file not found: {swissprot_fasta_file}. Please download it (e.g., uniprot_sprot.fasta).")


    print(f"\n--- Searching SwissProt ({swissprot_fasta_file}) with HMM ---")
    swiss_domtblout_path = run_hmmsearch(
        hmm_file_for_swiss,
        swissprot_fasta_file,
        swiss_search_output_base,
        e_value=swiss_search_e_value,
        use_max=False, # Get domain hits
        domtblout=True,
        use_wsl=use_wsl
    )


    # Parse SwissProt hmmsearch results
    if not swiss_domtblout_path or not os.path.exists(swiss_domtblout_path):
        raise FileNotFoundError(f"HMM search domain output file for SwissProt not found: {swiss_domtblout_path}")

    print("\n--- Parsing SwissProt HMM Search Results ---")
    swiss_results = []

    swiss_qresults = SearchIO.parse(swiss_domtblout_path, "hmmsearch3-domtab")
    for qresult in swiss_qresults:
        for hit in qresult.hits:
                # Parse UniProt ID (handle different header formats like sp|P00974|...)
                uniprot_id_parts = hit.id.split('|')
                if len(uniprot_id_parts) >= 2:
                    uniprot_id = uniprot_id_parts[1] # Usually the accession is the second part
                else:
                    uniprot_id = hit.id # Fallback to the full ID

                for hsp in hit.hsps:
                    swiss_results.append({
                        "uniprot_id": uniprot_id,
                        "full_hit_id": hit.id, # Keep the full ID for reference
                        "evalue": hsp.evalue,
                        "bitscore": hsp.bitscore,
                        "domain_start": hsp.hit_start, # Start/end of the domain hit in the sequence
                        "domain_end": hsp.hit_end,
                        "query_start": hsp.query_start, # Start/end match in the HMM
                        "query_end": hsp.query_end,
                    })

    df_swissprot_hmm = pd.DataFrame(swiss_results)
    # Sort by E-value to see the best hits first
    df_swissprot_hmm = df_swissprot_hmm.sort_values(by="evalue").reset_index(drop=True)

    print(f"Found {len(df_swissprot_hmm)} potential Kunitz domain hits in SwissProt (based on E-value <= {swiss_search_e_value}).")
    print(df_swissprot_hmm.head())

    # Save the results to a CSV
    swiss_results_csv_path = f"{SWISS_DIR}/swissprot_{pfam_acc}_hmm_hits.csv" 
    df_swissprot_hmm.to_csv(swiss_results_csv_path, index=False)
    print(f"SwissProt HMM search results saved to: {swiss_results_csv_path}")



    # --- Load or Create Annotations ---
    annotations_csv = f"{SWISS_DIR}/swissprot_annotations.csv"
    dat_file = f"{SWISS_DIR}/uniprot_sprot.dat"

    if os.path.exists(annotations_csv):
        print(f"\n--- Loading existing annotations from {annotations_csv} ---")
        df_annotations = pd.read_csv(annotations_csv)
    else:
        print(f"\n--- No annotation CSV file found. Parsing {dat_file} ---")
        df_annotations = parse_swissprot_dat(dat_file)
        print("Parsed the data, now saving to CSV...")
        df_annotations.to_csv(annotations_csv, index=False)
        print(f"SwissProt Pfam annotations saved to: {annotations_csv}")


    ### Visualize
    if  df_annotations.empty:
        raise ValueError("No annotation data available. Cannot proceed with evaluation.")

    annotated = set(df_annotations[df_annotations['Pfam'].astype(str).str.contains(f"'{pfam_acc}'") ]['AC'])
    predicted = set(df_swissprot_hmm[df_swissprot_hmm['evalue'] <= swiss_search_e_value]['uniprot_id'])

    tp = len(predicted & annotated)
    fp = len(predicted - annotated)
    fn = len(annotated - predicted)
    tn = len(df_annotations) - len(annotated) - fp

    print(f"\nComparison with SwissProt {pfam_acc} Annotations:")
    print(f"  True Positives (HMM predicted, Pfam annotated): {tp}")
    print(f"  False Positives (HMM predicted, NOT Pfam ann.): {fp}")
    print(f"  False Negatives (NOT HMM predicted, Pfam ann.): {fn}")
    print(f"  True Negatives (NOT HMM predicted, NOT Pfam ann.): {tn}")

    # --- Metrics ---
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # Also called Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\nEvaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Specificity:{specificity:.4f}")
    print(f"F1-score:  {f1_score:.4f}")

    # --- Create Plot ---
    cm = np.array([[tn, fp], [fn, tp]])
    labels = ['TN', 'FP', 'FN', 'TP']
    counts = [f"{v:,}" for v in cm.flatten()]
    annot_labels = np.array([f"{l}\n{c}" for l, c in zip(labels, counts)]).reshape(2, 2)

    fig, ax = plt.subplots(figsize=(7, 6)) # Adjust figsize if needed

    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', cbar=False, ax=ax, annot_kws={"size": 12}) # Pass ax, adjust font size if needed
    ax.set_ylabel('Actual Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'HMM Validation on SwissProt ({pfam_acc}, E-val ≤ {swiss_search_e_value})')

    metrics_text = (
        f"Accuracy:    {accuracy:.4f}\n"
        f"Precision:   {precision:.4f}\n"
        f"Recall:      {recall:.4f}\n"
        f"Specificity: {specificity:.4f}\n"
        f"F1-score:    {f1_score:.4f}"
    )

    ax.text(0.5, -0.18, metrics_text, # Adjust Y position (-0.18) if needed
            ha='center', va='top', transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8)) # Optional box

    plt.tight_layout(pad=2.0) # Increase padding if text gets cut off

    # --- Save Plot ---
    plot_filename = f"{SWISS_DIR}/hmm_validation_swissprot_{pfam_acc}.png"
    plt.savefig(plot_filename)
    print(f"Saved SwissProt validation plot with metrics to: {plot_filename}")
    plt.close(fig) # Close the figure to free up memory
    
    
    ### --- Plot PR/ROC curves ---
    all_acs = df_annotations['AC'].tolist()
    
    # Ground truth: 1 if annotated with PF00014, else 0
    y_true = [1 if ac in annotated else 0 for ac in all_acs]

    # Get prediction scores: use -log10(evalue), or 0 if not predicted (i.e., not found in HMM hits)
    evalue_dict = dict(zip(df_swissprot_hmm['uniprot_id'], df_swissprot_hmm['evalue']))
    max_evalue = max(evalue_dict.values()) if evalue_dict else 1e-4

    y_scores = [-np.log10(evalue_dict[ac]) if ac in evalue_dict else 0 for ac in all_acs]

    # --- Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{SWISS_DIR}/roc_pr_curves_{pfam_acc}.png") 

    
    print("\n--- Pipeline Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pfam HMM pipeline.")
    parser.add_argument("--pfam_acc", default="PF00014", 
                        help="Pfam Accession ID (e.g., PF00014)")
    parser.add_argument("--report_name", default="rcsb_report.csv", 
                        help="The name of the RCSB report (e.g., rcsb_report.csv)")
    parser.add_argument("--no-wsl", dest='use_wsl', action='store_false', default=True, 
                        help="Run external commands directly on Linux. Use to turn off wsl.")
    
    args = parser.parse_args()

    # Construct the full input path
    input_report_path = os.path.join(INPUT_DIR, args.report_name).replace("\\", "/")

    # Check if the input file exists
    if not os.path.exists(input_report_path):
        print(f"Error: Input report file not found at {input_report_path}")
        sys.exit(1) # Exit if the file doesn't exist

    print(f"Using Pfam Accession: {args.pfam_acc}")
    print(f"Using Input Report: {input_report_path}")
    print(f"Using WSL: {args.use_wsl}\n")

    main(pfam_acc=args.pfam_acc, input_csv_path=input_report_path, use_wsl=args.use_wsl)