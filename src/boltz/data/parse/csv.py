from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from boltz.data import const
from boltz.data.types import MSA, MSADeletion, MSAResidue, MSASequence

# def parse_csv(
#     path: Path,
#     max_seqs: Optional[int] = None,
#     max_unpaired_seqs: Optional[int] = None,
# ) -> MSA:
#     """Process an A3M file.

#     Parameters
#     ----------
#     path : Path
#         The path to the a3m(.gz) file.
#     max_seqs : int, optional
#         The maximum number of sequences.

#     Returns
#     -------
#     MSA
#         The MSA object.

#     """
#     # Read file
#     data = pd.read_csv(path)

#     # Check columns
#     if tuple(sorted(data.columns)) != ("key", "sequence"):
#         msg = "Invalid CSV format, expected columns: ['sequence', 'key']"
#         raise ValueError(msg)

#     # Create taxonomy mapping
#     visited = set()
#     sequences = []
#     deletions = []
#     residues = []

#     seq_idx = 0
#     for line, key in zip(data["sequence"], data["key"]):
#         line: str
#         line = line.strip()
#         if not line:
#             continue

#         # Get taxonomy, if annotated
#         taxonomy_id = -1
#         if (str(key) != "nan") and (key is not None) and (key != ""):
#             taxonomy_id = key

#         # Skip if duplicate sequence
#         str_seq = line.replace("-", "").upper()
#         if str_seq not in visited:
#             visited.add(str_seq)
#         else:
#             continue

#         # Process sequence
#         residue = []
#         deletion = []
#         count = 0
#         res_idx = 0
#         for c in line:
#             if c != "-" and c.islower():
#                 count += 1
#                 continue
#             token = const.prot_letter_to_token[c]
#             token = const.token_ids[token]
#             residue.append(token)
#             if count > 0:
#                 deletion.append((res_idx, count))
#                 count = 0
#             res_idx += 1

#         res_start = len(residues)
#         res_end = res_start + len(residue)

#         del_start = len(deletions)
#         del_end = del_start + len(deletion)

#         sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
#         residues.extend(residue)
#         deletions.extend(deletion)

#         seq_idx += 1
#         if (max_seqs is not None) and (seq_idx >= max_seqs):
#             break

#     # Create MSA object
#     msa = MSA(
#         residues=np.array(residues, dtype=MSAResidue),
#         deletions=np.array(deletions, dtype=MSADeletion),
#         sequences=np.array(sequences, dtype=MSASequence),
#     )
#     return msa


# def parse_csv(
#     path: Path,
#     max_paired_seqs: Optional[int] = None,
#     max_unpaired_seqs: Optional[int] = None,
# ) -> MSA:
#     """Process an A3M file with paired/unpaired sequence filtering.

#     Parameters
#     ----------
#     path : Path
#         The path to the a3m(.gz) file.
#     max_paired_seqs : int, optional
#         The maximum number of paired sequences to include.
#     max_unpaired_seqs : int, optional
#         The maximum number of unpaired sequences to include.

#     Returns
#     -------
#     MSA
#         The MSA object.
#     """
#     # Read file
#     data = pd.read_csv(path)

#     # Check columns
#     if tuple(sorted(data.columns)) != ("key", "sequence"):
#         msg = "Invalid CSV format, expected columns: ['sequence', 'key']"
#         raise ValueError(msg)

#     # Create taxonomy mapping
#     visited = set()
#     sequences = []
#     deletions = []
#     residues = []
#     seq_idx = 0
#     paired_count = 0
#     unpaired_count = 0

#     for line, key in zip(data["sequence"], data["key"]):
#         line: str
#         line = line.strip()
#         if not line:
#             continue

#         # Get taxonomy, if annotated
#         taxonomy_id = -1
#         is_paired = False
#         if (str(key) != "nan") and (key is not None) and (key != ""):
#             taxonomy_id = key
#             # Assuming paired sequences have positive taxonomy_ids
#             is_paired = int(taxonomy_id) > 0

#         # Skip if we've reached the maximum for this sequence type
#         if (is_paired and max_paired_seqs is not None) and (paired_count >= max_paired_seqs):
#             continue
#         if (not is_paired and max_unpaired_seqs is not None) and (
#             unpaired_count >= max_unpaired_seqs
#         ):
#             continue

#         # Skip if duplicate sequence
#         str_seq = line.replace("-", "").upper()
#         if str_seq in visited:
#             continue
#         visited.add(str_seq)

#         # Process sequence
#         residue = []
#         deletion = []
#         count = 0
#         res_idx = 0

#         for c in line:
#             if c != "-" and c.islower():
#                 count += 1
#                 continue
#             token = const.prot_letter_to_token[c]
#             token = const.token_ids[token]
#             residue.append(token)
#             if count > 0:
#                 deletion.append((res_idx, count))
#                 count = 0
#             res_idx += 1

#         res_start = len(residues)
#         res_end = res_start + len(residue)
#         del_start = len(deletions)
#         del_end = del_start + len(deletion)

#         sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
#         residues.extend(residue)
#         deletions.extend(deletion)

#         # Update sequence counts
#         if is_paired:
#             paired_count += 1
#         else:
#             unpaired_count += 1

#         seq_idx += 1

#     # Create MSA object
#     msa = MSA(
#         residues=np.array(residues, dtype=MSAResidue),
#         deletions=np.array(deletions, dtype=MSADeletion),
#         sequences=np.array(sequences, dtype=MSASequence),
#     )

#     return msa


def parse_csv(
    path: Path,
    max_paired_seqs: Optional[int] = None,
    max_unpaired_seqs: Optional[int] = None,
    bottom_k: bool = False,
) -> MSA:
    """Process a CSV file with paired/unpaired sequence filtering.

    Parameters
    ----------
    path : Path
        The path to the CSV file.
    max_paired_seqs : int, optional
        The maximum number of paired sequences to include.
    max_unpaired_seqs : int, optional
        The maximum number of unpaired sequences to include.
    bottom_k : bool, default=False
        If True, select the last k sequences of each type.
        If False, select the first k sequences of each type.

    Returns
    -------
    MSA
        The MSA object.
    """
    # Read file
    data = pd.read_csv(path)
    # Check columns
    if tuple(sorted(data.columns)) != ("key", "sequence"):
        msg = "Invalid CSV format, expected columns: ['sequence', 'key']"
        raise ValueError(msg)

    if not bottom_k:
        # Original top-k sampling implementation
        visited = set()
        sequences = []
        deletions = []
        residues = []
        seq_idx = 0
        paired_count = 0
        unpaired_count = 0

        for line, key in zip(data["sequence"], data["key"]):
            line: str
            line = line.strip()  # noqa: PLW2901
            if not line:
                continue

            # Get taxonomy, if annotated
            taxonomy_id = -1
            is_paired = False
            if (str(key) != "nan") and (key is not None) and (key != ""):
                taxonomy_id = key
                # Assuming paired sequences have positive taxonomy_ids
                is_paired = int(taxonomy_id) > 0

            # Skip if we've reached the maximum for this sequence type
            if (is_paired and max_paired_seqs is not None) and (paired_count >= max_paired_seqs):
                continue
            if (not is_paired and max_unpaired_seqs is not None) and (
                unpaired_count >= max_unpaired_seqs
            ):
                continue

            # Skip if duplicate sequence
            str_seq = line.replace("-", "").upper()
            if str_seq in visited:
                continue
            visited.add(str_seq)

            # Process sequence
            residue = []
            deletion = []
            count = 0
            res_idx = 0
            for c in line:
                if c != "-" and c.islower():
                    count += 1
                    continue
                token = const.prot_letter_to_token[c]
                token = const.token_ids[token]
                residue.append(token)
                if count > 0:
                    deletion.append((res_idx, count))
                    count = 0
                res_idx += 1

            res_start = len(residues)
            res_end = res_start + len(residue)
            del_start = len(deletions)
            del_end = del_start + len(deletion)
            sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
            residues.extend(residue)
            deletions.extend(deletion)

            # Update sequence counts
            if is_paired:
                paired_count += 1
            else:
                unpaired_count += 1
            seq_idx += 1
    else:
        # Bottom-k sampling implementation
        visited = set()
        all_seqs = []  # Will hold (is_paired, line, taxonomy_id) tuples

        for line, key in zip(data["sequence"], data["key"]):
            line: str
            line = line.strip()
            if not line:
                continue

            # Get taxonomy, if annotated
            taxonomy_id = -1
            is_paired = False
            if (str(key) != "nan") and (key is not None) and (key != ""):
                taxonomy_id = key
                # Assuming paired sequences have positive taxonomy_ids
                is_paired = int(taxonomy_id) > 0

            # Skip if duplicate sequence
            str_seq = line.replace("-", "").upper()
            if str_seq in visited:
                continue
            visited.add(str_seq)

            # Store valid sequence
            all_seqs.append((is_paired, line, taxonomy_id))

        # Separate paired and unpaired sequences
        paired_seqs = [seq for seq in all_seqs if seq[0]]
        unpaired_seqs = [seq for seq in all_seqs if not seq[0]]

        # Apply bottom k sampling - take the last k sequences
        if max_paired_seqs is not None:
            paired_seqs = paired_seqs[-max_paired_seqs:] if paired_seqs else []
        if max_unpaired_seqs is not None:
            unpaired_seqs = unpaired_seqs[-max_unpaired_seqs:] if unpaired_seqs else []

        # Combine selected sequences in their original order
        selected_seqs = []
        for seq in all_seqs:
            if (seq[0] and seq in paired_seqs) or (
                not seq[0] and seq in unpaired_seqs
            ):  # Is paired and in our selected paired sequences
                selected_seqs.append(seq)

        # Process the selected sequences to build the MSA
        sequences = []
        deletions = []
        residues = []
        seq_idx = 0

        for is_paired, line, taxonomy_id in selected_seqs:
            # Process sequence
            residue = []
            deletion = []
            count = 0
            res_idx = 0
            for c in line:
                if c != "-" and c.islower():
                    count += 1
                    continue
                token = const.prot_letter_to_token[c]
                token = const.token_ids[token]
                residue.append(token)
                if count > 0:
                    deletion.append((res_idx, count))
                    count = 0
                res_idx += 1

            res_start = len(residues)
            res_end = res_start + len(residue)
            del_start = len(deletions)
            del_end = del_start + len(deletion)
            sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
            residues.extend(residue)
            deletions.extend(deletion)
            seq_idx += 1

    # Create MSA object
    msa = MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )
    return msa
