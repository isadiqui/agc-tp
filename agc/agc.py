#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""OTU clustering"""

import argparse
import sys
import os
import gzip
import statistics
import textwrap
from pathlib import Path
from collections import Counter
from typing import Iterator, Dict, List
# https://github.com/briney/nwalign3
# ftp://ftp.ncbi.nih.gov/blast/matrices/
import numpy as np
if not hasattr(np, "int"):
    np.int = int  
import nwalign3 as nw

__author__ = "Imane SADIQUI"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Imane SADIQUI"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Imane SADIQUI"
__email__ = "imane.sadiqui@etu.u-paris.fr"
__status__ = "Developpement"



def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments(): # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, usage=
                                     "{0} -h"
                                     .format(sys.argv[0]))
    parser.add_argument('-i', '-amplicon_file', dest='amplicon_file', type=isfile, required=True, 
                        help="Amplicon is a compressed fasta file (.fasta.gz)")
    parser.add_argument('-s', '-minseqlen', dest='minseqlen', type=int, default = 400,
                        help="Minimum sequence length for dereplication (default 400)")
    parser.add_argument('-m', '-mincount', dest='mincount', type=int, default = 10,
                        help="Minimum count for dereplication  (default 10)")
    parser.add_argument('-o', '-output_file', dest='output_file', type=Path,
                        default=Path("OTU.fasta"), help="Output file")
    return parser.parse_args()



def read_fasta(amplicon_file: Path, minseqlen: int) -> Iterator[str]:
    """Read a compressed fasta and extract all fasta sequences.

    :param amplicon_file: (Path) Path to the amplicon file in FASTA.gz format.
    :param minseqlen: (int) Minimum amplicon sequence length
    :return: A generator object that provides the Fasta sequences (str).
    """
    current_seq = []
    with gzip.open(amplicon_file, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    seq = "".join(current_seq)
                    if len(seq) >= minseqlen:
                        yield seq
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            seq = "".join(current_seq)
            if len(seq) >= minseqlen:
                yield seq


def dereplication_fulllength(amplicon_file: Path, minseqlen: int, mincount: int) -> Iterator[List]:
    """Dereplicate the set of sequence

    :param amplicon_file: (Path) Path to the amplicon file in FASTA.gz format.
    :param minseqlen: (int) Minimum amplicon sequence length
    :param mincount: (int) Minimum amplicon count
    :return: A generator object that provides a (list)[sequences, count] of sequence with a count >= mincount and a length >= minseqlen.
    """
    counts = Counter(read_fasta(amplicon_file, minseqlen))
    for seq, count in counts.most_common():
        if count >= mincount:
            yield [seq, count]

def get_identity(alignment_list: List[str]) -> float:
    """Compute the identity rate between two sequences

    :param alignment_list:  (list) A list of aligned sequences in the format ["SE-QUENCE1", "SE-QUENCE2"]
    :return: (float) The rate of identity between the two sequences.
    """
    seq1, seq2 = alignment_list[0], alignment_list[1]
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return (matches / len(seq1)) * 100.0



def abundance_greedy_clustering(
    amplicon_file: Path,
    minseqlen: int,
    mincount: int,
    chunk_size: int,
    kmer_size: int,
) -> List:
    """Compute an abundance greedy clustering regarding sequence count and identity.
    Identify OTU sequences.

    :param amplicon_file: (Path) Path to the amplicon file in FASTA.gz format.
    :param minseqlen: (int) Minimum amplicon sequence length.
    :param mincount: (int) Minimum amplicon count.
    :param chunk_size: (int) A fournir mais non utilise cette annee
    :param kmer_size: (int) A fournir mais non utilise cette annee
    :return: (list) A list of all the [OTU (str), count (int)] .
    """
    del chunk_size, kmer_size  # Non utilises cette annee
    matrix_path = str(Path(__file__).parent / "MATCH")
    otu_list = []

    for seq, count in dereplication_fulllength(amplicon_file, minseqlen, mincount):
        if not otu_list:
            otu_list.append([seq, count])
            continue

        is_new_otu = True
        for otu_seq, _ in otu_list:
            alignment = nw.global_align(
                seq,
                otu_seq,
                matrix=matrix_path,
                gap_open=-1,
                gap_extend=-1,
            )
            identity = get_identity(alignment)
            if identity > 97.0:
                is_new_otu = False
                break

        if is_new_otu:
            otu_list.append([seq, count])

    return otu_list


def write_OTU(OTU_list: List, output_file: Path) -> None:
    """Write the OTU sequence in fasta format.

    :param OTU_list: (list) A list of OTU sequences
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, "w", encoding="utf-8") as handle:
        for idx, (seq, count) in enumerate(OTU_list, start=1):
            handle.write(f">OTU_{idx} occurrence:{count}\n")
            handle.write(textwrap.fill(seq, width=80) + "\n")


#==============================================================
# Main program
#==============================================================
def main():  # pragma: no cover
    """Main program function."""
    args = get_arguments()
    otus = abundance_greedy_clustering(
        args.amplicon_file,
        args.minseqlen,
        args.mincount,
        chunk_size=100,
        kmer_size=8,
    )
    write_OTU(otus, args.output_file)


if __name__ == '__main__':
    main()
