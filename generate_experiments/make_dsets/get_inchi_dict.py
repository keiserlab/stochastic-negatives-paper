from __future__ import print_function
import gzip
import csv
import argparse
from multiprocessing import cpu_count, Pool
from rdkit.Chem import InchiToInchiKey, MolToInchi
from rdkit.Chem.rdmolfiles import MolFromSmiles
from standardiser import standardise
from standardiser.utils import StandardiseException

__author__ = "Elena Caceres"
__credits__ = []
__email__ = "ecaceres@keiserlab.org"
"""Read in a file containing mid\tsmiles with header file and create a new file in the format inchikey, mid, smiles"""



def inchi_key_mp_wrapper(args):
    """
    MP wrapper for calling pains filter
    Parameters
    ----------
    args : list
        mid, smiles
    Returns
    -------
    tuple : str
        inchi key, mid, smiles
    """
    mid, smiles = args
    ikey = get_inchi_key(mid, smiles)
    return((ikey, mid, smiles))


def get_inchi_key(mid, smi):
    """

    Parameters
    ----------
    mid : str
        molecule identifier
    smi : str
        SMILES string for molecule

    Returns
    -------
    inchi_key : str
        inchi key for molecule generated from smi (None if not possible)

    """
    mol = MolFromSmiles(smi)
    std_mol = None
    if mol is None:
        print("\tCould not create mol object from {}: {}, because mol == Nonetype".format(mid, smi))
        # Filter for mols that can't be standardised
        return None
    else:
        try:
            std_mol = standardise.apply(mol)
        except StandardiseException as e:
            print("\tUsed unstandardised mol. Could not create standardized mol from {}: {}, because of standardise "
                  "exception error: {}".format(mid, smi, e))
            std_mol = None
            if mol is not None:
                return InchiToInchiKey(MolToInchi(mol))
    return InchiToInchiKey(MolToInchi(std_mol))


def run(in_file, out_file):
    """
    Parameters
    ----------
    in_file : str
    out_file : str

    Returns
    -------
    """
    num_cpus = cpu_count()
    ppool = Pool(num_cpus)
    # read our smiles file

    with gzip.open(in_file, "r") as fin, gzip.open(out_file, "w") as fo:
        # read and update input inchi keys
        inchi_key_mid_smiles = {}
        reader = csv.reader(fin, delimiter='\t')
        #skip header
        reader.next()
        # write inchis to file.
        writer = csv.writer(fo, delimiter='\t')
        writer.writerow(["INCHI key","mid", "smiles"])
        args = ((mid, smi) for mid, smi in reader)
        for ikey in ppool.imap_unordered(inchi_key_mp_wrapper, args, chunksize=100):
            if ikey[0] != None:
                writer.writerow(ikey)
    ppool.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(""""get INCHI keys for given molecules""")
    parser.add_argument("in_file", type=str,
                        help="""Input file (*.smi.gz) with SMILES to assess whether aggregators present""")
    parser.add_argument("out_file", type=str,
                        help="""path to save final data (*.inchikey.gz)""")

    params = parser.parse_args()
    kwargs = dict(params._get_kwargs())
    inf = kwargs.pop("in_file")
    outf = kwargs.pop("out_file")
    print(run(inf, outf))
