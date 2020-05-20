from __future__ import print_function
import argparse
import csv
import gzip
import time
from rdkit import rdBase
from rdkit.Chem import MolFromSmiles
from standardiser import standardise
from standardiser.utils import StandardiseException
from rdkit.Chem.Descriptors import MolWt
from multiprocessing import cpu_count, Pool


def mw_mp_wrapper(args):
        """
        MP wrapper for calling pains filter
        Parameters
        ----------
        args : list
            mid, smiles
        Returns
        -------
        tuple : str
            passes, mid, smiles
        """
        mid, smiles, cutoff = args
        # passes is a bool if it passes the MW filter
        passes = mw_pass_fail(mid, smiles, cutoff)
        return ((passes, mid, smiles))


def mw_pass_fail(mid, smi, cutoff):
    """

    Parameters
    ----------
    mid : str
        molecule identifier
    smi : str
        SMILES string for molecule

    Returns
    -------
    bool
        Whether the mol passes the molecular weight cutoff(True) or not(False). Mols failing to generate from smiles
        also return False.

    """

    mol = MolFromSmiles(smi)
    std_mol = None
    if mol is None:
        print("\tCould not create mol object from {}: {}, because mol == Nonetype".format(mid, smi))
        # Filter for mols that can't be standardised
        return False
    else:
        try:
            std_mol = standardise.apply(mol)
        except StandardiseException as e:
            print("\tUsed unstandardised mol. Could not create standardized mol from {}: {}, because of standardise "
                  "exception error: {}".format(mid, smi, e))
            std_mol = None
            if mol is not None:
                return MolWt(mol) < cutoff
    return MolWt(std_mol) < cutoff


def run(in_file, out_file, mw_filtered, cutoff=800.):
    num_cpus = cpu_count()
    ppool = Pool(num_cpus)

    with gzip.open(in_file, "r") as fin, gzip.open(out_file, "w") as fo, gzip.open(mw_filtered, "w") as flog:
        # read and update input inchi keys
        reader = csv.reader(fin, delimiter='\t')
        # skip header
        reader.next()
        # write inchis to file.
        writer = csv.writer(fo, delimiter='\t')
        writer.writerow(["mid", "smiles"])
        flog.write("RDKit version: {}\n".format(rdBase.rdkitVersion))
        flog.write("Time: {}\n\n".format(time.asctime()))
        flog.write("mid\tsmiles\n")
        args = ((mid, smi, cutoff) for mid, smi in reader)
        for mwresults in ppool.imap_unordered(mw_mp_wrapper, args, chunksize=100):
            # if molecule from smiles is under the cutoff weight
            if mwresults[0]:
                writer.writerow([mwresults[1], mwresults[2]])
            else:
                flog.write("{}\t{}\n".format(mwresults[1], mwresults[2]))
    ppool.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(""""Filter PAINS from *.smi.gz file""")
    parser.add_argument('in_file', type=str,
                        help="""Input file (*.smi.gz) to convert SMILES to ECFP. Must
                                contain header""")
    parser.add_argument('out_file', type=str,
                        help="""path to save final data (*.smi.gz)""")
    parser.add_argument('mw_filtered', type=str,
                        help="""path to save log of mw filtered compounds""")
    parser.add_argument('-c', '--cutoff', type=float, default=800,
                        help="""MW cutoff for a given molecule. DEFAULT:800""")

    params = parser.parse_args()
    kwargs = dict(params._get_kwargs())
    inf = kwargs.pop('in_file')
    outf = kwargs.pop('out_file')
    log_file = kwargs.pop('mw_filtered')

    print(run(inf, outf, log_file, **kwargs))
