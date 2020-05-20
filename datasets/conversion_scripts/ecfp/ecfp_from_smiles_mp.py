import argparse
from rdkit.Chem import rdMolDescriptors as rdMD
from multiprocessing import cpu_count, Pool
from rdkit.Chem import MolFromSmiles
#from rdkit.Chem.Draw import SimilarityMaps as CSM
from rdkit import Chem
import standardiser
from standardiser import standardise
from standardiser.utils import StandardiseException, errors
import gzip
import csv

__author__ = "Elena Caceres"
__credits__ = []
__email__ = "ecaceres@keiserlab.org"
"""Read in a file containing mid\tsmiles with header file and create a new file in the format mid\tfingerprint"""

MID_COL = 0
SMI_COL = 1
SKIP_HEADER = False
DELIMITER_IN = '\t'
DELIMITER_OUT = '\t'


def fingerprint_mp_wrapper(args):
    mid, smiles, kwargs = args
    fp = get_sim_fingerprint(smiles, **kwargs)
    if fp is not None:
        return((fp.ToBitString(), mid))
    else:
        return((fp, mid))


def get_sim_fingerprint(smiles, **kwargs):
    """Calculates the Morgan fingerprint

    Parameters
    ----------
    smiles : str
        SMILES string to fingerprint
    **kwargs
        radius : int (default 2)
            the maximum radius
        nBits : int (default 1024)
            size of the bit vector (only for fpType = 'bv')

        fpType : str (default 'bv')
            the type of Morgan fingerprint -- 'count' or 'bv' (bitvector)

        useFeatures : bool (default False)
            if false - ConnectivityMorgan, if true: FeatureMorgan
        useChirality : bool (default False)
            if False - Don't use Chirality in ECFP hash, if True - Do.
    Returns
    -------
    rdkit.DataStructs.cDataStructs.ExplicitBitVect : 2D ndarray the Morgan
    fingerprint with the environments of atomId removed
    """

    kwargs.setdefault('radius', 2)
    kwargs.setdefault('nBits', 4096)
    kwargs.setdefault('fpType', 'bv')
    kwargs.setdefault('useFeatures', False)
    kwargs.setdefault('useChirality', False)
    mol = MolFromSmiles(smiles)
    radius = kwargs.pop('radius')
    kwargs.pop('fpType')
    # standardize mol
    if mol is None:
        print("\tCould not create mol object from %s,"
              " because mol == Nonetype" % smiles)
        return None
    # Filter for mols that can't be standardised
    try:
        std_mol = standardise.apply(mol)
    except StandardiseException, e:
        print("\tUsed unstandardised mol. Could not create"
              "standardised ECFP from %s, because of "
              "standardise exception error: %s" % (smiles, e))
        # make note, but return unstandardized mol. We'll just count this as
        # random noisei
        return rdMD.GetMorganFingerprintAsBitVect(mol, radius, **kwargs)
    return rdMD.GetMorganFingerprintAsBitVect(std_mol, radius, **kwargs)


def run(in_file, out_file, mid_col=MID_COL, smi_col=SMI_COL, skip_header=SKIP_HEADER,
        delimiter_in=DELIMITER_IN, delimiter_out=DELIMITER_OUT, **kwargs):
    """ Makes a file with fingerprint\tmolecule_id from *.smi.gz file

    Parameters
    ----------
    in_file : str
            path to *.smi.gz file
    out_file : str
            path to save final file to *.gz (tab delimited or as defined by delimiter_in)
    mid_col: int
            molecule id column number in in_file
    smi_col : int
            smile column number in in_file
    skip_header: int, bool
            False or number of rows to skip at start of in_file (defaults to False)
    delimiter_in: char
            delimiter of in_file
    delimiter_out: char
            delimiter of out_file
    **kwargs
        radius : int (default 2)
            the maximum radius
        nBits : int (default 1024)
            size of the bit vector (only for fpType = 'bv')
        fpType : str (default 'bv')
            the type of Morgan fingerprint -- 'count' or 'bv' (bitvector)
        useFeatures : bool (default False)
            if false - ConnectivityMorgan, if true: FeatureMorgan
        useChirality : bool (default False)
            if False - Don't use Chirality in fingerprints, if True - Do
    Returns
    -------
    """
    num_cpus = cpu_count()
    ppool = Pool(num_cpus)
    kwargs.setdefault('radius', 2)
    kwargs.setdefault('nBits', 4096)
    #kwargs.setdefault('fpType', 'bv')
    kwargs.setdefault('useFeatures', False)
    kwargs.setdefault('useChirality', False)
    # ==========================================================================
    # read file & create values
    # ==========================================================================
    # read our smiles file
    open_in = gzip.open if in_file.endswith('.gz') else open
    open_out = gzip.open if out_file.endswith('.gz') else open
    with open_in(in_file, 'rb') as f, open_out(out_file, 'wb') as fo:
        reader = csv.reader(f, delimiter=delimiter_in)
        writer = csv.writer(fo, delimiter=delimiter_out)
        # skip header...
        for h in range(skip_header):
            reader.next()
        args = ((row[mid_col], row[smi_col], kwargs)
                for row in reader)
        for fingerprints in ppool.imap_unordered(fingerprint_mp_wrapper, args, chunksize=100):
            if fingerprints[0] != None:
                writer.writerow(fingerprints)
    ppool.close()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(""""Generate fingerprints from *.smi.gz file""",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    parser.add_argument('in_file', type=str,
                        help="""Input file (*.smi.gz) to convert SMILES to ECFP. Must
                                contain header""")
    parser.add_argument('out_file', type=str,
                        help="""path to save final datai (*.smi.gz)""")
    parser.add_argument('-b', '--nBits', type=int, default=4096,
                        help="""Length of fingerprint bitvector after folding
                                (default 1024)""")
    parser.add_argument('-r', '--radius', type=int, default=2,
                        help="""ECFP radius (default 2)""")
    parser.add_argument('-t', '--fpType', type=str, default='bv',
                        help="""CURRENTLY DISABLED. BV only. the type of Morgan fingerprint -- 'count' or
                        'bv' (bitvector, default)""")
    parser.add_argument('-f', '--useFeatures', type=bool, default=False,
                        help="""If false (default) - ConnectivityMorgan, if true:
                                 FeatureMorgan""")
    parser.add_argument('-c', '--useChirality', type=bool, default=False,
                        help="""If False (Default) - Do not use chirality when making
                                fingerprints""")
    parser.add_argument('--mid_col', type=int, default=MID_COL,
                        help="""molecule id column number in in_file""")
    parser.add_argument('--smi_col', type=int, default=SMI_COL,
                        help="""smile column number in in_file""")
    parser.add_argument('--skip_header', type=int, default=SKIP_HEADER,
                        help="""smile column number in in_file""")
    parser.add_argument('--delimiter_in', type=str, default=DELIMITER_IN,
                        help="""delimiter of in_file""")
    parser.add_argument('--delimiter_out', type=str, default=DELIMITER_OUT,
                        help="""delimiter of out_file""")

    params = parser.parse_args()
    kwargs = dict(params._get_kwargs())
    usebv = kwargs.pop('fpType')
    username = kwargs.pop('in_file')
    out_file = kwargs.pop('out_file')

    print run(username, out_file, **kwargs)
