#!/usr/bin/env python
#
# KEISER 2014-05-15
#
# quick python test script to generate rdkit ecfp (morgan) fp with
# multiprocessing

from optparse import OptionParser
from multiprocessing import cpu_count, Pool
# from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdmolfiles import ForwardSDMolSupplier
import gzip
import csv
from os import listdir
from os.path import isfile, join

DEF_OUTFILE = 'out.fp'
RADIUS = 2
NBITS = 1024


# args helper
def get_fingerprint_mp(args):
    mol = args
    # mol = MolFromSmiles(smi)
    # if mol is None:
    #     return (None, None)
    fp = GetMorganFingerprintAsBitVect(mol, RADIUS, NBITS)
    return (fp.ToBitString())


def main(sdfdir, outfile):
    num_cpus = cpu_count()
    ppool = Pool(num_cpus)
    outfile = str(outfile + '.csv.gz')

    onlyfiles = [join(sdfdir, f) for f in listdir(
        sdfdir) if isfile(join(sdfdir, f)) and f.endswith('.sdf.gz')]

    with gzip.open(outfile, 'wb') as out:
        writer = csv.writer(out)
        for molfile in onlyfiles:
            f = gzip.open(molfile)
            reader = ForwardSDMolSupplier(f)
            args = (mol for mol in reader)
            fps = ppool.map(get_fingerprint_mp, args, chunksize=100)

            for fp in fps:
                writer.writerow([fp])

    return
# end main


if __name__ == '__main__':

    usage = 'usage: prog [options] "molecules.sdf.gz directory"'
    parser = OptionParser(usage)
    parser.add_option('-o', '--outfile', dest='outfile', metavar='FILE',
                      help='Output to FILE (default default)',
                      action='store', default=DEF_OUTFILE)
    options, args = parser.parse_args()

    try:
        arg1, = args
    except:
        parser.error("Incorrect number of arguments")

    main(arg1, options.outfile)

# end __main__
