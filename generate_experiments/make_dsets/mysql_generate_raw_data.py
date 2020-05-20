"""
Copyright (C) 2013-2015 SeaChange Pharmaceuticals, Michael Mysinger & Elena Caceres
Prepare activity list from ChEMBL database files and optional target designations.
input: 
output: activity_list.txt
        "doc_id     date     Chembl_molecule_id     ChEMBL_Target_ID     pref_name
            activity     relationship"
This activity normalization code was originally started by 
Michael Keiser in 2010. Major additions were later done by:
KEISER - 20100127, 20101008; BET - 20120725; MYSINGER - 20120517, 20131008
CACERES - 20150908
"""
import argparse
import getpass
import logging
import MySQLdb
import sys
import tempfile

from seacore.util.util import gopen

__authors__ = "Elena Caceres, Michael Mysinger"
__credits__ = []
__email__ = "ecaceres@keiserlab.org"


MAX_AFFINITY = 10000   # 10uM in nM units
MIN_AFFINITY = 1.0e-5  # 10fM in nM units

# convenience constants for bucketValue()
UNITS2NM = {'M': 1e9, 'mM': 1e6, 'uM': 1e3, 'nM': 1, 'pM': 1e-3, 'fM': 1e-6,
            'fmol/ml': 1e-3, 'pmol/ml': 1, 'nmol/ml': 1e3, 'umol/ml': 1e9,
            'mmol/ml': 1e12, 'M-1': 1e-9, 'NULL': 1}

STD_RELATIONS = ['=']
BOUNDING_RELATIONS = ['<', '<=']
# MMM - Dropped '~'
#   Only <200 were normalizable in ChEMBL17, so I do not think
#   they are worth the uncertainty of including them.

EXCLUDE_SOURCES = ['7', '15']
DRUG_MAT_SOURCE = ['15']
# Exclude PubChem (src_id = 7), due to many customer concerns about this data
# Exclude DrugMatrix (src_id = 15), for use as an external validation set

# Weak confidences and curations are removed by the strong flag
NON_PROTEIN_CONFIDENCE = ['0', '1', '2', '3']
HOMOLOGOUS_CONFIDENCE = ['4', '6', '8']
WEAK_CONFIDENCE = NON_PROTEIN_CONFIDENCE + HOMOLOGOUS_CONFIDENCE

WEAK_CURATION = ['Autocuration']


class ScriptError(StandardError):
    def __init__(self, msg, value=99):
        self.value = value
        Exception.__init__(self, msg)


def minus_log(x):
    return 10**(9-x)


def plus_log(x):
    # Unfortunately these are more often the wrong sign than the right one
    if x > 2:
        x = -x
    return 10**(9+x)


def identity(x):
    return x

FUNCTIONAL_TYPES = {'AC50': identity,
                    'GI50': identity,
                    'LD50': identity,
                    'ED50': identity,
                    'ID50': identity,
                    "pD'2": minus_log,
                    'pD2': minus_log,
                    'pA2': minus_log,
                    'Log AC50': plus_log,
                    'Log GI50': plus_log,
                    'Log LD50': plus_log,
                    '-Log AC50': minus_log,
                    '-Log GI50': minus_log,
                    '-Log LD50': minus_log}

BOUNDING_TYPES = {
             # Below all mean that affinity is at least this good
             'IC60': identity,
             'IC70': identity,
             'IC80': identity,
             'IC90': identity,
             'IC95': identity,
             'IC99': identity,
             'Log IC60': plus_log,
             'Log IC70': plus_log,
             'Log IC80': plus_log,
             'Log IC90': plus_log,
             'Log IC95': plus_log,
             'Log IC99': plus_log,
             '-Log IC60': minus_log,
             '-Log IC70': minus_log,
             '-Log IC80': minus_log,
             '-Log IC90': minus_log,
             '-Log IC95': minus_log,
             '-Log IC99': minus_log,
             }

# Assume log(Ki) always in log(molar) units and that NULL means no convert
STD_TYPES = {'Ki': identity,
             'Kd': identity,
             'IC50': identity,
             'pKi': minus_log,
             'pKd': minus_log,
             'pIC50': minus_log,
             '-Log Ki': minus_log,
             '-Log Kd': minus_log,
             '-Log IC50': minus_log,
             '-Log KD': minus_log,
             'Log 1/Ki': minus_log,
             'Log 1/Kd': minus_log,
             'Log 1/IC50': minus_log,
             'log(1/Ki)': minus_log,
             'log(1/Kd)': minus_log,
             'log(1/IC50)': minus_log,
             'Log Ki': plus_log,
             'Log Kd': plus_log,
             'Log IC50': plus_log,
             'logKi': plus_log,
             'logKd': plus_log,
             'logIC50': plus_log,
             # Agonist binding types
             'EC50': identity,
             'pEC50': minus_log,
             '-Log EC50': minus_log,
             'Log 1/EC50': minus_log,
             'log(1/EC50)': minus_log,
             'Log EC50': plus_log,
             'logEC50': plus_log,
             }


def normalize_affinity(value, units, stdtype, relation, verbose=False,
                       all_binding_data=False):
    """Normalize an affinity to nM."""
    if not all_binding_data:
        if relation not in STD_RELATIONS:
            if verbose:
                logging.warn("Skipping relation operator %s" % relation)
            return None
    if stdtype not in STD_TYPES:
        if verbose:
            logging.warn("Skipping relation type %s" % stdtype)
        return None
    if units not in UNITS2NM:
        if verbose:
            logging.warn("Skipping unknown unit type %s" % units)
        return None
    try:
        numeric = float(value)
    except ValueError:
        if verbose:
            logging.warn("Skipped %s, as it does not seem numeric" % value)
        return None
    conversion_function = STD_TYPES[stdtype]
    try:
        converted = conversion_function(numeric)
    except OverflowError:
        if verbose:
            logging.warn("Skipped %s of type %s, as the value overflowed" % (
                    value, stdtype))
        return None
    return converted * UNITS2NM[units]


def activity_list(in_f, min_affinity=MIN_AFFINITY, max_affinity=MAX_AFFINITY,
                  verbose=False, functional=False, bounding=False,
                  strong=False, all_binding_data=False, drugmat=False):
    """Prepare activities list from ChEMBL exported activities.txt."""
    if functional:
        logging.info('\tEnabling functional assay types.')
        STD_TYPES.update(FUNCTIONAL_TYPES)
    if bounding:
        logging.info('\tEnabling bounding assay types and relations.')
        STD_TYPES.update(BOUNDING_TYPES)
        STD_RELATIONS.extend(BOUNDING_RELATIONS)
    if strong:
        logging.info('\tRemoving weak curations and confidence scores.')
    if all_binding_data:
        logging.info('\tOutputing all binding data.')
        if max_affinity == MAX_AFFINITY:
            # For all data use 100 mM default, but can be explicitly changed
            max_affinity = 1.0e8
    logging.info('\tOnly writing affinities between %g and %g nM' % (min_affinity, max_affinity))
    logging.info('\tNormalizing activities.')
    # Handle headers
    in_f.next()
    yield "\t".join(("doc_id",
                     "year",
                     "ChEMBL_Target_ID",
                     "ChEMBL_Molecule_ID",
                     "Chembl_Target_pref_name",
                     "Activity (nM)",
                     "Relation",
                     "SMILES")) + "\n"

    for line in in_f:
        splits = line.strip().split('\t')
        src_id = splits[7]
        if drugmat:
            if src_id not in DRUG_MAT_SOURCE:
                if verbose:
                    logging.warn("Skipped line containing source id %s" % src_id)
                continue
        else:
            if src_id in EXCLUDE_SOURCES:
                if verbose:
                    logging.warn("Skipped line containing source id %s" % src_id)
                continue
        if strong:
            curation = splits[9]
            if curation in WEAK_CURATION:
                if verbose:
                    logging.warn("Skipped weak curation %s" % curation)
                continue
            confidence = splits[8]
            if confidence in WEAK_CONFIDENCE:
                if verbose:
                    logging.warn("Skipped weak confidence %s" % confidence)
                continue
        units, value, stdtype, relation = splits[2:6]
        affinity = normalize_affinity(value, units, stdtype, relation,
                                      verbose=verbose,
                                      all_binding_data=all_binding_data)
        if affinity is None:
            continue
        if affinity < min_affinity or affinity > max_affinity:
            continue
        chembl_tid = splits[0]
        chembl_mid = splits[1]
        doc_id = splits[6]
        smiles = splits[10]
        pref_name = splits[11]
        year = splits[12]
        yield "\t".join((doc_id,
                         year,
                         chembl_tid,
                         chembl_mid,
                         pref_name,
                         "%.4e" % affinity,
                         relation,
                         smiles)) + "\n"
    return


def handler(infile, outfile=None, **kwargs):
    """I/O handling for the script."""
    in_f = gopen(infile, "r")
    if outfile is None:
        out_f = sys.stdout
    else:
        out_f = gopen(outfile, "w")
    try:
        try:
            for result in activity_list(in_f, **kwargs):
                out_f.write(result)
        except ScriptError, message:
            logging.error(message)
            return message.value
    finally:
        in_f.close()
        out_f.close()
    return 0


def make_data(username, opt_targets=None, cutoff=None):
    """
    Builds dataset for ChEMBL20 neural networks saves to a tmp file
    Parameters
    ----------
    username : str
        username to log into database
    opt_targets : str (default None)
        optional .txt target file where gene ids/preferred names are pulled out 
        FORMAT: tab separated file with header
        (e.g.   Gene_ID name
                CP3A4   cytochrome p450 3A4
                ADRB1   beta-1 adrenergic receptor
                ...
        )
    Returns
    -------
    tmp_dest : str
        pathname where tmp file was stored
    """
    host = "localhost"
    database = "chembl_20"
    # prompt for password
    p = getpass.getpass("Enter password: ")
    # open database connection
    db = MySQLdb.connect(host=host, user=username, passwd=p, db=database)
    print("Connected to %s as %s@%s" % (database, username, host))

    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    tmp_dest = ("%s.csv" % tempfile.mktemp())

    if cutoff is None:
        cutoff_join = " "
        cutoff_and = " "
    else:
        cutoff_join = (" INNER JOIN compound_properties AS cp "
                      " ON cp.molregno = act.molregno ")
        cutoff_and = (" AND cp.full_mwt <= %.2f " % (cutoff)) 
    # =========================================================================
    # No target file supplied.
    # =========================================================================
    if opt_targets is None:
        # ---------------------------------------------------------------------
        # Define target dictionary to all chembl
        # ---------------------------------------------------------------------
        td_query = ("INNER JOIN target_dictionary  AS td  "
                    "ON td.tid=assays.tid  ")
    # =========================================================================
    # Target file supplied.
    # =========================================================================
    else:
        # ---------------------------------------------------------------------
        # Load temp class names
        # ---------------------------------------------------------------------
        print("Creating targets table...")
        # temp table to load class names
        query = ("CREATE TEMPORARY TABLE tmp_targets ("
                 "Gene_ID varchar(20) NOT NULL,"
                 "name varchar(200) NOT NULL);")
        # execute SQL query using execute() method.
        print query
        cursor.execute(query)

        print("Loading targets to table...")
        # load data into temporary table
        query = ("LOAD DATA LOCAL INFILE "
                 "'%s' "
                 "INTO TABLE tmp_targets "
                 "FIELDS TERMINATED BY '\t' "
                 "LINES TERMINATED BY '\n' "
                 "IGNORE 1 LINES;" % opt_targets)
        print query
        cursor.execute(query)

        # ---------------------------------------------------------------------
        # Define target dictionary to only those in targets table
        # ---------------------------------------------------------------------
        # String only. To be executed later.
        td_query = ("INNER JOIN (SELECT * FROM tmp_targets AS tt "
                    "INNER JOIN target_dictionary  "
                    "ON target_dictionary.pref_name LIKE tt.name) AS td "
                    "ON td.tid=assays.tid  ")

    # =========================================================================
    # Print data to tmp file!
    # =========================================================================
    print("Merge all target data and export...")

    query = ("SELECT 'ChEMBL_Target_ID', "
                    "'ChEMBL_Molecule_ID', "
                    "'standard_units',  "
                    "'standard_value', "
                    "'standard_type', "
                    "'standard_relation', "
                    "'doc_id', "
                    "'src_id',  "
                    "'confidence_score', "
                    "'curated_by', "
                    "'canonical_smiles', "
                    "'pref_name', "
                    "'year' "
            "UNION ALL "
            "(SELECT tdchemblid, "
                    "mdchemblid, "
                    "IFNULL(standard_units, 'NULL'), "
                    "IFNULL(standard_value, 'NULL'), "
                    "IFNULL(standard_type, 'NULL'), "
                    "IFNULL(standard_relation, 'NULL'), "
                    "IFNULL(doc_id, 'NULL'), "
                    "src_id, "
                    "IFNULL(confidence_score, 'NULL'), "
                    "IFNULL(curated_by, 'NULL'), "
                    "IFNULL(canonical_smiles, 'NULL'), "
                    "pref_name, "
                    "IFNULL(year, 'NULL') "
            "INTO OUTFILE '%s' "
                    "FIELDS TERMINATED BY '\t'  "
                    "LINES TERMINATED BY '\n' "
            "FROM ( "
            "SELECT DISTINCT td.chembl_id as tdchemblid,  "
                            "td.pref_name,  "
                            "cs.canonical_smiles,  "
                            "md.chembl_id as mdchemblid,  "
                            "act.standard_units,  "
                            "act.standard_value,  "
                            "act.standard_type,  "
                            "act.standard_relation,  "
                            "act.doc_id,  "
                            "source.src_id,  "
                            "assays.confidence_score,  "
                            "assays.curated_by,  "
                            "d.year "
            "FROM assays  "
            "%s"
            "INNER JOIN activities AS act  "
            "ON act.assay_id=assays.assay_id  "
            "INNER JOIN docs as d "
            "ON d.doc_id=act.doc_id "
            "INNER JOIN molecule_dictionary AS md  "
            "ON md.molregno=act.molregno  "
            "INNER JOIN compound_structures AS cs "
            "ON cs.molregno = act.molregno  "
            "%s"
            "INNER JOIN source  "
            "ON source.src_id=assays.src_id "
            "WHERE assays.assay_type='B'  "
            "AND assays.confidence_score>=4  "
            "AND assays.relationship_type IN ('D', 'H')  "
            "AND td.species_group_flag=0  "
            "%s"
            "AND td.target_type  "
            "IN ('SINGLE PROTEIN', 'PROTEIN COMPLEX', 'PROTEIN FAMILY')) "
            "AS intermed "
            "ORDER BY doc_id, mdchemblid);" % (tmp_dest, td_query, cutoff_join, cutoff_and))
    print query
    cursor.execute(query)

    # =========================================================================
    # Close Database connection
    # =========================================================================
    db.close()
    return tmp_dest


def run(username, out_file, opt_targets=None, cutoff=None, **kwargs):
    """
    Builds dataset for ChEMBL20 neural networks, includes filtering and saving
    of files necessary for
    Parameters
    ----------
    username : str
        username to log into database
    out_file : str
        filename to save final files to
    opt_targets : str (default None)
        optional .txt target file where gene ids/preferred names are pulled out 
        FORMAT: tab separated file with header
        (e.g.   Gene_ID name
                CP3A4   cytochrome p450 3A4
                ADRB1   beta-1 adrenergic receptor
                ...
        )
    Returns
    -------
    """

    # Pull data from ChEMBL, make a tmp file to hold it.
    tmp_dest = make_data(username, opt_targets, cutoff)

    # send file to handler to print final output
    return handler(tmp_dest, out_file, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        "Get small dataset for the first test of multi-label"
        "data MLP from mySQL")
    parser.add_argument('username', type=str,
                        help="""Username to access chembl_20.""")
    parser.add_argument('out_file', type=str,
                        help="""Filename to save data.""")
    parser.add_argument('-t', '--opt_targets', type=str, 
                        default=None,
                        help="""Path to optional list of targets to use""")
    parser.add_argument('-c', '--cutoff', type=float,
                        default=None,
                        help="""Full_mwt cutoff for a ChEMBL Molecule. See compound_properties
                        in chembl20""")
    parser.add_argument('-m', '--max_affinity', type=float, 
                        default=MAX_AFFINITY,
                        help="""Only include affinities up to X nM.""")
    parser.add_argument('-f', '--functional', action='store_true',
                        help="""Include functional data types like ED50 and 
                                variants.""")
    parser.add_argument('-b', '--bounding', action='store_true',
                        help="""Include activities that are only upper bounds.""")
    parser.add_argument('-s', '--strong', action='store_true',
                        help="""Use only strongest ChEMBL data 
                        [direct confidence scores, manual curation].""")
    parser.add_argument('-a', '--all_binding_data', action='store_true',
                        help="""Output all binding affinity data (e.g. for machine
                            learning). Overrides many other options.""")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="""Be more 'noisy' about info, warnings, and errors.""")
    parser.add_argument('-d', '--drugmat', action='store_true', help="""Set to true to get drug
    matrix only.""")

    params = parser.parse_args()
    kwargs = dict(params._get_kwargs())
    username = kwargs.pop('username')
    out_file = kwargs.pop('out_file')

    # run the program
    run(username, out_file, **kwargs)

# end __main__
