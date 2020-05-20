import os
import sys
import gzip
import logging
import argparse
from common.h5py_loading import load_target_map
from common.util import config_logging, write_args_file, ready_dir


def prediction_iterator(prediction_file, cutoff):
    logging.info("Reading predictions from {}".format(prediction_file))
    gopen = gzip.open if prediction_file[-3:] == '.gz' else open
    with gopen(prediction_file, 'rb') as f:
        f.next()
        try:
            for row in f:
                row = row.split(',')
                if len(row) > 2:
                    if float(row[3]) < cutoff:
                        yield (row[0], row[1])
                else:
                    logging.warn("skipping invalid row\n{}".format(''.join(row)))
        except IOError as e:
            logging.error("IOError encountered in prediction file:\n\t{}".format(e))
            raise StopIteration()


def compound_target_indices_iterator(target_map_file, prediction_file, cutoff):
    logging.info("Reading target map from {}".format(target_map_file))
    target_map = load_target_map(target_map_file)
    for cpd, tgt in prediction_iterator(prediction_file, cutoff):
        if tgt in target_map:
            yield (int(cpd), target_map[tgt])
        else:
            logging.debug("no mapping found for {}".format(tgt))


def generate_blacklist_and_save(target_map_file, prediction_file, cutoff, output_file):
    gopen = gzip.open if output_file[-3:] == '.gz' else open
    logging.info("Saving blacklist to {}".format(output_file))
    with gopen(output_file, 'wb') as f:
        for compound_indx, target_indx in compound_target_indices_iterator(target_map_file, prediction_file, cutoff):
            f.write("{},{}\n".format(compound_indx, target_indx))


def main():
    usage = 'usage: %prog target_index.pkl sea_prediction.csv sea_blacklist.csv.gz 1.0e-5'
    parser = argparse.ArgumentParser(usage, fromfile_prefix_chars='@')
    parser.add_argument(dest='target_map_file', metavar='str',
                        help='Target map file for an hdf5 file',
                        action='store')
    parser.add_argument(dest='prediction_file', metavar='str',
                        help='SEA prediction output where keys are indices of unique fingerprints in an hdf5 file',
                        action='store')
    parser.add_argument(dest='output_file', metavar='str',
                        help='Output file', action='store')
    parser.add_argument(dest='cutoff', type=float,
                        help='SEA predictions with p-values below this cutoff are added to the blacklist',
                        action='store')
    args = parser.parse_args()
    output_directory = ready_dir(os.path.dirname(args.output_file))
    config_logging(output_directory=output_directory,
                   file_logging_level=logging.DEBUG)
    write_args_file(parser, output_directory)
    return generate_blacklist_and_save(args.target_map_file, args.prediction_file, args.cutoff, args.output_file)


if __name__ == '__main__':
    sys.exit(main())
# end __main__

