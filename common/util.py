__author__ = 'Nick'
import os
import ast
import sys
import time
import regex
import logging
import platform
import tempfile
import subprocess
import warnings
import pkgutil


def deprecated(func, message=''):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}. {}".format(func.__name__, message),
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


def natural_sort_key(s, _nsre=regex.compile('([0-9]+)')):
    """
    Sort numeric filenames appropriately (eg. x19 before x100). To be used as key in sorted()
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in regex.split(_nsre, s)]


def write_args_file(parser, path):
    """
    Writes command line arguments to a file ([main_module_name]_args.txt) in the provided path.
    Any parameters passed from a file will be read and written to the file as well.
    :param parser: ArgumentsParser instance
    :param path: directory in which to write [module]_args.txt
    """
    if parser.fromfile_prefix_chars is not None:
        args = parser._read_args_from_files(sys.argv[1:])
    else:
        args = sys.argv[1:]

    path = os.path.normpath(path)
    if os.path.isdir(path):
        main_module = os.path.splitext(os.path.split(sys.argv[0])[1])[0]
        path = os.path.join(path, main_module + '_args.txt')
    if os.path.exists(path):
        rename_file_old(path)
    logging.info('writing args to {}'.format(path))
    with open(path, 'w') as f:
        f.write('\n'.join(args))


def ready_dir(path):
    """
    Create any paths that don't already exists and return the normalized path.
    :param path: path to create if necessary
    :return: normalized path
    """
    path = os.path.normpath(path)
    if os.path.exists(path):
        file_to_make_sure_path_is_writable = tempfile.TemporaryFile(dir=path)
        file_to_make_sure_path_is_writable.close()
    else:
        os.makedirs(path)
    return path


def rename_file_old(path, _age=0):
    """
    renames any file to path + '__old' if the '__old' version already exists, renames with next unused age.
    :param path: filepath to rename
    :param _age: don't pass this parameter, let the recursion handle age
    """
    older_name = path + "__old" + ('_' + str(_age) if _age > 0 else '')
    if os.path.exists(older_name):
        rename_file_old(path, _age + 1)
    else:
        os.rename(path, older_name)


def config_logging(stream_logging_level=logging.INFO, output_directory='', file_logging_level=logging.INFO):
    """
    Configures a logging instance to output to std.err and to a file ([main_module_name]_info.log) in the provided path.
    :param stream_logging_level: (optional) defaults to INFO
    :param output_directory: (optional) directory in which to write [main_module_name]_info.log defaults to '.'
    :param file_logging_level: (optional) The logging level of the file handler defaults to INFO
    """
    logging.getLogger().setLevel(logging.NOTSET)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(stream_logging_level)
    logging.getLogger().addHandler(sh)
    config_file_logging(output_directory=output_directory, level=file_logging_level)


def config_file_logging(output_directory, level=logging.INFO):
    """
    Configures a logging instance to output to a file ([main_module_name]_info.log) in the provided path.
    :param output_directory: directory in which to write [main_module_name]_info.log
    :param level: The logging level of the file handler
    """
    main_module = os.path.splitext(os.path.split(sys.argv[0])[1])[0]
    filename = os.path.join(ready_dir(output_directory), main_module + '_{}.log'.format(logging.getLevelName(level)))
    if os.path.exists(filename):
        rename_file_old(filename)
    fh = logging.FileHandler(filename)
    fh.setLevel(level)
    logging.getLogger().addHandler(fh)
    if logging.getLogger().getEffectiveLevel() > level:
        logging.getLogger().setLevel(level)
    logging.info("Logging output of " + sys.argv[0] + " run from " + os.path.abspath(os.path.curdir))


def log_repo(level=logging.INFO, module=None):
    """
    Log the git repo url and the hash of the last commit.
    :param level: logging level of the file handler
    :param module: log git repo of module if module file points to a cloned repo (defaults to current repo)
    """
    if module:
        module_path = pkgutil.get_loader(module).filename
        path, _ = os.path.split(os.path.abspath(module_path))
    else:
        path = get_neural_nets_path()
    logging.log(level, "\n\tRepo Origin: {repo_url}\n\tRepo Hash: {repo_hash}".format(
        repo_url=get_repo_url(path=path),
        repo_hash=get_repo_hash(path=path)))


def log_machine(level=logging.INFO):
    """
    Log machine platform and timestamp.
    :param level: logging level of the file handler
    If none, this repo is used
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    logging.log(level, "\n\tPlatform: {machine_id}\n\tTime: {timestamp}".format(
                            machine_id=platform.platform(),
                            timestamp=timestamp))


def log_repo_and_machine(level=logging.INFO):
    """
    Log the current git repo url and the hash of the last commit along with machine platform and timestamp.
    :param level: logging level of the file handler
    """
    log_repo(level)
    log_machine(level)


def get_neural_nets_path():
    path, _ = os.path.split(os.path.abspath(__file__))
    return os.path.dirname(path)


def get_repo_url(path=get_neural_nets_path()):
    try:
        repo_url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'],
                                           cwd=path).replace('\n', '')
    except (subprocess.CalledProcessError, OSError):
        repo_url = 'unknown_repo_url'
    return repo_url


def get_repo_hash(path=get_neural_nets_path()):
    try:
        repo_hash = subprocess.check_output(['git', 'log', '-n', '1', '--pretty=format:"%H"'],
                                            cwd=path).replace('"', '')
    except (subprocess.CalledProcessError, OSError):
        repo_hash = 'unknown_repo_hash'
    return repo_hash


def find_json_in_string(string):
    return regex.findall("\{(?:[^{}]|(?R))*\}", string)


def function_name_from_serialized_str(serialized_str):
    return regex.sub(' object$', '', regex.sub(r'^function[ ]+', '', serialized_str.lstrip('<')).split(' at ')[0])


def find_in_dict(key, dictionary):
    """Yield occurrences of a key in nested python dictionaries and lists. See: stackoverflow.com/questions/9807634"""
    if hasattr(dictionary, 'iteritems'):
        for k, v in dictionary.iteritems():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in find_in_dict(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in find_in_dict(key, d):
                        yield result


def find_all_in_dict(key, dictionary):
    v = [val for val in find_in_dict(key, dictionary)]
    return v[0] if len(v) == 1 else None if len(v) == 0 else v


def get_build_train_params_from_log(logfile, keys):
    def find(key, dictionary):
        for k, v in dictionary.iteritems():
            if k == key:
                yield v
            elif isinstance(v, dict):
                for result in find(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in find(key, d):
                        yield result

    with open(logfile, 'r') as lf:
        log_string = find_json_in_string(lf.read().replace('\n', ' ').replace('<', "'<").replace('>', ">'")
                                         .replace(" inputs,", " 'inputs',"))
    param_dict = dict(map(lambda i, x: (i, ast.literal_eval(x)), list(enumerate(log_string))))

    return [tuple(find(k, param_dict)) for k in keys]


def get_script_path_from_log(logfile):
    """Parse logfile line that reads 'Logging output of [script_filename] run from [directory]'"""
    start, middle, end = "Logging output of ", " run from ", "\n"
    script_dir, script_name = None, None
    with open(logfile, 'r') as lf:
        for line in lf:
            if line.startswith(start):
                splitline = line.split(middle)
                script_dir, script_name = splitline[1].rstrip(), splitline[0].split(start)[1].lstrip()
                break
    logging.info('found script in log file:\n\t{}\n\t{}'.format((script_dir, script_name), logfile))
    return script_dir, script_name


def get_args_from_argfile(argfile, positions_n_options):
    """
    Read argfile, return args for given list of options and positions. positions_n_options can be a mix
     of integer positions for positional arguments and the option string eg. ['-o', 2, '--input-csv', 4].
     Returns a dict with positions and options found in arguments file as keys.
    :param argfile: path to ArgParse arguments file
    :param positions_n_options: list containing strings and integer positions for arguments to get
    :return: dict with positions_n_options found in argfile as keys and values found as values
    """
    positions_n_options = positions_n_options if isinstance(positions_n_options, list) else [positions_n_options]
    arg_name = {}
    args = {}
    for posopt in positions_n_options:
        if type(posopt) in (tuple, list):
            # if like ('-o', 2, '--out') any -o, 2 or --out will be returned as out
            for a in posopt:
                name = posopt[-1]
                arg_name[a] = name.lstrip('-') if type(name) is str else name
        else:
            arg_name[posopt] = posopt.lstrip('-') if type(posopt) is str else posopt

    with open(argfile, 'r') as af:
        lines = map(str.rstrip, af.readlines())
    for line_num, line in enumerate(lines):
        if line_num in arg_name:
            args[arg_name[line_num]] = line
        if line in arg_name:
            if line_num + 1 >= len(lines) or lines[line_num + 1].startswith('-') or lines[line_num + 1].strip() == '':
                args[arg_name[line]] = True
            else:
                args[arg_name[line]] = lines[line_num + 1]
    logging.debug(args)
    logging.debug(lines)
    return args


def get_logfile_and_argfile_from_dir(dir_path):
    """Return first .log and args.txt files found in dir_path,
    ignore args.txt and .log files from currently running script"""
    logfile, argfile = None, None
    this_scripts_module_name = os.path.splitext(os.path.split(sys.argv[0])[1])[0]
    logging.debug('this_scripts_module_name: {}'.format(this_scripts_module_name))
    logfiles = [f for f in os.listdir(dir_path) if '.log' in f]
    logfiles = [f for f in logfiles if f.endswith('.log')] + [f for f in sorted(logfiles, reverse=True) if not f.endswith('.log')]
    for filename in logfiles:
        if logfile is None and this_scripts_module_name not in filename:
            with open(os.path.join(dir_path, filename), 'r') as lf:
                for line in lf:
                    if regex.findall("^Epoch [0-9]+ took [0-9]+\.[0-9]+s$", line):
                        # found logfile
                        logfile = os.path.join(dir_path, filename)
                        logging.info('found logfile: {}'.format(logfile))
                        break
    if logfile is not None:
        logfile_prefix = os.path.basename(logfile).replace('INFO', '').replace(
            'DEBUG', '').replace('WARN', '')
        logfile_prefix = logfile_prefix[:logfile_prefix.index('.log')]
        for filename in os.listdir(dir_path):
            if argfile is None and filename.startswith(logfile_prefix) and 'args.txt' in filename and this_scripts_module_name not in filename:
                argfile = os.path.join(dir_path, filename)
                logging.info('found argfile: {}'.format(argfile))
    return logfile, argfile
