import os

# Set environment variables to disable multithreading as users will probably
# want to set the number of cores to the max of their computer.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import datetime as dt
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from pcigale.analysis_modules import get_module
from pcigale.managers.observations import ObservationsManager
from pcigale.managers.parameters import ParametersManager
from pcigale.session.configuration import Configuration
from pcigale.utils.console import INFO, WARNING, console
from pcigale.utils.info import Info
from pcigale.version import __version__

# Suppress floating-point errors as they do not provide useful information to
# end users while generating output noise
np.seterr(all="ignore")

def init(config):
    """Create a blank configuration file."""
    config.create_blank_conf()
    info = Info(config.config)
    console.print("\n")
    info.print_sedmodules_table()
    console.print(
        f"{INFO} The initial [bold]pcigale.ini[/bold] configuration file has "
        "been created. Please edit it to provide the data file (if fitting "
        "observations), the list of physical modules to use (see table above "
        "for the available modules), and the analysis module to adopt "
        "([bold]pdf_analysis[/bold] to fit observations or "
        "[bold]savefluxes[/bold] to save theoretical models). Once done, "
        "[bold]'pcigale genconf'[/bold] will add the configuraiton sections "
        "of the modules."
    )


def genconf(config):
    """Generate the full configuration."""
    config.generate_conf()

    # Pass config rather than configuration as the file cannot be auto-filled.
    info = Info(config.config)
    info.print_tables()

    console.print(
        f"{INFO} The [bold]pcigale.ini[/bold] configuration file has been "
        "updated and the configuration sections of the physical and the "
        "analysis modules have been added. Please edit it to provide the "
        "different values each of the physical properties should take for "
        "each physical module (note, this is not necessary if you use a "
        "parameters file). Once done, an optional sanity check can be done "
        "with [bold]'pcigale check'[/bold]. If everything is satisfactory, "
        "the code can be launched with [bold]'pcigale run'[/bold]."
    )


def check(config):
    """Check the configuration."""
    if conf := config.configuration:

        if conf['sed_modules'][0].split('_')[0] == 'sfhstohastic':
            conf['sed_modules_params'][conf['sed_modules'][0]]['nModels'] = (
                list(np.arange(conf['sed_modules_params'][conf['sed_modules'][0]]['nModels'][0])))

        params = ParametersManager(conf)
        if conf["analysis_method"] == "pdf_analysis":
            ObservationsManager(conf, params)
        info = Info(conf)
        info.print_tables()
        console.print(
            f"{INFO} No critical error has been found. Consider the "
            f"{WARNING} messages (if any). If everything is satisfactory, "
            "the code can be launched with [bold]'pcigale run'[/bold]."
        )


def run(config):
    """Run the analysis."""
    configuration = config.configuration


    if configuration['sed_modules'][0].split('_')[0] == 'sfhstohastic':
        configuration['sed_modules_params'][configuration['sed_modules'][0]]['nModels'] = (
            list(np.arange(configuration['sed_modules_params'][configuration['sed_modules'][0]]['nModels'][0])))


    if configuration:
        info = Info(config.configuration)
        info.print_tables()
        analysis_module = get_module(configuration["analysis_method"])

        start = dt.datetime.now()
        console.print(f"{INFO} Start: {start.isoformat('/', 'seconds')}")
        start = time.monotonic()  # Simpler time for run duration

        analysis_module.process(configuration)

        end = dt.datetime.now()
        console.print(f"{INFO} End: {end.isoformat('/', 'seconds')}")
        end = time.monotonic()

        delta = dt.timedelta(seconds=int(end - start))
        console.print(f"{INFO} Total duration: {delta}")


def main():
    Info.print_panel()
    if sys.version_info[:2] < (3, 8):
        raise Exception(
            f"Python {sys.version_info[0]}.{sys.version_info[1]} is "
            "unsupported. Please upgrade to Python 3.8 or later."
        )

    # We set the sub processes start method to spawn because it solves
    # deadlocks when a library cannot handle being used on two sides of a
    # forked process. This happens on modern Macs with the Accelerate library
    # for instance. On Linux we should be pretty safe with a fork, which allows
    # to start processes much more rapidly.
    if sys.platform.startswith("linux"):
        mp.set_start_method("fork")
    else:
        mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--conf-file",
        dest="config_file",
        help="Alternative configuration file to use.",
    )

    subparsers = parser.add_subparsers(help="List of commands")

    init_parser = subparsers.add_parser("init", help=init.__doc__)
    init_parser.set_defaults(parser="init")

    genconf_parser = subparsers.add_parser("genconf", help=genconf.__doc__)
    genconf_parser.set_defaults(parser="genconf")

    check_parser = subparsers.add_parser("check", help=check.__doc__)
    check_parser.set_defaults(parser="check")

    run_parser = subparsers.add_parser("run", help=run.__doc__)
    run_parser.set_defaults(parser="run")

    if len(sys.argv) == 1:
        parser.print_usage()
    else:
        args = parser.parse_args()

        if args.config_file:
            config = Configuration(Path(args.config_file))
        else:
            config = Configuration()

        if args.parser == "init":
            init(config)
        elif args.parser == "genconf":
            genconf(config)
        elif args.parser == "check":
            check(config)
        elif args.parser == "run":
            run(config)