import multiprocessing as mp
from glob import glob  # To allow the use of glob() in "eval..."
from pathlib import Path
from textwrap import wrap

import configobj
import numpy as np
import validate

from pcigale import analysis_modules, sed_modules
from pcigale.data import SimpleDatabase as Database
from pcigale.managers.parameters import ParametersManager
from pcigale.session import validation
from pcigale.utils.console import ERROR, INFO, WARNING, console
from pcigale.utils.io import read_table
from pcigale.warehouse import SedWarehouse


class Configuration:
    """This class manages the configuration of pcigale."""

    def __init__(self, filename=Path("pcigale.ini")):
        """Initialise a pcigale configuration.

        Parameters
        ----------
        filename: Path
            Name of the configuration file (pcigale.conf by default).

        """
        # We should never be in the case when there is a pcigale.ini but no
        # pcigale.ini.spec. While this seems to work when doing the pcigale
        # genconf, it actually generates an incorrect pcigale.ini.spec. The only
        # clean solution is to rebuild both files.
        if filename.is_file() and not filename.with_suffix(".ini.spec").is_file():
            raise Exception(
                "The pcigale.ini.spec file appears to be missing. "
                "Please delete the pcigale.ini file and regenrate "
                "it with 'pcigale init' and then 'pcigale genconf' "
                "after having filled the initial pcigale.ini "
                "template."
            )
        else:
            self.spec = configobj.ConfigObj(
                str(filename.with_suffix(".ini.spec")),
                write_empty_values=True,
                indent_type="  ",
                encoding="UTF8",
                list_values=False,
                _inspec=True,
            )
            self.config = configobj.ConfigObj(
                str(filename),
                write_empty_values=True,
                indent_type="  ",
                encoding="UTF8",
                configspec=self.spec,
            )

        # We validate the configuration so that the variables are converted to
        # the expected that. We do not handle errors at the point but only when
        # we actually return the configuration file from the property() method.
        self.config.validate(validate.Validator(validation.functions))

        self.pcigaleini_exists = filename.is_file()

    def create_blank_conf(self):
        """Create the initial configuration file

        Write the initial pcigale configuration file where the user can state
        which data file to use, which modules to use for the SED creation, as
        well as the method selected for statistical analysis.

        """
        # If there is a pre-existing pcigale.ini, delete the initial comment as
        # otherwise it gets added each time a pcigale init is run.
        self.config.initial_comment = []

        self.config["data_file"] = ""
        self.config.comments["data_file"] = (
            wrap("File containing the input data. The columns are:")
            + wrap("* 'id': name of the object.")
            + wrap(
                "* 'redshift': redshift of the object. If 0, the distance "
                "is assumed to be 10 pc. If negative the photometric mode is "
                "enabled. In that case an explicit redshift grid must be "
                "provided in the redshfting module."
            )
            + wrap(
                "* 'distance' (Mpc, optional): distance in Mpc. Optional, "
                "if present it will be used in lieu of the distance computed "
                "from the redshift assuming a Planck et al. (2018) cosmology."
            )
            + wrap(
                "* band fluxes: fluxes in filters in mJy. Each column is "
                "identified with a unique filter name, for instance "
                "jwst.nircam.F200W for the 2 μm filter of the NIRCAM "
                "instrument onboard JWST. The known filters are avalaible "
                "from the command 'pcigale-filters list'."
            )
            + wrap(
                "* line fluxes: fluxes in lines in W/m². Each column is "
                "identified after a unique line name prefixed with 'line.', for "
                "instance line.H-alpha. The known filters are available from the "
                "command 'pcigale-filters list'. It is possible to combine lines "
                "together, for instance line.NII-654.8+H-alpha+NII-658.3 will be "
                "the sum of Hα and the adjacent NII lines."
            )
            + wrap(
                "* physical properties: values of the intensive and "
                "extensive physical properties (e.g., dust.luminosity for the "
                "dust luminosity). To determine the format of the columns and "
                "the units, it is suggested to do a minimal run and examine the "
                "output FITS table."
            )
            + wrap(
                "* uncertainties: the uncertainties for the band and line "
                "fluxes as well as for the physical properties are given using "
                "the same column names with the '_err' suffix."
            )
            + wrap("Important notes:")
            + wrap("* Fluxes can be positive or negative.")
            + wrap(
                "* Upper limits are indicated with a negative value for "
                "the uncertainty."
            )
            + wrap(
                "* In case some fluxes are missing for some objects, "
                "they can be replaced with NaN."
            )
            + wrap(
                "* This file is optional to generate the configuration "
                "file, in particular for the savefluxes module."
            )
        )
        self.spec["data_file"] = "string()"

        self.config["parameters_file"] = ""
        self.config.comments["parameters_file"] = [""] + wrap(
            "Optional file containing the list of physical parameters. Each "
            "column must be in the form module_name.parameter_name, with each "
            "line being a different model. The columns must be in the order "
            "the modules will be called. The redshift column must be the last "
            "one. Finally, if this parameter is not empty, cigale will not "
            "interpret the configuration parameters given in pcigale.ini. "
            "They will be given only for information. Note that this module "
            "should only be used in conjonction with the savefluxes module. "
            "Using it with the pdf_analysis module will yield incorrect "
            "results."
        )
        self.spec["parameters_file"] = "string()"

        self.config["sed_modules"] = []
        self.config.comments["sed_modules"] = (
            [""]
            + wrap("Available modules to compute the models. The order must be kept.")
            + wrap("SFH:")
            + wrap("* sfh2exp: double exponential.")
            + wrap("* sfhdelayed: delayed SFH with optional exponential burst.")
            + wrap("* sfhdelayedbq: delayed SFH with optional constant burst/quench.")
            + wrap("* sfhfromfile: arbitrary SFH read from an input file.")
            + wrap("* sfhperiodic: periodic SFH, exponential, rectangle or delayed.")
            + wrap("* sfhstochastic_nonparametric (random changes using tStudent distribution)")
            + wrap("* sfhstochastic_regulator (Physical motivated random changes, Iyer+24)")
            + wrap("SSP:")
            + wrap("* bc03: Bruzual and Charlot 2003.")
            + wrap("* cb19: Charlot and Bruzual 2019.")
            + wrap("* bpassv2: BPASS 2.2 ")
            + wrap("* m2005: Maraston 2005.")
            + wrap("Nebular emission:")
            + wrap("* nebular: continuum and line nebular emission.")
            + wrap("Dust attenuation:")
            + wrap(
                "* dustatt_modified_CF00: modified Charlot & Fall 2000 "
                "attenuation law."
            )
            + wrap(
                "* dustatt_modified_starburst: modified Calzetti 2000 "
                "attenuaton law."
            )
            + wrap("Dust emission:")
            + wrap("* casey2012: Casey 2012 dust emission models.")
            + wrap("* dale2014: Dale et al. 2014 dust emission templates.")
            + wrap("* dl2007: Draine & Li 2007 dust emission models.")
            + wrap("* dl2014: Draine et al. 2014 update of the previous models.")
            + wrap("* themis: Themis dust emission models from Jones et al. (2017).")
            + wrap("AGN:")
            + wrap("* skirtor2016: AGN models from Stalevski et al. (2012, 2016).")
            + wrap("* fritz2006: AGN models from Fritz et al. (2006).")
            + wrap("X-ray:")
            + wrap("* yang20 (X-ray models from Yang et al. 2020).")
            + wrap("* lopez24 (X-ray models from Lopez et al. 2024).")
            + wrap("Radio:")
            + wrap(
                "* radio: galaxy synchrotron emission and AGN; note: "
                "skirtor2016/fritz2006 is needed for AGN."
            )
            + wrap("Restframe parameters:")
            + wrap(
                "* restframe_parameters: UV slope (β), IRX, D4000, "
                "equivalent width, etc."
            )
            + wrap("Redshift+IGM:")
            + wrap(
                "* redshifting: mandatory, also includes the IGM from "
                "Meiksin (2006)."
            )
        )
        self.spec["sed_modules"] = "cigale_string_list()"

        self.config["analysis_method"] = ""
        self.config.comments["analysis_method"] = [""] + wrap(
            "Method used for statistical analysis. Available methods: "
            "pdf_analysis, savefluxes."
        )
        self.spec["analysis_method"] = "string()"

        self.config["cores"] = mp.cpu_count()
        self.config.comments["cores"] = [""] + wrap(
            "Number of CPU cores to use in paralel."
        )
        self.spec["cores"] = "integer(min=1)"

        self.config.write()
        self.spec.write()

    def generate_conf(self):
        """Generate the full configuration file

        Reads the user entries in the initial configuration file and add the
        configuration options of all selected modules as well as the filter
        selection based on the filters identified in the data table file.

        """
        if self.pcigaleini_exists is False:
            raise Exception("pcigale.ini could not be found.")

        modules = self.config["sed_modules"]
        if "m2005" in modules:
            if ("yang20" in modules) or ("lopez24" in modules):
                raise Exception(
                    "The m2005 module is not compatible with the X-ray modules."
                )
        # Getting the list of the filters available in pcigale database
        with Database("filters") as db:
            filter_list = db.parameters["name"]
        with Database("nebular_lines") as db:
            lines = db.get(Z=0.02, logU=-2.0, ne=100.0).name

        filter_list += [f"line.{line}" for line in lines]

        if self.config["data_file"] != "":
            obs_table = read_table(self.config["data_file"])

            # Check that the the file was correctly read and that the id and
            # redshift columns are present in the input file
            if "col1" in obs_table.columns:
                raise Exception(
                    "The input could not be read properly. Verify its format "
                    "and that it does not have two columns with the same name."
                )
            if "id" not in obs_table.columns:
                raise Exception("Column id not present in input file")
            if "redshift" not in obs_table.columns:
                raise Exception("Column redshift not present in input file")

            # Finding the known filters in the data table
            bands = []
            for band in obs_table.columns:
                filter_name = band[:-4] if band.endswith("_err") else band
                if filter_name in filter_list:
                    bands.append(band)

            # Check that we don't have an band error without the associated
            # band
            for band in bands:
                if band.endswith("_err") and (band[:-4] not in bands):
                    raise Exception(
                        f"The observation table as a {band} column but no "
                        f"{band[:-4]} column."
                    )

            self.config["bands"] = bands
        else:
            self.config["bands"] = ""
        self.config.comments["bands"] = [""] + wrap(
            "Bands and emission lines to consider. To take uncertainties "
            "into account, the name of the band or line must be indicated "
            "with the _err suffix as in the input flux file. For instance: "
            "galex.FUV, galex.FUV_err. The list of allowed bands and lines "
            "can be obtained with pcigale-filters list. Line names must be "
            "prefixed with 'line.', for instance line.H-alpha, as in the "
            "input flux file."
        )
        self.spec["bands"] = "cigale_string_list()"

        self.config["properties"] = ""
        self.config.comments["properties"] = [""] + wrap(
            "Properties to be considered. All properties are to be given in "
            "the rest frame rather than the observed frame. This is the case "
            "for instance the equivalent widths and for luminosity "
            "densities. The names correspond to those used in the input file "
            "with the uncertainties indicated with the _err suffix."
        )
        self.spec["properties"] = "cigale_string_list()"

        # Additional error
        self.config["additionalerror"] = "0.1"
        self.config.comments["additionalerror"] = [""] + wrap(
            "Relative error added in quadrature to the uncertainties of the "
            "fluxes and the extensive properties. Uncertainties on "
            "intensive properties are left unchanged."
        )
        self.spec["additionalerror"] = "float(min=0.0)"

        # SED creation modules configurations. For each module, we generate
        # the configuration section from its parameter list.
        self.config["sed_modules_params"] = {}
        self.config.comments["sed_modules_params"] = ["", ""] + wrap(
            "Configuration of the SED creation modules."
        )
        self.spec["sed_modules_params"] = {}

        for module_name in self.config["sed_modules"]:
            self.config["sed_modules_params"][module_name] = {}
            self.spec["sed_modules_params"][module_name] = {}
            sub_config = self.config["sed_modules_params"][module_name]
            sub_spec = self.spec["sed_modules_params"][module_name]

            for name, (typ, description, default) in sed_modules.get_module(
                module_name, blank=True
            ).parameters.items():
                if default is None:
                    default = ""
                sub_config[name] = default
                sub_config.comments[name] = wrap(description)
                sub_spec[name] = typ
            self.config["sed_modules_params"].comments[module_name] = [
                sed_modules.get_module(module_name, blank=True).comments
            ]

        # Configuration for the analysis method
        self.config["analysis_params"] = {}
        self.config.comments["analysis_params"] = ["", ""] + wrap(
            "Configuration of the statistical analysis method."
        )
        self.spec["analysis_params"] = {}

        module_name = self.config["analysis_method"]
        for name, (typ, desc, default) in analysis_modules.get_module(
            module_name, blank=True
        ).parameters.items():
            if default is None:
                default = ""
            self.config["analysis_params"][name] = default
            self.config["analysis_params"].comments[name] = wrap(desc)
            self.spec["analysis_params"][name] = typ

        if "pdf_analysis" == module_name:
            bands = [band for band in self.config["bands"] if not band.endswith("_err")]
            self.config["analysis_params"]["bands"] = bands

        # Temporarily add a single redshift to be able to generate a single model
        self.config["sed_modules_params"]["redshifting"]["redshift"] = [0.0]
        self.add_line_list()
        self.check_and_complete_analysed_parameters()
        if (
            self.config["data_file"] != ""
            and self.config["analysis_method"] == "pdf_analysis"
        ):
            colnames = set(obs_table.colnames)
            vars = []
            for var in colnames & set(self.config["analysis_params"]["variables"]):
                vars.append(var)
                if (varerr := f"{var}_err") in colnames:
                    vars.append(varerr)
            self.config["properties"] = vars

        # Remove the redshift before saving the configuration file
        self.config["sed_modules_params"]["redshifting"]["redshift"] = ""
        self.config.write()
        self.spec.write()

        # Add the redshifts for the info table but do not save them
        self.complete_redshifts()

    @property
    def configuration(self):
        """Returns a dictionary for the session configuration if it is valid.
        Otherwise, print the erroneous keys.

        Returns
        -------
        configuration: dictionary
            Dictionary containing the information provided in pcigale.ini.
        """
        if self.pcigaleini_exists is False:
            raise Exception("pcigale.ini could not be found.")

        self.complete_redshifts()
        if (
            len(self.config["sed_modules_params"]["redshifting"]["redshift"]) == 0
            and len(self.config["parameters_file"]) == 0
        ):  # Stop if there is no redshift except when a parameters file is provided
            console.print(
                f"{ERROR} No redshift found. Either provide a flux file or "
                "an explicit redshift grid in the redshifting module."
            )
            return None

        self.add_line_list()
        self.check_and_complete_analysed_parameters()

        vdt = validate.Validator(validation.functions)
        validity = self.config.validate(vdt, preserve_errors=True)

        if validity is not True:
            console.print(
                f"{ERROR} The following issues have been found in pcigale.ini:"
            )
            for module, param, message in configobj.flatten_errors(
                self.config, validity
            ):
                if len(module) > 0:
                    console.print(
                        f"{ERROR} Module [b]{'/'.join(module)}[/b], "
                        f"parameter [b]{param}[/b]: {message}"
                    )
                else:
                    console.print(f"{ERROR} Parameter [b]{param}[/b]:" f"{message}")
            console.print(
                f"{INFO} Run the same command after having fixed pcigale.ini."
            )

            return None

        return self.config.copy()

    def complete_redshifts(self):
        """Complete the configuration when the redshifts are missing from the
        configuration file and must be extracted from the input flux file.
        """

        z_mod = self.config["sed_modules_params"]["redshifting"]["redshift"]
        if isinstance(z_mod, str) and not z_mod:
            if self.config["data_file"]:
                obs_table = read_table(self.config["data_file"])
                if "redshift_decimals" in self.config["analysis_params"]:
                    decimals = self.config["analysis_params"]["redshift_decimals"]
                    if decimals < 0:
                        z = list(np.unique(obs_table["redshift"]))
                    else:
                        z = list(
                            np.unique(
                                np.around(obs_table["redshift"], decimals=decimals)
                            )
                        )
                else:
                    z = list(np.unique(obs_table["redshift"]))
                if np.min(z) < 0.0:
                    console.print(
                        f"{WARNING} Some redshifts provided in the input "
                        "file are negative. Indicating the photometric "
                        "redshift mode is enabled. In this case an explicit "
                        "list of redshifts must be provided to the "
                        "redshifting module."
                    )
                self.config["sed_modules_params"]["redshifting"]["redshift"] = z
            elif self.config["parameters_file"]:
                # The entry will be ignored anyway. Just pass a dummy list
                self.config["sed_modules_params"]["redshifting"]["redshift"] = []
            else:  # No redshift found (e.g. and generating pcigale.ini)
                self.config["sed_modules_params"]["redshifting"]["redshift"] = []

    def add_line_list(self):
        """Add the list of lines to compute to the nebular module."""

        if "nebular" not in self.config["sed_modules"]:
            return

        # Lines used in the fitted bands.
        line_columns = {
            name[5:]
            for name in self.config["bands"]
            if name.startswith("line.") and not name.endswith("_err")
        }

        # Lines estimated in the pdf_analysis module.
        if self.config["analysis_method"] == "pdf_analysis":
            line_columns |= {
                name[5:]
                for name in self.config["analysis_params"]["bands"]
                if name.startswith("line.")
            }

        # Flatten the set for line combinations and remove the line. prefix
        lines = {line for column in line_columns for line in column.split("+")}

        self.config["sed_modules_params"]["nebular"]["line_list"] = " & ".join(lines)

    def check_and_complete_analysed_parameters(self):
        """Check that the variables to be analysed are indeed computed and if "
        "no variable is given, complete the configuration with all the "
        "variables extracted from a dummy run."""
        params = ParametersManager(self.config.dict())
        sed = SedWarehouse().get_sed(params.modules, params.from_index(0))
        info = list(sed.info.keys())

        if len(self.config["analysis_params"]["variables"]) > 0:
            nolog = [
                k if not k.endswith("_log") else k[:-4]
                for k in self.config["analysis_params"]["variables"]
            ]
            diff = set(nolog) - set(info)
            if len(diff) > 0:
                raise Exception(
                    f"{', '.join(diff)} unknown. "
                    f"Available variables are: {', '.join(info)}."
                )
        else:
            info.sort()
            self.config["analysis_params"]["variables"] = info