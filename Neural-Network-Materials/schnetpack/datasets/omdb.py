import os
import tarfile

import numpy as np
from ase.io import read
from ase.db import connect
from ase.units import eV

import schnetpack as spk
from schnetpack.datasets import DownloadableAtomsData


__all__ = ["OrganicMaterialsDatabase"]


class OrganicMaterialsDatabase(DownloadableAtomsData):
    """Organic Materials Database (OMDB) of bulk organic crystals.

    Registration to the OMDB is free for academic users. This database contains DFT
    (PBE) band gap (OMDB-GAP1 database) for 12500 non-magnetic materials.

    Args:
        path (str): path to directory containing database.
        cutoff (float): cutoff for bulk interactions.
        download (bool, optional): enable downloading if database does not exists.
        subset (list, optional): Deprecated! Do not use! Subsets are created with
            AtomsDataSubset class.
        load_only (list, optional): reduced set of properties to be loaded
        collect_triples (bool, optional): Set to True if angular features are needed.
        environment_provider (spk.environment.BaseEnvironmentProvider): define how
            neighborhood is calculated
            (default=spk.environment.SimpleEnvironmentProvider).

    References:
        Olsthoorn, B., Geilhufe, R.M., Borysov, S.S. and Balatsky, A.V. (2019),
        Band Gap Prediction for Large Organic Crystal Structures with Machine Learning.
        Adv. Quantum Technol., 2: 1900023. https://doi.org/10.1002/qute.201900023

    """

    BandGap = "band_gap"

    def __init__(
        self,
        path,
        download=True,
        subset=None,
        load_only=None,
        collect_triples=False,
        environment_provider=spk.environment.SimpleEnvironmentProvider(),
    ):
        available_properties = [OrganicMaterialsDatabase.BandGap]

        units = [eV]

        self.path = path

        dbpath = self.path.replace(".tar.gz", ".db")
        self.dbpath = dbpath

        if not os.path.exists(path) and not os.path.exists(dbpath):
            raise FileNotFoundError(
                "Download OMDB dataset (e.g. OMDB-GAP1_v1.1.tar.gz) from "
                "https://omdb.mathub.io/dataset and set datapath to this file"
            )

        super().__init__(
            dbpath=dbpath,
            subset=subset,
            load_only=load_only,
            collect_triples=collect_triples,
            available_properties=available_properties,
            units=units,
            environment_provider=environment_provider,
        )

        if download and not os.path.exists(dbpath):
            # Convert OMDB .tar.gz into a .db file
            self._convert()

    def _convert(self):
        """
        Converts .tar.gz to a .db file
        """
        print("Converting %s to a .db file.." % self.path)
        tar = tarfile.open(self.path, "r:gz")
        names = tar.getnames()
        tar.extractall()
        tar.close()

        structures = read("structures.xyz", index=":")
        Y = np.loadtxt("bandgaps.csv")
        [os.remove(name) for name in names]

        atoms_list = []
        property_list = []
        with connect(self.dbpath) as con:
            for i, at in enumerate(structures):
                atoms_list.append(at)
                property_list.append({OrganicMaterialsDatabase.BandGap: Y[i]})
        self.add_systems(atoms_list, property_list)
