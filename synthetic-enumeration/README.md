# Synthetic enumeration for free energy calculations

Modeling of design conformations suggested by this [excellent blog post from Pat Walters](http://practicalcheminformatics.blogspot.com/2020/03/building-on-fragments-from-diamondxchem_30.html) and the [accompanying code](https://github.com/PatWalters/fragment_expansion/tree/master/oechem_eval).

## Manifest

* `activity-data/` - activity data

### Setup scripts
* `01-fix-csv-files.sh` - permute columns of input files
* `02-generate-poses.py` - generate constrained input poses for a single fragment structure
* `03-sort-poses.py` - sort ligands from best to worst docking score to prioritize (retaining reference ligand first), filter out ligands without poses

### pymol scripts
* `binding_site_figure.py` : set up binding side visualization with various lead series
* `create_movie.py` : generate movie of docked poses while rocking

### Compound sets
* `sprint-1/` : Retrospective and prospective 3-aminopyridine
* `sprint-2/` : Nucleophilic displacement for 3-aminopyridine
  * `results/` : results and ordered compounds
* `sprint-3/` : Benzotriazoles

#### Sprint 1 : Retrospective and prospective 3-aminopyridine

* `primary_amine_enumeration_for_chodera_lab_FEP.csv` - primary amine series (843 compounds)
* `boronic_ester_enumeration_for_chodera_lab_FEP.csv` - boronic ester series (122 compounds)
* `activity-data-2020-07-29.csv` - all compounds with activity data for retrospective benchmarking (888)
* `aminopyridine_compounds_for_FEP_benchmarking.csv` - 3-aminopyridine retrospective benchmarking compounds from Matthew Robinson (70)
* `fastgrant-table1.csv` - sentinel cases from Fast Grant application (Alpha Lee) (11)
* `primary_amine_enumeration_for_chodera_lab_FEP-permuted-conformers-x10789.sdf.gz`
* `boronic_ester_enumeration_for_chodera_lab_FEP-permuted-conformers-x10789.sdf.gz`
* `aminopyridine_compounds_for_FEP_benchmarking-conformers-x10789.sdf` - 3-aminopyridine retrospective benchmarking compounds using single common fragment, prioritized by docked scores
* `aminopyridine_compounds_for_FEP_benchmarking-dockscores-x10789.sdf` - 3-aminopyridine retrospective benchmarking compounds using individual maximum common fragment enumeration, prioritized by docked scores
* `2020-07-24.json`: `primary_amine_enumeration_for_chodera_lab_FEP.csv` forward only built from `x2646`
* `2020-07-27.json`: `primary_amine_enumeration_for_chodera_lab_FEP.csv` and `boronic_ester_enumeration_for_chodera_lab_FEP.csv` forward only built from `x10789`
* `2020-07-28.json`: `primary_amine_enumeration_for_chodera_lab_FEP.csv` and `boronic_ester_enumeration_for_chodera_lab_FEP.csv` backward only built from `x10789`
* `2020-07-29-retrospective-aminopyridines.json`: retrospective 3-aminopyridine compounds that JDC fished out automatically, single common scaffold, sparse conformers ranked by steric clashes
* `2020-08-02-retrospective-aminopyridines-matt.json`: retrospective 3-aminopyridine compounds from Matthew Robinson, using a single common scaffold based on x10789, dense conformers ranked by dock scores (-Cl in wrong pocket)
* `2020-08-03-retrospective-aminopyridines-matt-dockscores.json`: retrospective 3-aminopyridine compounds from Matthew Robinson, using a individually selected common scaffolds based on x10789, dense conformers ranked by dock scores (-Cl in right pocket)
* `2020-08-06-fastgrant-table1.json`: sentinel cases from the FastGrant
* `RAL-THA-6b94ceba.csv` - Ralph Robinson 3-aminopyridine P4 pocket exploration [`RAL-THA-6b94ceba`](https://postera.ai/covid/submissions/6b94ceba-f352-4275-ad8d-e766e56e6fa4)
* `EDG-MED-0da5ad92.csv` - Ed Griffen 3-aminopyridine exploration [`EDG-MED-0da5ad92`](https://covid.postera.ai/covid/submissions/0da5ad92-2252-46c0-8428-da7b3552d800)
* `RAL-THA-6b94ceba-dockscores-x10789.sdf` - docked conformers
* `EDG-MED-0da5ad92-dockscores-x10789.sdf` - docked conformers
* `2020-08-12-RAL-THA-6b94ceba1.json`: Ralph 3-aminopyridine designs
* `2020-08-13-EDG-MED-0da5ad92.json`: Compounds that are in process, but we may want to terminate

#### Sprint 2 : Nucleophilic displacement for 3-aminopyridine

* `nucleophilic_displacement_enumeration_for_FEP.csv` - nucleophilic displacement series (15918)
* `nucleophilic_displacement_enumeration_for_FEP-permuted-conformers-x10789.sdf.gz` - nucleophilic displacement series - docked conformers
* `2020-08-14-nucleophilic-displacement.json`: nucelophilic displacement for 3-aminopyridine scaffold, backward only His41(+) Cys145(-)

#### Sprint 3 : Benzotriazoles

* `2020-08-20-benzotriazoles.csv` - benzotriazole derivatives
* `2020-08-20-benzotriazoles-dockscores-x10876.sdf` - docked to x10876

## Misc
* `activity-data-2020-07-29.csv`: activity data downloaded from the [COVID Moonshot](https://covid.postera.ai/covid/activity_data.csv) on 2020-07-29
* `activity-data-2020-07-29-conformers-x10789.sdf.gz`: strictly filtered 3-aminopyridine related set for retrospective testing, with activity data preserved as SD tags (40 compounds)

## Procedure

Given a single reference fragment structure (already in `../receptors/` and prepared by `../scripts/00-prep-all-receptors.py`):
* expand uncertain stereochemistry for all target molecules
* identify the **common core** shared by the bound fragment structure and all target molecules
* identify the most likely protonation state in solution for each target molecule
* densely enumerate conformers with Omega, constraining the common core positions to the bound fragment structure
* pick the conformer with the least clashes with protein atoms

The SDF file written out contains the protonated fragment as molecule 0 followed by all docked fragments.

## TODO

* Don't use MMFF for omega:
```
Warning: OEMMFFParams::PrepMol() : unable to type atom 21 N
```
