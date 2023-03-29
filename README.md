# MO_LWV
This repository has the code and data for Missouri redistricting for the League of Women Voters.

## Data
The **MO_Input_Senate** folder contains the initial district plan assignment files and hybrid county-tract shapefiles (with unit population, Democrat/Republican votes, and population broken down by race/ethnicity) for the Missouri state senate. The **MO_Input_House** and **MO_Input_Congress** folders contain analogous input data for the Missouri state house and Missouri congressional plans.

## Code

The **ReComLocalSearch_MO_2022.ipynb/ReComLocalSearch_MO_2022.py** code uses the open-source GerryChain package to implement _Recombination (ReCom)_ iterations within local search to improve an objective for a given district plan, subject to constraints. This code reads in parameters from **ReCom_PARAMETERS_MO_2022.csv.** There are also .yml environment files in the folder **Environment** that detail the versions of all packages that the code uses.

The **MetricEvaluation_MO_2022.ipynb/MetricEvaluation_MO_2022.py** code evaluates a given district plan (or plans) with respect to population balance, compactness, Efficiency Gap, Partisan Asymmetry, competitiveness, whole counties, etc. This code creates images of the parties' district vote-shares and a .csv file with all metric values.

## Parameter File

The local search code reads in parameters from **ReCom_PARAMETERS_MO_2022.csv.** The user does not need to alter the code itself, they only need to change the parameters in this file. The file is currently set up to optimize Missouri's state senate plan for compactness (measured by total perimeter), subject to population balance and whole-districts-within-counties constraints. With the given parameters, running the code will produce the assignment file of one state senate district plan optimized for compactness (located in a folder named Test) and should take approximately 30 seconds on a consumer-grade desktop/laptop computer.
