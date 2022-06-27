# MO_LWV
This repository has the code, data, and results for Missouri redistricting for the League of Women Voters.

## Data
The **InitialPlans** folder contains the initial district plan assignment files for the experiments. There are census tract approximations of the congressional and state senate plans, and a census block group approximation of the state house plan. There are also hybrid county-tract or hybrid block-block group approximations of these plans.

The **MO_TractInput**, **MO_BlockGroupInput**, and **MO_BlockInput** folders contain input files for census unit adjacency, population, Democrat/Republican votes, and population broken down by race/ethnicity. The **MO_HybridCountyTractInput_Congress_6+30SplitCounties**, **MO_HybridCountyTractInput_Senate_7+1SplitCounties**, and **MO_HybridBlockGroupBlockInput_House_20SplitBlockGroups** folders contain the corresponding input files for hybrid combinations of counties and tracts, or block groups and blocks.

## Code

The **ReCom_LocalSearch_MO.ipynb/ReCom_LocalSearch_MO.py** code uses _Recombination (ReCom)_ iterations within local search to improve an objective for a given district plan and constraints. This code reads in parameters from **ReCom_PARAMETERS_MO.csv**. The user does not need to alter the code itself, they only need to change the parameters in this file.

The **MetricEvaluation.ipynb/MetricEvaluation.py** code evaluates a given district plan with respect to population balance, compactness, Efficiency Gap, Partisan Asymmetry, competitiveness, etc. This code creates images of the parties' vote-seat curves and district vote-shares and a .txt file with all metric values. The user needs to enter a folder path for the district plan(s) to evaluation (line 190), specific maps in that folder if applicable (line 196), and the folder path for the census unit data (line 200).

## Results

The **FinalPlans** folder contains the final assignment files for Missouri's congressional, state senate, and state house district plans. The plans are labeled with COMP, EG, PA, or CMPTTV, to indicate that they were optimized with respect to compactness, Efficiency Gap, Partisan Asymmetry, or competitiveness.
