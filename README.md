# MR-MEF Control Signals for EV Charging

This repository is associated with the paper: 

Powell, Siobhan, Sonia Martin, Ram Rajagopal, Inês M. L. Azevedo, and Jacques de Chalendar. "Future-proof rates for controlled electric vehicle charging: comparing multi-year impacts of different emission factor signals." Energy Policy (2024).

Please cite this code. 

### About the Paper

In this paper we compare AEF, SR-MEF, and a new EF, the MR-MEF, to see which induces demand response that really reduces emissions. We focus on the particular use case where the tariff is only updated once every 5 years and limited to a fixed weekday and weekend profile. 

We use the US portion of WECC as our case study, building on the modelling in https://doi.org/10.1038/s41560-022-01105-7. See https://github.com/SiobhanPowell/speech-grid-impact for the associated code. 


### About the Code

The code is organized as follows: 

#### GridModel Folder
In `GridModel`, we update the dispatch model using recent data on planned renewables, plant retirements and additions, and the latest plant operations. 
1. Collect recent renewables data for the base modelling years: 2019 and 2022.
2. Update the list of generator retirements.
3. Update the list of generator additions.
4. Convert some of the new data to match older file formats (renaming columns, etc.)
5. Calculate scaling projections for renewables, batteries, demand. 
6. Generate the new dispatch model objects.
7. Calculate capacity limits with the new objects.

`future_grid.py` and `simple_dispatch.py` are the main code for the grid model. These are identical to versions in the `RunResults` folder. In the new data there is a mismatch between naming conventions on the units (UNT) and generation (GEN) tabs, so we converted the names using `match_egrid_unt_numbers.py`. In the end this did not change the model or results.

#### EVModel Folder
In `EVModel`, we use the model of EV charging in `speech_classes.py`. In the main folder there is a folder `Data` with a subfolder `CP136` where we put the data downloaded from this project https://github.com/SiobhanPowell/speech-grid-impact. 
0. Generate simulated uncontrolled home and workplace sessions using the speech model.
1. Given control signals (see `RunResults`), use the simulated data to generate controlled charging profiles.
2. Fit models of the mapping from uncontrolled to controlled, and apply these to the large-scale profiles.


#### RunResults Folder
In `RunResults` we do the main analysis of the paper. 
1. `Grid1_uncontrolled*.py` we run the dispatch with the uncontrolled charging profiles. In `Grid1_*MRMEF*.py` we run the dispatch model with varying changes to the demand profile, generating all results needed to calculate the MR-MEF signals.
2. Based on the results from 1, we calculate the AEF, SR-MEF, and MR-MEF signals.
3. We run the control and dispatch for the AEF, SR-MEF, and MR-MEF cases with the block of minimally constrained demand.
4. We run the dispatch using the controlled EV profiles.
5. We generate the plots based on 1-4. These all use the 2019 base model. 
6. We repeat step 1 for the 2022 base model.
7. We repeat step 2 for the 2022 base model.
8. We repeat step 3 for the 2022 base model.
9. We plot results for the 2022 base model. 

#### Data

The structure of the `Data` folder:
- `Control_Data_ModelObjects` where the control results from step 2 in `EVModel` are saved
- `CP136` where the speech model data is saved, downloaded from this project https://github.com/SiobhanPowell/speech-grid-impact. 
- `EVProfiles` where we save the controlled profiles. Uncontrolled profiles are taken from the same source as the `CP136` folder.
- `GridInputData` includes a subfolder `2019Final` with:
    - EIA923_Schedules_2_3_4_5_M_12_2019_Final_Revision.xslx available from here: https://www.eia.gov/electricity/data/eia923/
    - eGRID2019_data.xlsx available here: https://www.epa.gov/egrid/download-data
    - A subfolder CEMS with data, organized into further subfolders by state code. 
    - Part 2 Schedule 6 - Balancing Authority Hourly System Lambda.csv and Respondent IDs.csv from here: https://www.ferc.gov/industries-data/electric/general-information/electric-industry-forms/form-no-714-annual-electric/data
    - fuel_default_prices.xlsx available from: https://github.com/tdeetjen/simple_dispatch and updated with more recent data.
Some formats for the grid input data changed in the 2022 version. See the code in `GridModel` 4 for the conversions.