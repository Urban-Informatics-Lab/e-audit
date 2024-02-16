# e-audit
# A "No-Touch" Energy Audit that Integrates Machine Learning and Simulation
## Building Feature Classification via Load Matching on Simulated Building Database

This Python package was created to implement the methodology proposed in "E-Audit: A "No-Touch" Energy Audit that Integrates Machine Learning and Simulation" by Excell, Andrews, and Jain. 

This package allows a user to evaluate the performance of building features using load matching on a building's electricity load data. One year's worth of hourly electricity readings are matched to a simulated database of similar buildings to predict the energy performance and physical characteristics of building features (i.e., window U-factor, roof insulation, plug loads, equipment schedules, etc.). This can be used to determine the non-geometric characteristics of a building or a group of buildings for which these features are unknown, or to audit the energy performance of specific features over time. 

## Creating the simulated database
Before running the classification functions, the first step requires creating a simulated database of scenarios. This is done by creating a building energy model, "tagging" the building parameters that you're interested in evaluating, and simulating various values for those parameters. The following instructions explain how to do this in EnergyPlus, using the parametric simulation tool 'jEPlus'.

First, you need to determine which building use type and which climate zone you are interested in. The building use type can be user-defined (by creating a new building energy model in EnergyPlus), or it can be one of the DOE reference building types (https://www.energycodes.gov/prototype-building-models). The climate zone is used to determine the typical weather file (TMY) for that region; if more geographically-specific weather data is available, this should be used instead of a TMY file. EnergyPlus provides TMY files for locations around the globe: https://energyplus.net/weather. 

The next step is to select building parameters which you are interested in evaluating. These features define the parametric scenarios. For example, a lighting retrofit would require changing the lighting power density. The EnergyPlus IDF file (*.idf) needs to be "tagged" with the parameters that you will be varying. To do this, replace the existing value with a descriptive name tag, such as "@@lighting@@" for the lighting power density. In a csv file, record the name of the tag along with the potential values that this parameter should take. An example "tagged" IDF file and parameter csv file have been provided with the example data (/Sample data/Modeling inputs). 

This modified IDF file, the typical weather data, and the parameter csv are the inputs you'll need to run jEPlus. You will need to specify which results to collect: the output used for this package is the **hourly electricity data for the entire facility** in a *csv format*. This can be done by defining result collection/post-processing in jEPlus using an *.mvi and/or *.rvi file - examples are provided with the example data. You'll need to download and install both EnergyPlus and jEPlus to run the simulations. jEPlus can be run from the command line or by using a graphical user interface (GUI). See the jEPlus website for more information on how to run the parameteric simulation (http://www.jeplus.org/wiki/doku.php). An example simulation output created using the example inputs has been provided with the sample data. 

## Running the classification functions
To run the classification step, the simulation output for hourly electricity data should be in csv format, with one csv for each parametric scenario. These csvs should all be in one folder: you will need to provide the filepath to the function. You will also need the csv file that specifies which parameters were used to simulate each scenario; in jEPlus, this file is typically called "SimJobIndex.csv". 

There are three classification algorithms included in this package: Euclidean distance, k-nearest neighbors, and decision trees. The Euclidean distance matching algorithm finds the distance between two time series: the matching scenario is the one with the closest distance between its time series and the one provided. For the machine learning algorithms, time series statistics (mean, median, min, max, standard deviation) are calculated at weekly, monthly, and yearly intervals, and these statistics are used to perform the supervised learning. 

The classification functions can take the electricity load profile from a building or a group of buildings and provide a predicted classification for each building parameter that was defined by the simulated database. 

## Usage
An environment YAML file has been provided with the packages used to run this code. To create a conda environment using this file, run `conda env create -n <env_name> -f environment.yaml` in the terminal, inserting the name of your environment in place of `<env_name>`. This package runs on Python 3.8 or newer. 

*Note: the scikit-learn package may have an error when installing from the environment file. In this case, run `pip install scikit-learn` once the environment is activated.*

After importing the package, create an instance of the EAudit class. The EAudit class has a constructor that takes in the algorithm type as a parameter. For the algorithm type parameter, input ‘KNN’ for k-nearest neighbors, ‘DT’ for decision trees, or ‘Euc’ for Euclidean. 

Next, use the `process_alg()` method to generate the building feature classification from the building electricity load data. 

`process_alg()` takes in the following parameters: 
- Meter file(s) - file path (*str*)
- Meter file(s) - electricity column (*str*)
- Meter file(s) - Start Date - format “MM/DD/YYYY” (*str*)
- jEplus simulation file - file path (*str*)
- Actual data - file path (*str*)
- Actual data - building id (*str*)
- Actual data - date column (*str*)
- sq_ft (*int*) - square footage of simulated building
- J_conversion (*int*) - 1 if `meter_col` is in J, 0 if `meter_col` is in kWh
- Output files - file path (*str*) 
- Plot results (*True or False*) 

## Example 
Let's create `ea` as an instance of the EAudit class to run the k-nearest neighbors algorithm. 

    ea = EAudit('KNN')

    ea.process_alg(
        meter_path = "/Path_to/Meters_Example_IndividualFiles", 
        meter_col = "Electricity:Facility [J](Hourly)", 
        meter_date = "MM/DD/YYYY", 
        sim_job_path = "/Path_to/SimJobIndex_Example.csv", 
        actual_path = "/Path_to/Sample Building Electricity Data.csv", 
        actual_id = "ID", 
        actual_date = "Date.Time",
        actual_col = "kWh_norm_sf",
        sq_ft = 2000,
        J_conv = 1,    
        output_path = "/Path_to/Output_files", 
        plot_results = True
    )

## Tips 

- If errors arise when calling `process_alg()` check the variables used for each input. Refer to the meter file path as `meter_path`, meter column name as `meter_col`, etc. 
