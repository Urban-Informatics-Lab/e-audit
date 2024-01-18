# Usage
After importing the package, create an instance of the BuildingFeatures class. The BuildingFeatures class has a constructor that takes in the algorithm type as a parameter. For the algorithm type parameter, input ‘KNN’ for k-nearest neighbors, ‘DT’ for decision trees, or ‘Euc’ for Euclidean. 

Next, use the `process_alg()` method to generate the building feature classification from the building electricity load data. 

process_alg() takes in the following parameters: 
- Meter file(s) - file path (*str*)
- jEplus simulation file - file path (*str*)
- Date - format “MM/DD/YYYY” (*str*)
- Output files - file path (*str*) 
- Actual data - file path (*str*)
- Sq_ft (*int*) 
- J_conversion (*int*) - 1 if conversion is needed from J to kWh, 0 if not 

# Example 
Let's create `bf` as an instance of the BuildingFeatures class to run the k-nearest neighbors algorithm. 

    bf = BuildingFeatures('KNN')
    
    meter_files = "/Path_to/Meters_Example_IndividualFiles"
    sim_job = "/Path_to/SimJobIndex_Example.csv"
    date = "MM/DD/YYYY"
    output_files = "/Path_to/Output_files"
    actual_data = "/Path_to/Sample Building Electricity Data.csv"
    sq_ft = 10000
    J_conversion = 1 

    bf.process_alg(meter_files_dir, sim_job, date, output_files_path, actual_data, sq_ft, J_conversion)

# Tips 

- If errors arise when calling `process_alg()` check the order in which the parameters are entered. Make sure that it matches the example provided. 

- If a key error shows up from the meter files, check that the column name of your meter files matches with the sample meter file data provided. 