# Data presentation

## Sources

Data was collected from the following sources:

| #    | Name                                                         | Description                                                  | Link                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| *1*  | OWID data | The variables represent all of the main data related to confirmed cases, deaths, hospitalizations, and testing, as well as other variables of potential interest. | https://github.com/owid/covid-19-data/tree/master/public/data |
| *2*  | Region Mobility Report | Google's mobility dataset with mobility data in a country/region regarding different locations: grocery stores, pharmacy, work places...etc.   | https://www.google.com/covid19/mobility/ |
         |

- **OWID data can be saved automatically: running the function `update_owid` (from `..\utils.py`) returns the filepath to which the csv file has been saved in `./EPI`.
- **Mobility data zip file has to be downloaded from [HERE](https://www.google.com/covid19/mobility/) and placed in this directory.

## Data of interest

| File name                                  | Source #     | Description                                                  |
| ------------------------------------------ | ------------ | ------------------------------------------------------------ |
| `./EPI/owid_{last_update_date}.csv`              | *1*          | Variables: Total cases (i.e cumulative covid19 cases), stringency index (i.e level of measures implemented in the country/region), population in the country/region directly from *1* |
| `Region_Mobility_Report_CSVs.zip/2020_{iso_code_country}_Region_Mobility_Report.csv` | *2*          | Mobility data variables: 'retail_and_recreation_percent_change_from_baseline',       'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline |


## Data processing:

In `../utils.py`

* Creation of a stringency index smoothed as a 20-day rolling mean and normalized between 0 and 1: function ` get_stringency_data `
* Creation of a mobility index smoothed as a 20-day rolling mean and normalized between 0 and 1: function ` get_mobility_data `
