# Overview

This branch incorporated the faults injection module into the closed-loop simulation with the Extended Cambridge patient model and hybrid closed-loop MPC  controller running under a single scale (static body weight) setting

## Input

run manual_script.py

Except for scenarios and parameters needed by the original closed-loop testbed, the following parameters need to be given in `manual_script.py`

- **Faults Specification**

A CSV file describes when and what faults need to be injected, with columns 

['Start_Time', 'Period', 'Data Label', 'Description']  

'Start_Time': Fault start time. Should be a datetime object.   (e.g., `2023-01-20 12:35`)              

'Period': Length of fault injection. Count in minutes.    (e.g., `1`)                   

'Data Label': Fault category.  (e.g.,  `negative_spike`)                 

'Description': Explain the attack or malfunction simulated here. (
`"Isolated Unphysiological Spike: Single point negative spike during exercise."`)

A sample file was put under:

pymgipsim/faultsGeneration/faults_specification.csv

- **Simulation start time**

Indicate the beginning date and time of simulation, e.g., 
`pd.Timestamp('2023-01-01 00:00:00')`

## Output

All simulated states with faulty injection labels will be saved under

SimulationResults/[Your current simulation dictionary]/

Simulated Data: model_state_results.xlsx

Simulated states same as the original testbed. The last column 

`'faults_label'`

will contain one of the following labels if a fault was injected at that time:

`['max_basal', 'min_basal', 'positive_basal', 'negative_basal',                       'unknown_stop', 'unknown_under', 'false_bolus', 'false_meal',`

`'missing_signal', 'positive_spike', 'negative_spike', 'negative_bias',                  'positive_bias', 'min_reading', 'max_reading', 'repeated_reading', 'zero_reading', 'repeated_episode']`

Else, it will be None

## Faults Injection

We simulated 17 fault patterns with corresponding labels as follows:

| Failure Pattern | Manipulations                                                                                                                    | QA_id | Faulty data label |
| --- |----------------------------------------------------------------------------------------------------------------------------------| --- | --- |
| Missing signal | Randomly select N points to substitute consecutive M data points with NaN. Label the time point as missing_signal                | 1, 2, 3, 12, 43 | 7 |
| Positive spike readings | Abrupt spike: Spike value = original value + 60. Label with positive_spike                                                       | 9, 10, 13, 16, 24, 36, 38, 55                                                                                         | 8 |
| Negative  spike readings | Spike value = original value - 60. Label with negative_spike                                                                     | 4, 9, 13, 14, 15, 33, 42, 54, 55 | 9 |
| Repeated readings | Randomly select N data points, changing the following M data points with the same value.  Label with repeated_reading            | 6, 7, 8, 11, 40, 42, 45 | 14 |
| Negative biased readings | glucose+/-=32 for consecutive M data points, label with negative_bias                                                            | 42, 47, 53, 60                                                                                                        | 10 |
| Positive biased readings | glucose+=32 for consecutive M data points. label with positive_bias                                                              | 47, 53                                                                                                                | 11 |
| Minimize readings | Minimize BG readings to 70 for a random period. Label with min_reading                                                           | 42,                                                                                                                   | 12 |
| Maximize readings | Maximize BG readings to 180 for a random period. Label with max_reading                                                          | 50,                                                                                                                   | 13 |
| Repeated episode | Substitute true BG readings with replayed the meal-taking episode to incur overdosing. Label with repeated_episode               | 45,                                                                                                                   | 17 |
| Zero readings | glucose reading = 0. Label with zero_reading                                                                                     | 8, 11, 42 | 14 |
| False Meal | Falsely register meal taking in the controller but not in the patient  (when BG is low and not meal time). Label with false_meal | 44 | 15 |
| False bolus | Repeated previous bolus request during non-meal and exercise time. Label with false_bolus                                        | 45, 46, 48, 49                                                                                                                   | 16 |
| Biased basal | Basal insulin rate add/minus 0.5 for a random period. Label with positive_basal or negative_basal                                | 46, 49                                                                                                                           | 3, 4 |
| Maximize basal | maximize rate: action=2. Label with max_basal                                                                                    | 48, 49, 52                                                                                                                       | 1 |
| Minimize basal | minimize rate: action=0.  Label with min_basal                                                                                   | 48, 49, 52, 58 | 2 |
| Unknow stopped delivery | Report insulin rate as usual, but change the basal insulin rate in the patient model to 0. Label with unknown_stop               | 58, 59, 62, 63                                                                                                                   | 5 |
| Unknow under delivery | Report insulin rate as usual, but basal insulin rate in the patient model minus 0.5. Label with unknown_under                    | 60, 61, 64                                                                                                                       | 6 |

Hazards were labelled as follows:

| Hazards  | Descriptions | QA ids |
| --- | --- | --- |
| Hypoglycemia | Blood glucose less than 70 mg/dL. Label with hypoglycemia | 17, 18, 19, 20, 21, 22,  25, 26, 30, 31, 34, 35, 37 |
| Hyperglycemia | Blood glucose is higher than  180 mg/dL. Label with hyperglycemia | 21, 23, 27, 28, 29, 31, 32, 34, 37 |

### The logic behind fault injection

The fault injection runs inside the simulation loop in `singlescale.py`

- For failures’ impact on CGM readings:

`['missing_signal', 'positive_spike', 'negative_spike', 'negative_bias',                  'positive_bias', 'min_reading', 'max_reading', 'repeated_reading', 'zero_reading', 'repeated_episode']`

The faulty injection engine will change the BG state obtained from the patient model at the previous step before sending it to the controller. 

Considering in real world, false readings will not directly change the patient’s physiological state, so we separated the state input into the patient model and into controller. The controller will receive false BG readings, and all other states will be computed based on the false readings. At the same time, the simulated patient will be computed with true BG and the impacted insulin dosage.

- For failures’ impact on insulin delivery:

`['max_basal', 'min_basal', 'positive_basal', 'negative_basal',                       'unknown_stop', 'unknown_under', 'false_bolus', 'false_meal']`

The insulin dosage will be changed after the controller has computed the current insulin and before this insulin is delivered into the patient model.

For unknown malfunctions with the pump, the displayed insulin delivery is normal, but the injected dosage is wrong.