This folder contains UMA (Unified Model of Arithmetic) version 0.600. This version of UMA is substantively identical to the version (0.550) used for simulations submitted to Psychological Review in September 2022. Changes were made to make the code easier to understand (e.g., editing comments for clarity, deleting imports and functions that are no longer used) but no changes were made that affect the functioning of the model or simulations.

The three files in the same folder as this README constitute the model itself. 

> uma.py contains UMA's architecture and several production rules that are to be included in any UMA model.

> models.py contains the production rules used by the full model used in the reported simulations. To see the model in action on a few sample problems, uncomment and run one or more of the examples at the end of this file. Alternate models can be constructed by creating alternate rule sets (for example, subsets of the rules in models.py). This is illustrated in the last two examples, which use a modified version of the model that (unlike the default model) cannot commit errors in single digit whole number addition or multiplication.

> simulators.py contains code for running simulations using the model specified by uma.py and models.py. The code at the bottom specifies the simulations whose results were reported in the paper. The Simulation class is capable of running a wide variety of other simulations. The simulations take a long time to run and occasionally crash. If this happens, you can resume from the point at which the crash occurred by setting a value other than 0 for start_subjid in the call to Simulation.run.

The Output subfolder contains the output from simulators.py that was reported in the submission to Psychological Review.

> The subfolder ALL29_BCD contains the raw output of the simulations (one file per simulated participant per assessment per grade).

> The file "sim all29_BCD.csv" contains the results of calling Simulation.mergeResults on the contents of the subfolder "ALL29_BCD", namely, a single file containing all assessment results of all simulated participants.

> Note that the column names of the above files are slightly different from what you would get from a fresh run of the simulations. The reason is that the simulation output was generated using UMA 0.550 rather than 0.600 - see the first paragraph of this README.

The subfolders "Problem Sets Training" and "Problem Sets Testing" contain, respectively, the problem sets that were used to train and test UMA. These files are used by simulators.py to run simulations.
