This repo contains UMA (Unified Model of Arithmetic) version 0.600. This version of UMA is substantively identical to the version (0.550) used for simulations submitted to Psychological Review in September 2022. Changes were made to make the code easier to understand (e.g., editing comments for clarity, deleting imports and functions that are no longer used) but no changes were made that affect the functioning of the model or simulations.

uma.py contains UMA's architecture and several production rules that are to be included in any UMA model.

models.py contains the production rules used by the full model used in the reported simulations. To see the model in action on a few sample problems, uncomment and run one or more of the examples at the end of this file. 

Alternate models can be constructed by creating alternate rule sets (for example, subsets of the rules in models.py). This is illustrated in the last two examples, which use a modified version of the model that (unlike the default model) cannot commit errors in single digit whole number addition or multiplication.

simulators.py contains code for running simulations using the model specified by uma.py and models.py. The code at the bottom specifies the simulations whose results were reported in the paper. The Simulation class is capable of running a wide variety of other simulations. 

The simulations take a long time to run and occasionally crash. If this happens, you can resume from the point at which the crash occurred by setting a value other than 0 for start_subjid in the call to Simulation.run.
