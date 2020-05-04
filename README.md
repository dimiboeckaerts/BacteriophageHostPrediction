# BacteriophageHostPrediction
This repository contains the code and database related to our manuscript on predicting bacteriophage hosts based on receptor-binding proteins. This research is funded by a PhD fellowship strategic basic research from the Research Foundation – Flanders (FWO), grant number 1S69520N.

* <i>RBP_database.csv</i> contains the collected RBP sequences as described in Materials & Methods.

* <i>RBP_functions.py</i> is a Python script containing all the necessary manually implemented functions for the various analyses carried out in this study.

* <i>RBP_alignment.jl</i> is a Julia script used to compute pairwise alignments between the sequences to explore the diversity and assess redundancy in the database.

* <i>RBP_host_prediction.ipynb</i> is an IPython notebook containing all the analyses, model constructions and evaluations described in Materials & Methods.
