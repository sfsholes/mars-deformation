README
Sholes and Rivera-Hernandez 2022 (Icarus)
Code for: Constraints on the Uncertainty, Timing, and Magnitude of Potential \
Mars Oceans from Topographic Deformation Models

To run the model and create all the figures simply run the Python code:
-> python TID_Model.py

The TID model and other useful functions are in the TID_functions.py file.

I have set it up so that data is loaded as a "Shoreline" class (note: this does
not imply they are actually shorelines) which takes in a file path to a csv
file and initializes a lot of the needed information. You can play with the data
by accessing the .df variable which finds the contributions due to Tharsis
directly and its associated TPW via the premade pre_tharsis_maps() function
which handles the creation of the TID model map. 

Note: The optimization codes will create an output (.x, see code for examples)
which is a list object with item[0] being the best-fit value of C (percentage of
Tharsis buildup) and item[1] being the 'mean sea level.' However, note that the
latter is a remnant of older code and the rms function no longer uses that value
to minimize itself (instead it natively calculates the mean paleo-elevation).
So, do not use item[1] as the 'mean sea level' but rather calculate it yourself,
e.g., dataframe['OPT'].mean().

Note on Data: There is a slight offset (~20 km S) that we needed to do with the
original XY data provided by Ivanov et al. (2017). This is some weird projection
issue that ArcMap has between coordinate systems (and Citron et al. 2018 had to
do the same thing - personal communication). This does not change their results
nor ours as it is just a manual projection transformation and we matched it to
the geomorphology observed in the THEMIS-IR daytime mosaic. Note however, that
we did update their elevation data using the MOLA/HRSC 200 m/px digital
elevation model. This is also true with the data used in Sholes et al. (2021).
