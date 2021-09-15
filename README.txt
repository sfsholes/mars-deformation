README for Sholes and Rivera-Hernandez (2021) submitted to Icarus
September 14, 2021

The file [TID_model.py] can be run in the terminal and will plot out all four figures used in the paper as well as print out the stats used in Table 1. 

The file [TID_functions.py] contains the acutal functions used for modeling the Tharsis-induced deformation of the input features along with the associated true polar wander component. 

Please cite the Sholes and Rivera-Hernandez (2021) paper along with the Citron et al. (2018) paper that the model is based on. 

The [data] directory contains the necessary input data derived from external sources which are:
    [geo_CSlm.npy] and [shp_cslm.npy] are the shape and geoid components of the Tharsis deformation from Citron et al. (2018)
    [megt90n000cb.txt] is the 4 pixels per degree topographic map of Mars from the Mars Orbiter Laser Altimeter (MOLA) see 
                       https://pds-geosciences.wustl.edu/missions/mgs/megdr.html
    [Bouley2016_FlattenedNorthernPlains.csv] is a topographic map of Mars with the excess topography of the northern plains of Mars from Bouley et al. (2016). 

The [input] directory contains all the different mappings of the Arabia and Deuteronilus Levels along with the open basin deltas and valley network termini. 
    [fig3] directory contains the different mappings of the Arabia Level from Sholes et al. (2021)
    [fig4] directory contains the original open basin delta data from di Achille and Hynek (2010), updated delta data from Rivera-Hernandez and Palucis (2019), and the valley netork termini data from Chan et al. (2018). 


References:

Bouley, S., Baratoux, D., Matsuyama, I., Forget, F., Séjourné, A., Turbet, M., & Costard, F. (2016). Late Tharsis formation and implications for early Mars. Nature, 531(7594), 344-347. doi:10.1038/nature17171

Chan, N. H., Perron, J. T., Mitrovica, J. X., & Gomez, N. A. (2018). New Evidence of an Ancient Martian Ocean from the Global Distribution of Valley Networks. Journal of Geophysical Research: Planets, 123(8), 2138-2150. doi:10.1029/2018JE005536

Citron, R. I., Manga, M., & Hemingway, D. J. (2018). Timing of oceans on Mars from shoreline deformation. Nature, 555(7698), 643-646. doi:10.1038/nature26144
Di Achille, G., & Hynek, B. M. (2010). Ancient ocean on Mars supported by global distribution of deltas and valleys. Nature Geoscience, 3(7), 459-463. doi:10.1038/ngeo891

Rivera-Hernandez, F., & Palucis, M. C. (2019). Do deltas along the crustal dichotomy boundary of Mars in the Gale crater region record a northern ocean? Geophysical Research Letters. doi:10.1029/2019GL083046

Sholes, S. F., Dickeson, Z. I., Montgomery, D., & Catling, D. (2021). Where are Mars’ Hypothesized Ocean Shorelines? Large Lateral and Topographic Offsets Between Different Versions of Paleoshoreline Maps. Journal of Geophysical Research: Planets, 126. doi:10.1029/2020JE006486

