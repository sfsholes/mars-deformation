# Tharsis-Induced Deformation (TID) Model
# This is for Sholes and Rivera-Hernandez (2021) submitted to Icarus
# Please cite both the above and Citron et al. (2018) of which this work
#  is heavily based on. The input data is also provided in this directory
#  which comes from Sholes et al. (2021) in JGR:Planets and their respective
#  original sources.

# This model takes in CSV files for features (with columns Lat, Lon, and Elevation [m])
# and removes the deformation caused by Tharsis and its associated true polar wander
# Running this file will produce Figures 1-4 of the paper and requires the TID_functions
#  file which does the actual calculations based on the Citron et al. paper

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd
import math
import copy
import os
import scipy.optimize as optimize
from TID_functions import *

def figure1():
    """Plots the replication of the Citron et al. (2018) deformation curve with the
    actual paleotopography of the data. Also plots the full Carr and Head (2003)
    Arabia Level paleo and modern topography with full deformation curves."""

    perron = Shoreline("input/Perron2007_Arabia.csv", sea_level=sea_citron)
    perron_data = perron.df
    carr = Shoreline("input/CarrHead2003_Arabia.csv", sea_level=sea_citron)
    carr_data = carr.df
    carr_df = copy.deepcopy(carr_data)

    print_stats(perron, name='Perron et al. 2007')

    # To plot the model curve, need to break it into chunks
    # This is because plotting LON rather than distance along level
    #   and there is overlap as they levels cross each other in longitude
    section_1 = drop_return(carr_df, carr_df[(carr_df["LON"]<-147.9) & (carr_df["LAT"]<23.0)].index)
    section_2 = drop_return(carr_df, carr_df[(carr_df["LON"]<-128.0)].index)
    section_3 = drop_return(carr_df, carr_df[(carr_df["LON"]<-56.0)].index)
    section_4 = drop_return(carr_df, carr_df[(carr_df["LON"]<-48.0) & (carr_df["LAT"]>44.0)].index)
    section_5 = drop_return(carr_df, carr_df[(carr_df["LON"]<-48.0) & (carr_df["LAT"]>30.0)].index)
    section_6 = drop_return(carr_df, carr_df[(carr_df["LON"]<-42.0)].index)
    section_7 = drop_return(carr_df, carr_df[(carr_df["LON"]<77.5)].index)
    section_8 = drop_return(carr_df, carr_df[(carr_df["LON"]<80.5) & (carr_df["LAT"]>26.2)].index)
    section_9 = drop_return(carr_df, carr_df[(carr_df["LON"]<80.5) & (carr_df["LAT"]>22.7)].index)
    section_10 = drop_return(carr_df, carr_df[(carr_df["LON"]<80.5) & (carr_df["LAT"]>20.8)].index)
    section_11 = drop_return(carr_df, carr_df[(carr_df["LON"]<80.5) & (carr_df["LAT"]>18.2)].index)
    section_12 = drop_return(carr_df, carr_df[(carr_df["LON"]<80.0) & (carr_df["LAT"]>15.5)].index)
    section_13 = drop_return(carr_df, carr_df[(carr_df["LON"]<98.5)].index)
    section_14 = drop_return(carr_df, carr_df[((carr_df["LON"] < 106) & (((108-98)*(carr_df["LAT"]-10.5) - (0-10.5)*(carr_df["LON"]-98)) < 0))].index)
    section_15 = drop_return(carr_df, carr_df[((carr_df["LON"] < 110) & (((108-98)*(carr_df["LAT"]-10.5) - (0-10.5)*(carr_df["LON"]-98)) > 0) & (carr_df["LAT"] < 11.6))].index)
    section_16 = carr_df

    # For analyzing we can group chunks into regions
    memnonia = pd.concat([section_1, section_2], axis=0, ignore_index=True)
    tempe = pd.concat([section_3], ignore_index=True)
    chryse = pd.concat([section_4, section_5, section_6], axis=0, ignore_index=True)
    deuteronilus = pd.concat([section_7], ignore_index=True)
    nilosyrtis = pd.concat([section_8, section_9, section_10, section_11, section_12], axis=0, ignore_index=True)
    isidis = pd.concat([section_13], ignore_index=True)
    amenthes = pd.concat([section_14, section_15], axis=0, ignore_index=True)
    nepenthes = pd.concat([section_16], ignore_index=True)

    # Print out the normal stats using the Citron et al. 2018 sea level
    print('STATS FOR CARR AND HEAD 2003 SECTIONS WITH CITRON ET AL 2018 PARAMS')
    print_stats(carr, section=memnonia, name="Memnonia")
    print_stats(carr, section=tempe, name="Tempe")
    print_stats(carr, section=chryse, name="Chryse")
    print_stats(carr, section=deuteronilus, name="Deuteronilus")
    print_stats(carr, section=nilosyrtis, name="Nilosyrtis")
    print_stats(carr, section=isidis, name="Isidis")
    print_stats(carr, section=amenthes, name="Amenthes")
    print_stats(carr, section=nepenthes, name="Nepenthes")

    ##### ----- OPTIMIZE FOR C AND Z ----- #####
    # optimize.minimize(function, parameters [C, sea_level], dataframe, bounds=((C_min, C_max), (sea_level_min, sea_level_max)))
    # sea_level should be set to min and max paleo, but since C is changing those values changes as well, so leaving as None for now
    # don't need to pass sea_level in parameters, rms function can deal without it (by calculating it)
    result_memnonia = optimize.minimize(rms, citron_params, args=(memnonia), bounds=((0,1), (None, None)))
    result_tempe = optimize.minimize(rms, citron_params, args=(tempe), bounds=((0,1), (None, None)))
    result_chryse = optimize.minimize(rms, citron_params, args=(chryse), bounds=((0,1), (None, None)))
    result_deuteronilus = optimize.minimize(rms, citron_params, args=(deuteronilus), bounds=((0,1), (None, None)))
    result_nilosyrtis = optimize.minimize(rms, citron_params, args=(nilosyrtis), bounds=((0,1), (None, None)))
    result_isidis = optimize.minimize(rms, citron_params, args=(isidis), bounds=((0,1), (None, None)))
    result_amenthes = optimize.minimize(rms, citron_params, args=(amenthes), bounds=((0,1), (None, None)))
    result_nepenthes = optimize.minimize(rms, citron_params, args=(nepenthes), bounds=((0,1), (None, None)))

    # CALCULATE OPTIMIZED PALEO TOPOGRAPHIES
    #result.x[0] is C and result.x[1] is sea_level
    memnonia['OPT'] = (memnonia["ELEV"] - result_memnonia.x[0]*memnonia['THARSIS'] - result_memnonia.x[0]*memnonia['TPW'])
    tempe['OPT'] = (tempe["ELEV"] - result_tempe.x[0]*tempe["THARSIS"] - result_tempe.x[0]*tempe["TPW"])
    chryse['OPT'] = chryse["ELEV"] - result_chryse.x[0]*chryse["THARSIS"] - result_chryse.x[0]*chryse["TPW"]
    deuteronilus['OPT'] = deuteronilus["ELEV"] - result_deuteronilus.x[0]*deuteronilus["THARSIS"] - result_deuteronilus.x[0]*deuteronilus["TPW"]
    nilosyrtis['OPT'] = nilosyrtis["ELEV"] - result_nilosyrtis.x[0]*nilosyrtis["THARSIS"] - result_nilosyrtis.x[0]*nilosyrtis["TPW"]
    isidis['OPT'] = isidis["ELEV"] - result_isidis.x[0]*isidis["THARSIS"] - result_isidis.x[0]*isidis["TPW"]
    amenthes['OPT'] = amenthes["ELEV"] - result_amenthes.x[0]*amenthes["THARSIS"] - result_amenthes.x[0]*amenthes["TPW"]
    nepenthes['OPT'] = nepenthes["ELEV"] - result_nepenthes.x[0]*nepenthes["THARSIS"] - result_nepenthes.x[0]*nepenthes["TPW"]

    print('STATS FOR CARR AND HEAD 2003 SECTIONS OPTIMIZED FOR C')
    print_stats(carr, section=memnonia, paleo='OPT', Ci=result_memnonia.x[0], name="Optimized Memnonia")
    print_stats(carr, section=tempe, paleo='OPT', Ci=result_tempe.x[0], name="Optimized Tempe")
    print_stats(carr, section=chryse, paleo='OPT', Ci=result_chryse.x[0], name="Optimized Chryse")
    print_stats(carr, section=deuteronilus, paleo='OPT', Ci=result_deuteronilus.x[0], name="Optimized Deuteronilus")
    print_stats(carr, section=nilosyrtis, paleo='OPT', Ci=result_nilosyrtis.x[0], name="Optimized Nilosyrtis")
    print_stats(carr, section=isidis, paleo='OPT', Ci=result_isidis.x[0], name="Optimized Isidis")
    print_stats(carr, section=amenthes, paleo='OPT', Ci=result_amenthes.x[0], name="Optimized Amenthes")
    print_stats(carr, section=nepenthes, paleo='OPT', Ci=result_nepenthes.x[0], name="Optimized Nepenthes")



    fig, axs = plt.subplots(2,1)
    # Plot the Perron et al. 2007 data segment (replicating the Citron et al. 2018 plot)
    axs[0].plot(perron_data["LON"], perron_data["ELEV"], color="k", marker="^", linewidth=0, label="Modern Topography")
    axs[0].plot(perron_data["LON"], perron_data["PALEO"], 'r.', label="Pre-Tharsis Topography")
    axs[0].plot(perron_data["LON"], perron_data["THARSIS"] + perron_data["TPW"] + sea_citron , 'b', label="TID Model Deformation Curve")  #Model Curve

    # Horizontal Lines for "Sea Levels"
    axs[0].axhline(perron_data["PALEO"].mean(), color='r', alpha=0.3, label="Mean Pre-Tharsis Elevation")
    axs[0].axhline(perron_data["ELEV"].mean(), color='k', alpha=0.3, label="Mean Modern Elevation")
    axs[0].axhline(sea_citron, color='b', alpha=0.3, label="Citron et al. 2018 Sea Level")

    # Plot the rest of the Carr and Head 2003 Arabia Level
    axs[1].plot(carr_data["LON"], carr_data["ELEV"], color='k', marker='^', linewidth=0, label="Modern Topography")
    axs[1].plot(carr_data["LON"], carr_data["PALEO"], color='r', marker='.', linewidth=0, label="Global Paleotopography")

    # Plot the sectioned data curves with Citron sea level
    c_gs = '#F5AB49'
    axs[1].plot(section_1["LON"], section_1["THARSIS"] + section_1["TPW"] + sea_citron, color=c_gs, label="Fixed Global Paleo Sea Level")
    axs[1].plot(section_2["LON"], section_2["THARSIS"] + section_2["TPW"] + sea_citron, color=c_gs)
    axs[1].plot(section_3["LON"], section_3["THARSIS"] + section_3["TPW"] + sea_citron, color=c_gs)
    axs[1].plot(section_4["LON"], section_4["THARSIS"] + section_4["TPW"] + sea_citron, color=c_gs)
    axs[1].plot(section_5["LON"], section_5["THARSIS"] + section_5["TPW"] + sea_citron, color=c_gs)
    axs[1].plot(section_6["LON"], section_6["THARSIS"] + section_6["TPW"] + sea_citron, color=c_gs)
    axs[1].plot(section_7["LON"], section_7["THARSIS"] + section_7["TPW"] + sea_citron, color=c_gs)
    axs[1].plot(section_8["LON"], section_8["THARSIS"] + section_8["TPW"] + sea_citron, color=c_gs)
    axs[1].plot(section_9["LON"], section_9["THARSIS"] + section_9["TPW"] + sea_citron, color=c_gs)
    axs[1].plot(section_10["LON"], section_10["THARSIS"] + section_10["TPW"] + sea_citron, color=c_gs)
    axs[1].plot(section_11["LON"], section_11["THARSIS"] + section_11["TPW"] + sea_citron, color=c_gs)
    axs[1].plot(section_12["LON"], section_12["THARSIS"] + section_12["TPW"] + sea_citron, color=c_gs)
    axs[1].plot(section_13["LON"], section_13["THARSIS"] + section_13["TPW"] + sea_citron, color=c_gs)
    axs[1].plot(section_14["LON"], section_14["THARSIS"] + section_14["TPW"] + sea_citron, color=c_gs)
    axs[1].plot(section_15["LON"], section_15["THARSIS"] + section_15["TPW"] + sea_citron, color=c_gs)
    axs[1].plot(section_16["LON"], section_16["THARSIS"] + section_16["TPW"] + sea_citron, color=c_gs)

    # Plot the optimized model curves
    c_ss = '#6797EB'
    axs[1].plot(section_1['LON'], section_1['THARSIS']*result_memnonia.x[0] + section_1['TPW']*result_memnonia.x[0] + result_memnonia.x[1], color=c_ss, label="Segmented Paleo Sea Levels")
    axs[1].plot(section_2['LON'], section_2['THARSIS']*result_memnonia.x[0] + section_2['TPW']*result_memnonia.x[0] + result_memnonia.x[1], color=c_ss)
    axs[1].plot(section_3['LON'], section_3['THARSIS']*result_tempe.x[0] + section_3['TPW']*result_tempe.x[0] + result_tempe.x[1], color=c_ss)
    axs[1].plot(section_4['LON'], section_4['THARSIS']*result_chryse.x[0] + section_4['TPW']*result_chryse.x[0] + result_chryse.x[1], color=c_ss)
    axs[1].plot(section_5['LON'], section_5['THARSIS']*result_chryse.x[0] + section_5['TPW']*result_chryse.x[0] + result_chryse.x[1], color=c_ss)
    axs[1].plot(section_6['LON'], section_6['THARSIS']*result_chryse.x[0] + section_6['TPW']*result_chryse.x[0] + result_chryse.x[1], color=c_ss)
    axs[1].plot(section_7['LON'], section_7['THARSIS']*result_deuteronilus.x[0] + section_7['TPW']*result_deuteronilus.x[0] + result_deuteronilus.x[1], color=c_ss)
    axs[1].plot(section_8['LON'], section_8['THARSIS']*result_nilosyrtis.x[0] + section_8['TPW']*result_nilosyrtis.x[0] + result_nilosyrtis.x[1], color=c_ss)
    axs[1].plot(section_9['LON'], section_9['THARSIS']*result_nilosyrtis.x[0] + section_9['TPW']*result_nilosyrtis.x[0] + result_nilosyrtis.x[1], color=c_ss)
    axs[1].plot(section_10['LON'], section_10['THARSIS']*result_nilosyrtis.x[0] + section_10['TPW']*result_nilosyrtis.x[0] + result_nilosyrtis.x[1], color=c_ss)
    axs[1].plot(section_11['LON'], section_11['THARSIS']*result_nilosyrtis.x[0] + section_11['TPW']*result_nilosyrtis.x[0] + result_nilosyrtis.x[1], color=c_ss)
    axs[1].plot(section_12['LON'], section_12['THARSIS']*result_nilosyrtis.x[0] + section_12['TPW']*result_nilosyrtis.x[0] + result_nilosyrtis.x[1], color=c_ss)
    axs[1].plot(section_13['LON'], section_13['THARSIS']*result_isidis.x[0] + section_13['TPW']*result_isidis.x[0] + result_isidis.x[1], color=c_ss)
    axs[1].plot(section_14['LON'], section_14['THARSIS']*result_amenthes.x[0] + section_14['TPW']*result_amenthes.x[0] + result_amenthes.x[1], color=c_ss)
    axs[1].plot(section_15['LON'], section_15['THARSIS']*result_amenthes.x[0] + section_15['TPW']*result_amenthes.x[0] + result_amenthes.x[1], color=c_ss)
    axs[1].plot(section_16['LON'], section_16['THARSIS']*result_nepenthes.x[0] + section_16['TPW']*result_nepenthes.x[0] + result_nepenthes.x[1], color=c_ss)


    # Plot the labels:
    axs[1].text(-155, -0.600, 'Memnonia', ha='center')
    axs[1].text(-70, 0, 'Tempe', ha='center')
    axs[1].text(-50,-0.900, 'SW Chryse', ha='center')
    axs[1].text(15, 0, 'Deuteronilus', ha='center')
    axs[1].text(75, 1.100, 'Nilosyrtis', ha='center')
    axs[1].text(92, 0.500, 'Isidis', ha='center')
    axs[1].text(110, 0.900, 'Amenthes\nPlanum', ha='center')
    axs[1].text(130, 0.200, 'Nepenthes', ha='center')


    # Plot the vertical lines showing location of top plot in bottom plot
    axs[1].axvline(x=-28, linestyle='--', alpha=0.3)
    axs[1].axvline(x=78, linestyle='--', alpha=0.3)

    # Add fancy arrows showing placement of top plot
    xy_a = (-28, -4.300)
    xy_b = (-28, 2.000)
    con = ConnectionPatch(
        xyA=xy_a, coordsA=axs[0].transData,
        xyB=xy_b, coordsB=axs[1].transData,
        arrowstyle="->", shrinkB=5)
    fig.add_artist(con)

    xy_a1 = (78, -4.300)
    xy_b1 = (78, 2.000)
    con1 = ConnectionPatch(
        xyA=xy_a1, coordsA=axs[0].transData,
        xyB=xy_b1, coordsB=axs[1].transData,
        arrowstyle="->", shrinkB=5)
    fig.add_artist(con1)

    # Set FIG params
    axs[0].set_ylabel("Elevation [km]")
    axs[1].set_ylabel("Elevation [km]")
    axs[1].set_xlabel("Longitude")
    axs[0].set_xlim(-28,78)
    axs[0].set_ylim(-4.300,0.500)
    axs[1].set_xlim(-180,180)
    axs[1].set_ylim(-6.300,2.000)

    for i in [0,1]:
        axs[i].minorticks_on()
        axs[i].tick_params(bottom=True, top=True, left=True, right=True, direction="in")
        axs[i].grid(which='major', linestyle='-', linewidth=0.5, color='black', alpha=0.1)
        axs[i].grid(which='minor', linestyle=':', linewidth=0.4, color='black', alpha=0.1)

    axs[0].legend(loc='upper left', ncol=2)
    axs[1].legend(loc='lower left', ncol=2)

    fig.subplots_adjust(hspace=0.10)
    plt.show()

def figure2():
    """Plots the modern topography of the different mapped Arabia levels (top) and
    the paleotopography of the different mapped Arabia levels (bottom)."""

    meta_datalist = grab_data(path="input/fig3")

    A_COLORS = ['#EE6677', '#4477AA', '#228833', '#CCBB44', '#BBBBBB', '#AA3377', '#66CCEE', '#000000']
    #           P89         P93         CP01        CH03        P07        W04        P10         S21
    A_LABELS = ['Parker et al. 1989', 'Parker et al. 1993', 'Clifford and Parker 2001', 'Carr and Head 2003', 'Perron et al. 2007', 'Webb 2004', 'Parker et al. 2010', 'Sholes et al. 2021']

    i = 0
    fig, axs = plt.subplots(2,1)

    for data in meta_datalist:
        df_opt = optimize.minimize(rms, citron_params, args=(data.df), bounds=((0,1), (None, None)))
        data.df['OPT'] = (data.df['ELEV'] - df_opt.x[0]*data.df['THARSIS'] - df_opt.x[0]*data.df['TPW'])
        print_stats(data, section=data.df, paleo='OPT', Ci=df_opt.x[0], name=A_LABELS[i])

        # This is for printing the stats for each level without the surrounding Isidis Planitia points
        # (because post-formational deformation by Isidis may cause the extra problems observed)
        noIs = copy.deepcopy(data.df)
        noIs = noIs[(noIs['LON']<77.0) | (noIs['LON']>99.0)]
        noIs_opt = optimize.minimize(rms, citron_params, args=(noIs), bounds=((0,1), (None, None)))
        noIs['OPT'] = (noIs['ELEV'] - noIs_opt.x[0]*noIs['THARSIS'] - noIs_opt.x[0]*noIs['TPW'])
        print_stats(data, section=noIs, paleo='OPT', Ci=noIs_opt.x[0], name=A_LABELS[i] + ' No Isidis')

        axs[0].plot(data.df['LON'], data.df['ELEV'], marker="^", linewidth=0, markersize=3, color=A_COLORS[i], label=A_LABELS[i])
        axs[1].plot(data.df['LON'], data.df['OPT'], marker='.', linewidth=0, markersize=3, color=A_COLORS[i])

        i += 1

    axs[0].text(0, 2.00, "Modern Topography", ha='center', fontsize=14, fontweight='bold')
    axs[1].text(0, 2.00, "Paleotopography", ha='center', fontsize=14, fontweight='bold')

    axs[1].set_xlabel("Longitude")
    axs[0].set_ylim(-6.5,3.0)
    axs[1].set_ylim(-6.500,3.000)

    for j in [0, 1]:
        axs[j].minorticks_on()
        axs[j].set_xlim(-180,180)
        axs[j].set_ylabel('Elevation [km]')
        axs[j].tick_params(bottom=True, top=True, left=True, right=True, direction="in")
        axs[j].grid(which='major', linestyle='-', linewidth=0.5, color='black', alpha=0.1)
        axs[j].grid(which='minor', linestyle=':', linewidth=0.4, color='black', alpha=0.1)

    leg = axs[0].legend(loc="lower left", ncol=4, fontsize=9, handletextpad=0.1)
    for i in range(len(A_LABELS)):
        leg.legendHandles[i]._legmarker.set_markersize(9)
        #leg.legendHandles[i]._sizes = [120]

    fig.subplots_adjust(hspace=0.15)
    plt.show()

def figure3():
    """Plots the Deuteronilus Level modern and paleotopography"""

    ivanov = Shoreline("input/Ivanov2017_Deuteronilus.csv")

    # Define the different regions
    phlegra1 = ivanov.df[:276]
    tantalus = ivanov.df[277:537]
    tempe = ivanov.df[538:5892]
    cydonia = ivanov.df[5893:8335]
    utopia = ivanov.df[8336:13915]
    phlegra2 = ivanov.df[13916:]

    # I add in Phlegra here since the modern topography is so similar, but could also be run as its own region
    tempe_cydonia = pd.concat([phlegra1, tempe, cydonia, phlegra2], axis=0, ignore_index=True)

    # Optimize each region for C
    tempe_opt = optimize.minimize(rms, citron_ivanov, args=(tempe_cydonia), bounds=((0,1), (None, None)))
    utopia_opt = optimize.minimize(rms, citron_ivanov, args=(utopia), bounds=((0,1), (None, None)))
    tantalus_opt = optimize.minimize(rms, citron_ivanov, args=(tantalus), bounds=((0,1), (None, None)))

    # Calculate the best fit paleotopography
    for df in [phlegra1, tempe, cydonia, phlegra2]:
        df['OPT'] = (df['ELEV'] - tempe_opt.x[0]*df['THARSIS'] - tempe_opt.x[0]*df['TPW'])
    tempe_cydonia['OPT'] = (tempe_cydonia['ELEV'] - tempe_opt.x[0]*tempe_cydonia['THARSIS'] - tempe_opt.x[0]*tempe_cydonia['TPW'])
    print_stats(ivanov, section=tempe_cydonia, paleo='OPT', Ci=tempe_opt.x[0], name="Phlegra-Tempe-Cydonia")
    utopia['OPT'] = (utopia['ELEV'] - utopia_opt.x[0]*utopia['THARSIS'] - utopia_opt.x[0]*utopia['TPW'])
    print_stats(ivanov, section=utopia, paleo='OPT', Ci=utopia_opt.x[0], name="Utopia")
    tantalus['OPT'] = (tantalus['ELEV'] - tantalus_opt.x[0]*tantalus['THARSIS'] - tantalus_opt.x[0]*tantalus['TPW'])
    print_stats(ivanov, section=tantalus, paleo='OPT', Ci=tantalus_opt.x[0], name="Tantalus")

    # Colorblind-friendly color codes
    c_now = '#000000'
    c_paleo = '#000000'
    c_citron = '#66CCEE'   #cyan
    a_paleo_sec = 0.5  #alpha

    # PLOT THE FIGURES
    fig, axs = plt.subplots(2,1,sharex=True)

    axs[0].plot(ivanov.df["LON"], ivanov.df["ELEV"], marker="^", linewidth=0, markersize=2, color=c_now)
    axs[1].plot(ivanov.df["LON"], ivanov.df["ELEV"] - citron_ivanov[0]*ivanov.df["THARSIS"] - citron_ivanov[0]*ivanov.df["TPW"], marker='.', linewidth=0, markersize=2, color=c_paleo)

    for i in [0,1]:
        axs[i].minorticks_on()
        axs[i].grid(which='major', linestyle='-', linewidth=0.5, color='black', alpha=0.1)
        axs[i].grid(which='minor', linestyle=':', linewidth=0.4, color='black', alpha=0.1)
        axs[i].set_xlim(-180,180)
        axs[i].set_ylim(-4.4, -2.8)
        axs[i].set_ylabel("Elevation [km]")

        axs[i].text(-178, -3.4, "Phlegra", va='center', fontsize=10)
        axs[i].text(178, -3.25, "Phlegra", ha='right', va='center', fontsize=10)
        axs[i].text(-90, -3.05, "Tantalus", ha='center', va='center', fontsize=10)
        axs[i].text(-30, -3.1, "Tempe-Chryse-Acidalia", ha='center', va='center', fontsize=10)
        axs[i].text(20, -3.25, "Cydonia-Deuteronilus", ha='center', va='center', fontsize=10)
        axs[i].text(100, -3.2, "Pyramus-Astapus-Utopia-Elysium", ha='center', va='center', fontsize=10)

    axs[0].text(0, -2.9, "Modern Topography", ha='center', va='center', fontsize=14, fontweight='bold')
    axs[1].text(0, -2.9, "Paleotopography", ha='center', va='center', fontsize=14, fontweight='bold')
    axs[1].set_xlabel("Longitude")

    dfs = [phlegra1, tantalus, tempe, cydonia, utopia, phlegra2]
    opts = [tempe_opt, tantalus_opt, tempe_opt, tempe_opt, utopia_opt, tempe_opt]
    cs = ['#EE7733', '#CC3311', '#EE7733', '#EE7733', '#EE3377', '#EE7733']
    label_list = [f"C={tempe_opt.x[0]:.2f}, Z={tempe_opt.x[1]:.2f}", f"C={tantalus_opt.x[0]:.2f}, Z={tantalus_opt.x[1]:.2f}", None, None, \
                    f"C={utopia_opt.x[0]:.2f}, Z={utopia_opt.x[1]:.2f}", None]
    citron_label = ["C=0.17, Z=-3.68", None, None, None, None, None]
    i = 0
    for df in dfs:
        # PLOT THE CITRON MODEL CURVES
        axs[0].plot(df["LON"], citron_ivanov[0]*df["THARSIS"] + citron_ivanov[0]*df["TPW"] + citron_ivanov[1], color=c_citron, label=citron_label[i])
        # PLOT THE SECTIONED MODEL
        axs[0].plot(df["LON"], opts[i].x[0]*df["THARSIS"] + opts[i].x[0]*df["TPW"] + opts[i].x[1], \
                    color=cs[i], linestyle='--', label=label_list[i])
        # PLOT THE SECTIONED DATA
        axs[1].plot(df["LON"], df["OPT"], marker='.', linewidth=0, markersize=3, color=cs[i], alpha=a_paleo_sec)
        i += 1

    leg = axs[0].legend(loc='upper left')
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
    fig.subplots_adjust(hspace=0.04)

    plt.show()

def figure4():
    """Plot the paleo and modern topography for the open basin deltas and valley network termini"""

    delta_data = Shoreline('input/fig4/DiAchille2010_Deltas.csv')
    valley_data = Shoreline('input/fig4/Chan2018_S1.csv')
    shoreline_data = Shoreline('input/fig3/3_Carr2003_Arabia_Topo.csv')
    updated_deltas = Shoreline('input/fig4/RiveraHernandezDeltas.csv')

    data = [shoreline_data, valley_data, delta_data, updated_deltas]
    markers_m = ['.', 'P', 'D', '*']
    markers_p = ['^', 'X', 'd', 'p']
    sizes = [1, 6, 6, 6]
    colors_m = ['#000000', '#CC3311', '#EE7733', '#EE3377']
    colors_p = ['#BBBBBB', '#33BBEE', '#0077BB', '#009988']
    alphas = [0.4, 1, 1, 1]
    label_m = ["Putative Shoreline", "Valley Network Termini", "Open Basin Deltas", "Updated Open Basin Deltas"]
    label_p = ["Paleo Putative Shoreline","Paleo Valley Network Termini", "Paleo Open Basin Deltas", "Paleo Updated Open Basin Deltas"]

    fig, axs = plt.subplots(1,1)
    i = 0
    for item in data:
        opt = optimize.minimize(rms, [citron_params], args=(item.df), bounds=((0,1), (None, None)))
        item.df['OPT'] = (item.df['ELEV'] - item.df['THARSIS']*opt.x[0] - item.df['TPW']*opt.x[0])
        axs.plot(item.df['LON'], item.df['ELEV'], marker=markers_m[i], linewidth=0, markersize=sizes[i], color=colors_m[i], alpha=alphas[i], label=label_m[i])
        axs.plot(item.df['LON'], item.df['OPT'], marker=markers_p[i], linewidth=0, markersize=sizes[i], color=colors_p[i], alpha=alphas[i], label=label_p[i])

        print_stats(item, section=item.df, paleo='OPT', Ci=opt.x[0], name=label_m[i])
        i += 1

    axs.set_ylabel("Elevation [km]")
    axs.set_xlabel("Longitude")
    axs.set_xlim(-180,180)
    axs.tick_params(bottom=True, top=True, left=True, right=True, direction="in")
    axs.minorticks_on()
    axs.grid(which='major', linestyle='-', linewidth=0.5, color='black', alpha=0.1)
    axs.grid(which='minor', linestyle=':', linewidth=0.4, color='black', alpha=0.1)
    axs.legend(loc="lower left", ncol=2)
    plt.show()

figure1()
figure2()
figure3()
figure4()
