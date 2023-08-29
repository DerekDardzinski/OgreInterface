import os
from core.interface_generator import *
from core.slab_generator import *
from core.overlap_OF import *
from core.generators import *
import argparse
from itertools import product
import seaborn as sns
from pymatgen.symmetry.analyzer import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--path', dest='path', type=str, default='./test/organic/')
    # parser.add_argument('--isorganic', dest='isorganic',
    #                     type=bool, default=True)
    parser.add_argument('--repair', dest='repair', type=bool, default=True)
    return parser.parse_args()

'''
Usage: 
python main.py --path /test/organic/
python main.py --path /test/organic/ --repair True
python main.py --path /test/inorganic/
'''

def Struc_flip(struc):
    struc_coords = struc.frac_coords
    z_frac_coords = struc_coords[:, 2]
    new_z_frac_coords = []
    for i in z_frac_coords:
        new_z_frac_coords.append(1 - i)

    struc_coords[:, 2] = new_z_frac_coords
    new_struc = Structure(struc.lattice.matrix, species=struc.species, coords=struc_coords)
    new_struc_space = SpacegroupAnalyzer(new_struc)
    new_struc_pri = new_struc_space.get_primitive_standard_structure()
    new_struc = new_struc_pri.get_reduced_structure()

    return new_struc



def main():
    args = parse_arguments()
    working_dir = args.path
    ZSLGenerator.working_dir = working_dir
    # ZSLGenerator.generate_sl_transformation_sets = generate_sl_transformation_sets_Sam
    ZSLGenerator.get_equiv_transformations = get_equiv_transformations_Sam
    Input_dat = open(working_dir+ "Input.dat")
    Dat_list = []
    for i in Input_dat.readlines():
        if "------" in i:
            break
        Dat_list.append(i.split()[1])
    sub_name = working_dir + Dat_list[0]
    film_name = working_dir + Dat_list[1]

    '''
    isorganic = Dat_list[-1]
    if 'T' in isorganic or 't' in isorganic:
        isorganic = True
        if args.repair:
            Possible_surfaces_generator = repair_organic_slab_generator
            # Possible_surfaces_generator = repair_organic_slab_generator_graph
            # Possible_surfaces_generator = orgslab_generator
        else:
            Possible_surfaces_generator = organic_slab_generator
    else:
        isorganic = False
        Possible_surfaces_generator = inorganic_slab_generator
    
    '''
    # To be deleted
    # isorganic = True
    # Possible_surfaces_generator = repair_organic_slab_generator
    ##

    sub_miller = Dat_list[2]
    sub_miller = sub_miller.replace("(", "")
    sub_miller = sub_miller.replace(")", "")
    sub_miller = sub_miller.split(",")
    sub_miller = (int(sub_miller[0]), int(sub_miller[1]), int(sub_miller[2]))
    film_miller = Dat_list[3]
    film_miller = film_miller.replace("(", "")
    film_miller = film_miller.replace(")", "")
    film_miller = film_miller.split(",")
    film_miller = (int(film_miller[0]), int(
        film_miller[1]), int(film_miller[2]))
    sub_layers = int(Dat_list[4])
    film_layers = int(Dat_list[5])
    vacuum = int(Dat_list[6])
    max_area = int(Dat_list[7])
    max_area_ratio_tol = float(Dat_list[8])
    max_length_tol = float(Dat_list[9])
    max_angle_tol = float(Dat_list[10])
    distance = float(Dat_list[11])
    nStruc = int(Dat_list[12])


    if (Dat_list[13] == "True") or (Dat_list[13] == "true"):
        shift_gen = True
    else:
        shift_gen = False

    x_range = Dat_list[14]
    y_range = Dat_list[16]
    z_range = Dat_list[18]
    try:
        if x_range != "None":
            x_range = x_range.replace("[", "")
            x_range = x_range.replace("]", "")
            x_range = x_range.split(",")
            x_range = [float(x_range[0]), float(x_range[1])]
        if y_range != "None":
            y_range = y_range.replace("[", "")
            y_range = y_range.replace("]", "")
            y_range = y_range.split(",")
            y_range = [float(y_range[0]), float(y_range[1])]
        if z_range != "None":
            z_range = z_range.replace("[", "")
            z_range = z_range.replace("]", "")
            z_range = z_range.split(",")
            z_range = [float(z_range[0]), float(z_range[1])]
    except:
        pass

    x_steps = int(Dat_list[15])
    y_steps = int(Dat_list[17])
    z_steps = int(Dat_list[19])
    if (Dat_list[20] == "True") or (Dat_list[20] == "true"):
        scoring = True
    else:
        scoring = False
    fparam = 0.5

    if (Dat_list[21] == "True") or (Dat_list[21] == "true"):
        is_organic = True
        isorganic = True
    else:
        is_organic = False
        isorganic = False

    if (Dat_list[22] == "True") or (Dat_list[22] == "true"):
        is_miller_search = True
    else:
        is_miller_search = False

    max_sub_index = int(Dat_list[23])
    max_film_index = int(Dat_list[24])

    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    # start
    ####################################################################
    ase_substrate = read(sub_name)
    ase_film = read(film_name)
    pmg_substrate = Structure.from_file(sub_name, primitive=False)
    pmg_film = Structure.from_file(film_name, primitive=False)
    # Finding possible superlattice tranformation matrices

    try:
        os.remove(working_dir+ "misfit_ratios")
    except:
        pass

    ZSL = ZSLGenerator(max_area_ratio_tol=max_area_ratio_tol, max_area=max_area,
                       max_length_tol=max_length_tol, max_angle_tol=max_angle_tol)

    Sub_analyzer = SubstrateAnalyzer(ZSL, film_max_miller=10)

    if is_miller_search:

        def misfit_extractor(txt):
            area_misfits = []
            txt_lines = txt.readlines()
            if len(txt_lines) == 0:
                area_misfit = 1
            else:
                for l in txt_lines:
                    l_split = l.split()
                    area_misfits.append(float(l_split[1]))
                area_misfit = area_misfits[0]

            return area_misfit

        # sub_elements = list(range(-max_sub_index, max_sub_index + 1))
        sub_elements = list(range(0, max_sub_index + 1))
        # film_elements = list(range(-max_film_index, max_film_index + 1))
        film_elements = list(range(0, max_film_index + 1))

        sub_indices = list(product(sub_elements, repeat=3))
        film_indices = list(product(film_elements, repeat=3))
        # x_indices = [(0, 0, 1), (0, 1, 1), (1, 1, 1), (0, 0, 2), (0, 1, 2), (0, 2, 2), (1, 1, 2), (1, 2, 2), (2, 2, 2)]


        '''
        x_indices = [(0,1,0), (1,0 , 0), (1, 1, 0), (0, 0, 1), (0,1,1), (1, 0, 1), (1, 1, 1)]
        sub_indices = x_indices
        film_indices = [(0,1,0),(1,1,0),(1,1,1)]
        '''

        sub_indices = [(0,1,0), (1,0,0), (1,1,0), (0,0,1), (0,1,1), (1,0,1), (1,1,1), (-1,0,1), (-1,1,1)]
        film_indices = [(0,1,0), (1,0,0), (1,1,0), (0,0,1), (0,1,1), (1,0,1), (1,1,1), (-1,0,1), (-1,1,1)]




        #film_indices = [(0,1,0), (0,2,0), (1,1,0), (1,2,0),(2,1,0),(2,2,0),(1,1,1),(1,2,1),(2,2,1),(2,2,2) ]
        area_misfit_2d = []
        misfit_f_list = []
        miller_f_list = []

        # for sub_mi_index in range(len(sub_indices)) :
        #     sub_mi = sub_indices[sub_mi_index]
        #     ext_new = "Results_" + str(sub_mi)
        #     ext_Dir_new = os.path.join(working_dir, ext_new)
        #     if (os.path.isdir(ext_Dir_new) == False):
        #         os.mkdir(ext_Dir_new)
        #     res_list = []
        #     area_misfit_1d = []
        #     for film_mi_index in range(len(film_indices)):
        #         film_mi = film_indices[film_mi_index]
        #         try:
        #             Match_finder = Sub_analyzer.calculate(film=pmg_film, substrate=pmg_substrate,
        #                                                   substrate_millers=[sub_mi], film_millers=[film_mi])
        #             Match_list = list(Match_finder)
        #             res_file = open(working_dir + "Ogre_output")
        #             area_misfit = misfit_extractor(res_file)
        #             area_misfit_1d.append(area_misfit)
        #             res_lines = res_file.readlines()
        #             os.rename(working_dir + "Ogre_output",
        #                       ext_Dir_new + "/" + "Ogre_output" + str(sub_mi) + "_" + str(film_mi))
        #
        #             misfit_f_list.append(np.round(area_misfit, 8))
        #             miller_f_list.append("sub: " + str(sub_mi)+ "/ film: "+ str(film_mi))
        #
        #         except Exception as e:
        #             area_misfit_1d.append(1)
        #             pass
        #     area_misfit_2d.append(area_misfit_1d)
        ext_new = "Scan_Results"
        ext_Dir_new = os.path.join(working_dir, ext_new)
        if (os.path.isdir(ext_Dir_new) == False):
            os.mkdir(ext_Dir_new)
        total_int = 0
        for film_mi_index in range(len(film_indices)) :
            film_mi = film_indices[film_mi_index]
            res_list = []
            area_misfit_1d = []
            for sub_mi_index in range(len(sub_indices)):
                sub_mi = sub_indices[sub_mi_index]
                try:
                    Match_finder = Sub_analyzer.calculate(film=pmg_film, substrate=pmg_substrate,
                                                          substrate_millers=[sub_mi], film_millers=[film_mi])
                    Match_list = list(Match_finder)
                    total_int += len(Match_list)

                    res_file = open(working_dir + "Ogre_output")
                    area_misfit = misfit_extractor(res_file)
                    area_misfit_1d.append(area_misfit )
                    res_lines = res_file.readlines()
                    os.rename(working_dir + "Ogre_output",
                              ext_Dir_new + "/" + "Ogre_output" + str(sub_mi) + "_" + str(film_mi))

                    misfit_f_list.append(np.round(area_misfit, 8) * 100)
                    miller_f_list.append("sub: " + str(sub_mi)+ "/ film: "+ str(film_mi))

                except Exception as e:
                    area_misfit_1d.append(1)
                    pass
            area_misfit_2d.append(area_misfit_1d)

        print("----------------------------------------------")
        print(total_int)
        print("----------------------------------------------")

        min_misfit_val = min(misfit_f_list)
        misfit_counter = 0
        for misfit_value in misfit_f_list:
            if misfit_value == min_misfit_val:
                print("misfit_val:" ,misfit_value ," ", miller_f_list[misfit_counter])
            misfit_counter += 1

        area_mifit_array = np.array(area_misfit_2d)
        for i_iter in range(len(area_mifit_array)):
            for j_iter in range(len(area_mifit_array[0])):
                if area_mifit_array[i_iter, j_iter] == 1:
                    area_mifit_array[i_iter, j_iter] = np.nan
        x_tick_list = []
        y_tick_list = []
        for i_index in sub_indices:
            x_tick_list.append(str(i_index))
        for j_index in film_indices:
            y_tick_list.append(str(j_index))

        fig_dim_x = int(len(area_mifit_array) )
        fig_dim_y = int(len(area_mifit_array[0]) )
        # plt.figure(figsize=(fig_dim_x, fig_dim_y))
        # plt.figure(figsize=(13, 13))
        # Transpose just added
        # trans_area_misfit_array =  area_mifit_array.T
        fig = plt.figure(figsize=(16, 14))
        ax1 = fig.subplots()
        g = sns.heatmap(area_mifit_array , cmap=plt.cm.jet, cbar_kws={'label': 'Misfit Percentage'}, linewidths=1, linecolor='black')
        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontsize(32)

        cax = plt.gca().figure.axes[-1]
        # cax.yaxis.label.set_size(16)
        # cax.yaxis.labelpad = 6
        cax.tick_params(labelsize=32)

        # g = sns.heatmap(trans_area_misfit_array, cmap=plt.cm.jet)
        g.set_xticklabels(x_tick_list, rotation=40, fontsize = 32)
        g.set_yticklabels(y_tick_list, rotation=30, fontsize = 32)
        # plt.xlabel("Substrate (TTF) Miller index", fontsize = 14)
        # plt.ylabel("Film (TCNQ) Miller index", fontsize = 14)
        # plt.title("Misfit plot", fontsize = 17)
        plt.savefig(working_dir + "Miller_search_result.png")



    Match_finder = Sub_analyzer.calculate(film=pmg_film, substrate=pmg_substrate, substrate_millers=[sub_miller],
                                          film_millers=[film_miller])
    Match_list = list(Match_finder)

    # Cleave slabs
    if is_organic == False:
        substrate_slabs = Possible_surfaces_generator(
            ase_substrate, sub_miller, sub_layers, vacuum, working_dir)
        film_slabs = Possible_surfaces_generator(
            ase_film, film_miller, film_layers, vacuum, working_dir)
    else:
        substrate_slab_generator = OrganicSlabGenerator(ase_substrate, sub_miller, [sub_layers], vacuum,
                                                        [1, 1, 1],  working_dir)
        film_slab_generator = OrganicSlabGenerator(ase_film, film_miller, [film_layers], vacuum,
                                                        [1, 1, 1], working_dir)
        substrate_slabs = substrate_slab_generator.cleave()[0]
        film_slabs = film_slab_generator.cleave()[0]

    ##################################
    # importing generated matrices

    sets = open(working_dir + "matrices_sets")
    film_sub_matices = []
    sets = sets.readlines()
    for x in sets:
        x = x.replace("[", "")
        x = x.replace("]", "")
        x = x.split(",")
        film_mat = [[float(x[0]), float(x[1])], [float(x[2]), float(x[3])]]
        sub_mat = [[float(x[4]), float(x[5])], [float(x[6]), float(x[7])]]
        film_sub_matices.append([film_mat, sub_mat])

    if len(film_sub_matices) <= nStruc:
        nStruc = len(film_sub_matices)

    for i in range(len(substrate_slabs)):
    # for i in [0]:
        for j in range(len(film_slabs)):
            ext = "Interfaces_Sub_" + str(i) + "_Film_" + str(j)
            ext_Dir = os.path.join(working_dir, ext)
            if (os.path.isdir(ext_Dir) == False):
                os.mkdir(ext_Dir)

            k_num = 0
            for k in range(nStruc):
                # for k in [1]:
                Interface = Interface_generator(substrate_slabs[i], film_slabs[j], film_sub_matices[k][1],
                                                film_sub_matices[k][0], distance, fparam)
                Iface = Interface[0]
                sub_coords = Interface[2]
                film_coords = Interface[3]
                sub_slab_struc = Interface[4]
                film_slab_struc = Interface[5]
                Poscar(sub_slab_struc).write_file(ext_Dir + "/POSCAR_sub_" + str(sub_miller), direct=False)
                Poscar(film_slab_struc).write_file(ext_Dir + "/POSCAR_film_" + str(
                    film_miller), direct=False)

                print("###################################################")
                print("############# Slab generation is done #############")
                print("###################################################")

                '''
                print("#########################################################")
                print("#############Score calculation start#####################")
                print("#########################################################")
                rel_OL, rel_empt, int_score1, int_score2 = surface_probe_interface(sub_slab_struc, film_slab_struc,
                                                                          probe_rad=1.2,
                                                          calc_roughness= False, calc_OL= True,  visualize= True
                        , scan_step= 0.8)
                print("----------------------Results----------------------------")
                print("Sub_MI:", sub_miller, "Film_MI:",film_miller, "Rel_OL:", rel_OL, "Rel_Empt:", rel_empt, \
                                                                              "Score1:",  int_score1, "score2:", \
                                                                              int_score2)
                Is_right_handed = Interface[6]
                '''


                # Very important

                # if Is_right_handed == False:
                #     continue
                # else:
                #     if k > 0:
                #         k_num += 1

                if scoring:
                    if is_organic == False:
                        Poscar(sub_slab_struc.get_primitive_structure()).write_file(ext_Dir + "/POSCAR_sub", direct=False)
                        Poscar(film_slab_struc.get_primitive_structure()).write_file(ext_Dir + "/POSCAR_film", direct=False)

                        OL_sub_Input = PBC_coord_gen(sub_slab_struc, ext_Dir, k, 0,
                                                     1, 1, file_gen=True)
                        OL_film_Input = PBC_coord_gen(film_slab_struc, ext_Dir, k, 1,
                                                      1, 1, file_gen=True)

                        sub_OL_vol = Mont_geo(OL_sub_Input[0], OL_sub_Input[1])
                        film_OL_vol = Mont_geo(OL_film_Input[0], OL_film_Input[1])
                        sub_sp_vol = 0
                        film_sp_vol = 0
                        for sp in sub_slab_struc.species:
                            sub_sp_vol += sphe_vol(rad_dic[str(sp)])
                        for sp in film_slab_struc.species:
                            film_sp_vol += sphe_vol(rad_dic[str(sp)])

                        sub_OL = sub_OL_vol / sub_sp_vol
                        film_OL = film_OL_vol / film_sp_vol
                        average_OL = (sub_OL + film_OL) / 2

                sub_coords = Interface[2]
                film_coords = Interface[3]
                # Poscar(Iface).write_file(ext_Dir + "/POSCAR_Iface_" + str(k), direct=False)
                if shift_gen:
                    # print(xy_steps)
                    if x_range != "None":
                        x_grid_range = np.linspace(x_range[0], x_range[1], x_steps)
                        x_ref = film_coords[:, 0].copy()
                    else:
                        x_grid_range = [0]
                    if y_range != "None":
                        y_grid_range = np.linspace(y_range[0], y_range[1], y_steps)
                        y_ref = film_coords[:, 1].copy()
                    else:
                        y_grid_range = [0]
                    if z_range != "None":
                        z_grid_range = np.linspace(z_range[0], z_range[1], z_steps)
                        z_ref = film_coords[:, 2].copy()
                    else:
                        z_grid_range = [0]

                if shift_gen:
                    ext2 = "Interfaces_Sub_" + str(i) + "_Film_" + str(j) + "/Shifts_Iface_" + str(k_num)
                    ext_Dir2 = os.path.join(working_dir, ext2)
                    if (os.path.isdir(ext_Dir2) == False):
                        os.mkdir(ext_Dir2)

                    Score_array = np.zeros((len(x_grid_range), len(y_grid_range), len(z_grid_range)))
                    OL_array = np.zeros((len(x_grid_range), len(y_grid_range), len(z_grid_range)))

                    interface_latt = Iface.lattice.matrix
                    count_x = 1
                    for ii in range(len(x_grid_range)):
                        for jj in range(len(y_grid_range)):
                            for kk in range(len(z_grid_range)):
                                print(count_x)
                                count_x += 1

                                if x_range != "None":
                                    film_coords[:, 0] = x_ref + x_grid_range[ii]
                                if y_range != "None":
                                    film_coords[:, 1] = y_ref + y_grid_range[jj]
                                if z_range != "None":
                                    film_coords[:, 2] = z_ref + z_grid_range[kk]

                                int_coords = np.concatenate((sub_coords, film_coords), axis=0)
                                new_int = Structure(Iface.lattice.matrix, Iface.species, int_coords,
                                                    coords_are_cartesian=True)
                                reduced_int = new_int.get_reduced_structure()

                                reduced_int = Struc_flip(reduced_int)
                                # reduced_int = reduced_int.get_reduced_structure()
                                if isorganic == True:
                                    Poscar(reduced_int).write_file(ext_Dir2 +
                                                                                             "/POSCAR_Iface_" + str(
                                        k_num)
                                                                                             + "_x_" + str(
                                        int(x_grid_range[ii] * 10)) + "_y_" + str(
                                        int(y_grid_range[jj] * 10)) + "_z_" + str(int(z_grid_range[kk] * 10)),
                                                                                             direct= False)

                                    # reduced_int_SC = new_int
                                    # reduced_int_SC.make_supercell([3, 3, 1])
                                    # int_coords_SC = reduced_int_SC.cart_coords
                                    # int_SC_sp = reduced_int_SC.species

                                    # min_SC_z = min(int_coords[:,[2]])
                                    # min_int_z = min(int_coords[:,[2]])

                                    #######################
                                    # Poscar(reduced_int.get_reduced_structure()).write_file(
                                    #     ext_Dir2 + "/POSCAR_red_" + str(k)
                                    #     + "_x_" + str(int(x_grid_range[ii] * 10)) + "_y_" + str(
                                    #         int(y_grid_range[jj] * 10)) + "_z_" + str(int(z_grid_range[kk] * 10)),
                                    #     direct=False)
                                    # Poscar(reduced_int_SC).write_file(
                                    #     ext_Dir2 + "/POSCAR_SC_" + str(k)
                                    #     + "_x_" + str(int(x_grid_range[ii] * 10)) + "_y_" + str(
                                    #         int(y_grid_range[jj] * 10)) + "_z_" + str(int(z_grid_range[kk] * 10)),
                                    #     direct=False)




                                # reduced_int = reduced_int.get_reduced_structure()
                                # int_coords = reduced_int.cart_coords
                                if scoring:

                                    if isorganic == False:
                                        sub_sp_num = Interface[1][0]
                                        film_sp_num = Interface[1][1]
                                        # print(len(sub_coords) + len(film_coords))
                                        sub_slab_zmat = sub_coords[:, [2]]
                                        modif_film_slab_zmat = film_coords[:, [2]]

                                        sub_max_list = coords_sperator(sub_slab_zmat, 5, True)
                                        film_min_list = coords_sperator(modif_film_slab_zmat, 5, False)
                                        surf_int_species = []
                                        surf_int_coords = []

                                        surf_sub_sp = []
                                        surf_sub_coords = []
                                        surf_film_sp = []
                                        surf_film_coords = []

                                        for kkk2 in range(len(int_coords)):
                                            for k1 in sub_max_list:
                                                if (round(int_coords[kkk2, 2], 8) == round(k1[0], 8)):
                                                    surf_sub_coords.append(int_coords[kkk2, :])
                                                    surf_sub_sp.append(Iface.species[kkk2])
                                            for k2 in film_min_list:
                                                if (round(int_coords[kkk2, 2], 8) == round(k2[0], 8)):
                                                    surf_film_coords.append(int_coords[kkk2, :])
                                                    surf_film_sp.append(Iface.species[kkk2])

                                        for kkk in range(len(int_coords)):
                                            for k1 in sub_max_list:
                                                if (round(int_coords[kkk, 2], 8) == round(k1[0], 8)):
                                                    surf_int_coords.append(int_coords[kkk, :])
                                                    surf_int_species.append(Iface.species[kkk])
                                            for k2 in film_min_list:
                                                if (round(int_coords[kkk, 2], 8) == round(k2[0], 8)):
                                                    surf_int_coords.append(int_coords[kkk, :])
                                                    surf_int_species.append(Iface.species[kkk])


                                        for kkk in range(len(int_coords)):
                                            for k1 in sub_max_list:
                                                if (round(int_coords[kkk, 2], 8) == round(k1[0], 8)):
                                                    surf_int_coords.append(int_coords[kkk, :])
                                                    surf_int_species.append(Iface.species[kkk])
                                            for k2 in film_min_list:
                                                if (round(int_coords[kkk, 2], 8) == round(k2[0], 8)):
                                                    surf_int_coords.append(int_coords[kkk, :])
                                                    surf_int_species.append(Iface.species[kkk])

                                        surf_int_coords = np.array(surf_int_coords)
                                        surf_film_coords = np.array(surf_film_coords)
                                        surf_int_coords = np.array(surf_int_coords)
                                        # print(len(surf_int_coords))
                                        surf_struc = Structure(Iface.lattice.matrix, surf_int_species, surf_int_coords,
                                                               coords_are_cartesian=True)
                                        # surf_sub_struc2 = Structure(Iface.lattice.matrix, surf_sub_sp, surf_sub_coords,
                                        #                             coords_are_cartesian=True)
                                        # surf_film_struc2 = Structure(Iface.lattice.matrix, surf_film_sp, surf_film_coords,
                                        #                              coords_are_cartesian=True)
                                        # surf_sub_coords = np.array(surf_sub_coords)
                                        # surf_film_coords = np.array(surf_film_coords)
                                        # surf_sub_struc = Structure(Iface.lattice.matrix, surf_sub_sp , surf_sub_coords,
                                        #                            coords_are_cartesian= True)
                                        # surf_film_struc = Structure(Iface.lattice.matrix, surf_film_sp, surf_film_coords,
                                        #                             coords_are_cartesian= True)

                                        # Poscar(surf_sub_struc.get_reduced_structure()).write_file(ext_Dir2 + "/POSCAR_s_sub" + str(k)
                                        #                                + "_x_" + str(int(x_grid_range[ii] * 10)) + "_y_" + str(
                                        #     int(y_grid_range[jj] * 10)) + "_z_" + str(int(z_grid_range[kk] * 10)), direct=False)

                                        Poscar(surf_struc.get_reduced_structure()).write_file(ext_Dir2 + "/POSCAR_surf_" + str(k)
                                                                       + "_x_" + str(int(x_grid_range[ii] * 10)) + "_y_" + str(
                                            int(y_grid_range[jj] * 10)) + "_z_" + str(int(z_grid_range[kk] * 10)), direct=False)

                                        surf_struc = surf_struc.get_reduced_structure()
                                        surf_sub_struc2 = surf_sub_struc2.get_reduced_structure()
                                        surf_film_struc2 = surf_film_struc2.get_reduced_structure()

                                        z_len = max(surf_struc.cart_coords[:, 2]) - min(surf_struc.cart_coords[:, 2])
                                        z_len += 2 * max_rad
                                        reduced_int2 = new_int.get_reduced_structure()

                                        reduced_int2.make_supercell([3, 3, 1])

                                        OL_Input = PBC_coord_gen(reduced_int, ext_Dir2, k, int(x_grid_range[ii] * 10),
                                                                 int(y_grid_range[jj] * 10), int(z_grid_range[kk] * 10),
                                                                 file_gen=False)

                                        OL_Input22 = PBC_coord_gen(reduced_int2, ext_Dir2, k, int(x_grid_range[ii] * 10),
                                                                 int(y_grid_range[jj] * 10), int(z_grid_range[kk] * 10),
                                                                 file_gen=True)

                                        OL_Input_sub = PBC_coord_gen(surf_sub_struc2, " ", 1, 1, 1, 1, file_gen=False)
                                        OL_Input_film = PBC_coord_gen(surf_film_struc2, " ", 1, 1, 1, 1, file_gen=False)

                                        CC_dic = OL_Input[2]
                                        vec1_len = np.linalg.norm(surf_struc.lattice.matrix[0]) + 2 * max_rad
                                        vec2_len = np.linalg.norm(surf_struc.lattice.matrix[1]) + 2 * max_rad
                                        vec_ang = angle(surf_struc.lattice.matrix[0], surf_struc.lattice.matrix[1])
                                        surf_cube_vol = np.sin(vec_ang) * vec1_len * vec2_len * z_len
                                        surf_tot_vol = 0
                                        for i_sp in surf_struc.species:
                                            surf_tot_vol += sphe_vol(rad_dic[str(i_sp)])
                                        int_OL_vol = Mont_geo(OL_Input[0], OL_Input[1])- Mont_geo(OL_Input_sub[0], OL_Input_sub[1])
                                        - Mont_geo(OL_Input_film[0], OL_Input_film[1])
                                        un_occ_vol = surf_cube_vol - (surf_tot_vol - int_OL_vol)

                                        Score_array[ii, jj, kk] = (0.5 * int_OL_vol + 0.5 * un_occ_vol)
                                        OL_array[ii, jj, kk] = int_OL_vol

                                    Poscar(reduced_int.get_primitive_structure()).write_file(ext_Dir2 +
                                                                                            "/POSCAR_Iface_" + str(k_num)
                                                                   + "_x_" + str(int(x_grid_range[ii] * 10)) + "_y_" + str(
                                        int(y_grid_range[jj] * 10)) + "_z_" + str(int(z_grid_range[kk] * 10)), direct=True)



                                    if isorganic == True:
                                        # print(ii, jj, kk)
                                        sub_sp_num = Interface[1][0]
                                        film_sp_num = Interface[1][1]
                                        # print(len(sub_coords) + len(film_coords))
                                        sub_slab_zmat = sub_coords[:, [2]]
                                        modif_film_slab_zmat = film_coords[:, [2]]

                                        sub_max_list = coords_sperator(sub_slab_zmat, 5, True)
                                        # film_min_list = coords_sperator(modif_film_slab_zmat, 5, False)
                                        film_min_list = coords_sperator(modif_film_slab_zmat, 5, False)
                                        # print("$$$$$$$$$$$$$$")
                                        # print(sub_max_list)
                                        # print("$$$$$$$$$$$$$$$$$444")
                                        surf_int_species = []
                                        surf_int_coords = []

                                        surf_sub_sp = []
                                        surf_sub_coords = []
                                        surf_film_sp = []
                                        surf_film_coords = []

                                        # new_int = Structure(Iface.lattice.matrix, Iface.species, int_coords,
                                        #                     coords_are_cartesian=True)
                                        # reduced_int = new_int.get_reduced_structure()
                                        # reduced_int_SC = reduced_int.get_reduced_structure()
                                        # reduced_int_SC.make_supercell([2, 2, 1] )
                                        # reduced_int_SC = reduced_int_SC.get_reduced_structure()
                                        # int_coords_SC = reduced_int_SC.cart_coords
                                        # int_SC_sp = reduced_int_SC.species
                                        #
                                        # Poscar(reduced_int_SC).write_file(ext_Dir2 + "/POSCAR_SC_" + str(k)
                                        #                                + "_x_" + str(
                                        #     int(x_grid_range[ii] * 10)) + "_y_" + str(
                                        #     int(y_grid_range[jj] * 10)) + "_z_" + str(int(z_grid_range[kk] * 10)),
                                        #                                direct=True)

                                        for kkk2 in range(len(int_coords)):
                                            for k1 in sub_max_list:
                                                if (round(int_coords[kkk2, 2], 8) == round(k1[0], 8)):
                                                    surf_sub_coords.append(int_coords[kkk2, :])
                                                    surf_sub_sp.append(Iface.species[kkk2])
                                            for k2 in film_min_list:
                                                if (round(int_coords[kkk2, 2], 8) == round(k2[0], 8)):
                                                    surf_film_coords.append(int_coords[kkk2, :])
                                                    surf_film_sp.append(Iface.species[kkk2])

                                        # print("$$$$$$$$$$$$$$$$$$$$$")
                                        # print(len(int_coords_SC))
                                        # print("$$$$$$$$$$$$$$$$$$$$$")

                                        # for kkk2 in range(len(int_coords_SC)):
                                        #     for k1 in sub_max_list:
                                        #         print(round(abs(int_coords_SC[kkk2, 2]) , 8), "  ",round(k1[0], 8)  )
                                        #         if (round(abs(int_coords_SC[kkk2, 2]) , 8) == round(k1[0], 8)):
                                        #             surf_sub_coords.append(int_coords_SC[kkk2, :])
                                        #             surf_sub_sp.append(int_SC_sp[kkk2])
                                        #     for k2 in film_min_list:
                                        #         print(round(abs(int_coords_SC[kkk2, 2]), 8), "  ", round(k2[0], 8))
                                        #         if (round(abs(int_coords_SC[kkk2, 2]) , 8) == round(k2[0], 8)):
                                        #             surf_film_coords.append(int_coords_SC[kkk2, :])
                                        #             surf_film_sp.append(int_SC_sp[kkk2])
                                        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4")
                                        # print(min_int_z, min_SC_z)
                                        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4")
                                        # print(surf_sub_sp)
                                        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4")




                                        #Original
                                        # for kkk2 in range(len(int_coords_SC)):
                                        #     for k1 in sub_max_list:
                                        #         print("@@@@@")
                                                # print(round(int_coords_SC[kkk2, 2], 8))
                                                # if (round(int_coords_SC[kkk2, 2], 8) == round(k1[0], 8)):
                                                #     surf_sub_coords.append(int_coords_SC[kkk2, :])
                                                #     surf_sub_sp.append(int_SC_sp[kkk2])
                                            # for k2 in film_min_list:
                                            #     if (round(int_coords_SC[kkk2, 2], 8) == round(k2[0], 8)):
                                            #         surf_film_coords.append(int_coords_SC[kkk2, :])
                                            #         surf_film_sp.append(int_SC_sp[kkk2])

                                        '''
                                        for kkk2 in range(len(int_coords_SC)):
                                            for k1 in sub_max_list:
                                                # print("@@@@@")
                                                # print(round(int_coords_SC[kkk2, 2], 8))
                                                if (abs(round( int_coords_SC[kkk2, 2], 8))  == round(k1[0], 8)):
                                                    surf_sub_coords.append(int_coords_SC[kkk2, :])
                                                    surf_sub_sp.append(int_SC_sp[kkk2])
                                            for k2 in film_min_list:
                                                if (abs(round( int_coords_SC[kkk2, 2], 8))  == round(k2[0], 8)):
                                                    surf_film_coords.append(int_coords_SC[kkk2, :])
                                                    surf_film_sp.append(int_SC_sp[kkk2])
                                        '''




                                        for kkk in range(len(int_coords)):
                                            for k1 in sub_max_list:
                                                if (round(int_coords[kkk, 2], 8) == round(k1[0], 8)):
                                                    surf_int_coords.append(int_coords[kkk, :])
                                                    surf_int_species.append(Iface.species[kkk])
                                            for k2 in film_min_list:
                                                if (round(int_coords[kkk, 2], 8) == round(k2[0], 8)):
                                                    surf_int_coords.append(int_coords[kkk, :])
                                                    surf_int_species.append(Iface.species[kkk])
                                        # print("88888888888888")
                                        # print(np.array(surf_int_coords)[:,2])
                                        # for kkk in range(len(int_coords)):
                                        #     for k1 in sub_max_list:
                                        #         if (round(int_coords[kkk, 2], 8) == round(k1[0], 8)):
                                        #             surf_int_coords.append(int_coords[kkk, :])
                                        #             surf_int_species.append(Iface.species[kkk])
                                        #     for k2 in film_min_list:
                                        #         if (round(int_coords[kkk, 2], 8) == round(k2[0], 8)):
                                        #             surf_int_coords.append(int_coords[kkk, :])
                                        #             surf_int_species.append(Iface.species[kkk])
                                        #
                                        surf_sub_coords = np.array(surf_sub_coords)
                                        surf_film_coords = np.array(surf_film_coords)
                                        surf_int_coords = np.array(surf_int_coords)


                                        # print(len(surf_int_coords))
                                        # surf_struc = Structure(Iface.lattice.matrix, surf_int_species, surf_int_coords,
                                        #                        coords_are_cartesian=True)

                                        # print("################")
                                        # print(surf_sub_sp)
                                        # print("################")

                                        surf_sub_struc2 = Structure(Iface.lattice.matrix, surf_sub_sp,
                                                                    surf_sub_coords,
                                                                    coords_are_cartesian=True)

                                        surf_film_struc2 = Structure(Iface.lattice.matrix, surf_film_sp,
                                                                     surf_film_coords,
                                                                     coords_are_cartesian=True)
                                        # surf_sub_coords = np.array(surf_sub_coords)
                                        # surf_film_coords = np.array(surf_film_coords)
                                        # surf_sub_struc = Structure(Iface.lattice.matrix, surf_sub_sp , surf_sub_coords,
                                        #                            coords_are_cartesian= True)
                                        # surf_film_struc = Structure(Iface.lattice.matrix, surf_film_sp, surf_film_coords,
                                        #                             coords_are_cartesian= True)

                                        # Poscar(surf_sub_struc.get_reduced_structure()).write_file(ext_Dir2 + "/POSCAR_s_sub" + str(k)
                                        #                                + "_x_" + str(int(x_grid_range[ii] * 10)) + "_y_" + str(
                                        #     int(y_grid_range[jj] * 10)) + "_z_" + str(int(z_grid_range[kk] * 10)), direct=False)

                                        # Poscar(surf_sub_struc2.get_reduced_structure()).write_file(ext_Dir2 + "/POSCAR_sub_" + str(k)
                                        #                                + "_x_" + str(int(x_grid_range[ii] * 10)) + "_y_" + str(
                                        #     int(y_grid_range[jj] * 10)) + "_z_" + str(int(z_grid_range[kk] * 10)), direct=False)

                                        # Poscar(surf_film_struc2.get_reduced_structure()).write_file(
                                        #     ext_Dir2 + "/POSCAR_film_" + str(k)
                                        #     + "_x_" + str(int(x_grid_range[ii] * 10)) + "_y_" + str(
                                        #         int(y_grid_range[jj] * 10)) + "_z_" + str(int(z_grid_range[kk] * 10)),
                                        #     direct=False)
                                        # PBC_film_Input = PBC_coord_gen(surf_film_struc2.get_reduced_structure(), ext_Dir2, k, int(x_grid_range[ii] * 10),
                                        #                          int(y_grid_range[jj] * 10), int(z_grid_range[kk] * 10),
                                        #                          file_gen=True)

                                        # surf_struc = surf_struc.get_reduced_structure()


                                        surf_sub_struc2 = surf_sub_struc2.get_reduced_structure()
                                        surf_film_struc2 = surf_film_struc2.get_reduced_structure()
                                        # surf_sub_struc2.make_supercell([3, 3, 1])
                                        # surf_film_struc2.make_supercell([3, 3, 1])
                                        # Poscar(surf_sub_struc2).write_file(ext_Dir2 + "/POSCAR_sub_" + str(k)
                                        #                                + "_x_" + str(
                                        #     int(x_grid_range[ii] * 10)) + "_y_" + str(
                                        #     int(y_grid_range[jj] * 10)) + "_z_" + str(int(z_grid_range[kk] * 10)),
                                        #                                direct=False)
                                        # Poscar(surf_film_struc2).write_file(ext_Dir2 + "/POSCAR_film_" + str(k)
                                        #                                + "_x_" + str(
                                        #     int(x_grid_range[ii] * 10)) + "_y_" + str(
                                        #     int(y_grid_range[jj] * 10)) + "_z_" + str(int(z_grid_range[kk] * 10)),
                                        #                                direct=False)

                                        '''
                                        z_len = max(surf_struc.cart_coords[:, 2]) - min(surf_struc.cart_coords[:, 2])
                                        z_len += 2 * max_rad
                                        reduced_int2 = new_int.get_reduced_structure()
    
                                        reduced_int2.make_supercell([1, 1, 1])
                                        OL_Input = PBC_coord_gen(reduced_int, ext_Dir2, k, int(x_grid_range[ii] * 10),
                                                                 int(y_grid_range[jj] * 10), int(z_grid_range[kk] * 10),
                                                                 file_gen=False)
    
                                        OL_Input22 = PBC_coord_gen(reduced_int2, ext_Dir2, k, int(x_grid_range[ii] * 10),
                                                                 int(y_grid_range[jj] * 10), int(z_grid_range[kk] * 10),
                                                                 file_gen=False)
    
                                        OL_Input_sub = PBC_coord_gen(surf_sub_struc2, " ", 1, 1, 1, 1, file_gen=False)
                                        OL_Input_film = PBC_coord_gen(surf_film_struc2, " ", 1, 1, 1, 1, file_gen=False)
    
                                        CC_dic = OL_Input[2]
                                        vec1_len = np.linalg.norm(surf_struc.lattice.matrix[0]) + 2 * max_rad
                                        vec2_len = np.linalg.norm(surf_struc.lattice.matrix[1]) + 2 * max_rad
                                        vec_ang = angle(surf_struc.lattice.matrix[0], surf_struc.lattice.matrix[1])
                                        surf_cube_vol = np.sin(vec_ang) * vec1_len * vec2_len * z_len
                                        surf_tot_vol = 0
                                        for i_sp in surf_struc.species:
                                            surf_tot_vol += sphe_vol(rad_dic[str(i_sp)])
                                        int_OL_vol = Mont_geo(OL_Input[0], OL_Input[1])- Mont_geo(OL_Input_sub[0], OL_Input_sub[1])
                                        - Mont_geo(OL_Input_film[0], OL_Input_film[1])
                                        un_occ_vol = surf_cube_vol - (surf_tot_vol - int_OL_vol)
    
                                        Score_array[ii, jj, kk] = (0.5 * int_OL_vol + 0.5 * un_occ_vol)
    
    
                                        '''

                                        # OL_calc = Contour_overlap(surf_sub_struc2.get_reduced_structure(), surf_film_struc2.get_reduced_structure())
                                        # Cont_OL = OL_calc[0]
                                        Cont_OL = 1
                                        # Cell_vol = OL_calc[1]
                                        Cell_vol = 1000
                                        # Cont_OL = 1
                                        # OL_array[ii, jj, kk] = int_OL_vol
                                        OL_array[ii, jj, kk] = Cont_OL

                                    Poscar(reduced_int).write_file(ext_Dir2 + "/POSCAR_Iface_" + str(k)
                                                                   + "_x_" + str(int(x_grid_range[ii] * 10)) + "_y_" + str(
                                        int(y_grid_range[jj] * 10)) + "_z_" + str(int(z_grid_range[kk] * 10)), direct=False)

                                    # Poscar(reduced_int.get_primitive_structure()).write_file(ext_Dir2 + "/POSCAR_Iface_pri_" + str(k)
                                    #                                + "_x_" + str(
                                    #     int(x_grid_range[ii] * 10)) + "_y_" + str(
                                    #     int(y_grid_range[jj] * 10)) + "_z_" + str(int(z_grid_range[kk] * 10)),
                                    #                                direct=False)
                    if scoring:
                        for zz in range(len(z_grid_range)):
                            xi, yi = np.meshgrid(x_grid_range, y_grid_range)
                            # plt.figure()
                            # contour2 = plt.contourf(xi, yi, Score_array[:, :, zz], 200, cmap='jet')
                            # plt.colorbar(contour2)
                            # plt.title("Score 2D- Contour " + " Z: " + str(z_grid_range[zz]))
                            # plt.xlabel("X displacement")
                            # plt.ylabel("Y displacement")
                            #
                            plt.figure()
                            contour3 = plt.contourf(xi, yi, OL_array[:, :, zz], 200, cmap='jet')
                            plt.colorbar(contour3)
                            plt.title("OL 2D- Contour " + " Z: " + str(z_grid_range[zz]))
                            plt.xlabel("X displacement")
                            plt.ylabel("Y displacement")

                            plt.savefig(working_dir +"OL_Contour_C:"+str(rad_dic['C'])+ "_H:" + str(rad_dic['H'])+".png" )
                            plt.show()
                    # print(np.argwhere(Score_array == np.amin(Score_array)), "   ",
                    #       np.argwhere(OL_array == np.amin(OL_array)))
                    # print(OL_array[2,3,0])

    os.remove(working_dir + "matrices_sets")


if __name__ == "__main__":
    main()
