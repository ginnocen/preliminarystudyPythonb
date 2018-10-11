#########################################################
#                                                       #
#   List of kinemtaic variables for different D mesons  #
#                                                       #
#########################################################

# common variables to all D mesons analyses
common_variables     = ['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML',]
additional_variables = ['inv_mass_ML','pt_cand_ML','pt_prong0_ML','pt_prong1_ML']

# specific variables of different D mesons
D0_adj_var    = ['cos_t_star_ML', 'imp_par_prong0_ML', 'imp_par_prong1_ML','imp_par_prod_ML']
Ds_adj_var    = ['sig_vert_ML','delta_mass_KK_ML','cos_PiDs_ML','cos_PiKPhi_3_ML']
Dplus_adj_var = ['sig_vert_ML']
additional_variables_2 = ['pt_prong2_ML']

#
# --- create final lists of variables
#

D0_var    = common_variables.copy()
Ds_var    = common_variables.copy() 
Dplus_var = common_variables.copy()

#D0_var        += additional_variables.copy()
#Ds_adj_var    += additional_variables.copy()
#Dplus_adj_var += additional_variables.copy()

D0_var    += D0_adj_var.copy()
Ds_var    += Ds_adj_var.copy()
Dplus_var += Dplus_adj_var.copy()

#D0_var        += additional_variables_2.copy()
#Ds_adj_var    += additional_variables_2.copy()
#Dplus_adj_var += additional_variables_2.copy()

# create a dictionary with the D meson variables
D_dictionary = {
                        'D0'    : D0_var,
                        'Ds'    : Ds_var,
                        'Dplus' : Dplus_var
               }




'''
Variabili comuni a tutte le candidate:
{"inv_mass","pt_cand","d_len","d_len_xy","norm_dl_xy","cos_p","cos_p_xy","imp_par_xy","pt_prong0","pt_prong1"};

Variabili aggiuntive per la D0:
{"cos_t_star", "imp_par_prong0", "imp_par_prong1","imp_par_prod"};

Variabili aggiuntive per la Ds: 
{"pt_prong2","sig_vert","delta_mass_KK","cos_PiDs","cos_PiKPhi_3"};

Variabili aggiuntive per la D+:
 {"pt_prong2","sig_vert"};

Variabili PID
   -   {"nsigTPC_Pi_0","nsigTPC_K_0","nsigTOF_Pi_0","nsigTOF_K_0","nsigTPC_Pi_1","nsigTPC_K_1","nsigTOF_Pi_1","nsigTOF_K_1","nsigTPC_Pi_2","nsigTPC_K_2","nsigTOF_Pi_2","nsigTOF_K_2"};
   -   {"nsigComb_Pi_0","nsigComb_K_0","nsigComb_Pi_1","nsigComb_K_1","nsigComb_Pi_2","nsigComb_K_2","","","","","","”}; 
        Questo secondo set rappresenta un’alternativa al primo set, in particolare la sigma TPC e la sigma TOF vengono combinate.
'''
 
