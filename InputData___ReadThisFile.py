#Just Change this TWO lines
# from zData1 import input_properties_case_AssaelEx8_1Pg230
# from zData2 import input_properties_case_AssaelEx8_2Pg231
# from zData3 import input_properties_case_ElliottEx15_8Pg598
# from zData4 import input_properties_case_WalasEx6_6Pg326
# from zData5 import input_properties_case_SandlerPg500
# from zData6 import input_properties_case_whitson_problem_18_PR
# from zData7 import input_properties_case_noel_nevers_exF5_pg344
# from zData8 import input_properties_case_artigo_moises_2013___POE_ISO_5, comp1, \
#     comp2, kij, experimental_data_case_artigo_moises_2013___AB_ISO_5
from zData9 import input_properties_case_artigo_moises_2013___POE_ISO_7, comp1, comp2, kij, \
     experimental_data_case_artigo_moises_2013___POE_ISO_7
# from zData10 import input_properties_case_REFPROP___ButaneOctane, comp1, comp2, kij, refprop_data


'''GET INPUT PROPERTIES'''
props = input_properties_case_artigo_moises_2013___POE_ISO_7(comp1, comp2, kij)
exp_data = experimental_data_case_artigo_moises_2013___POE_ISO_7()


