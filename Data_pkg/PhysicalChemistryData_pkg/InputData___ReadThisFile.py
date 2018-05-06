#Just Change this TWO lines
# from Data_pkg.PhysicalChemistryData_pkg.zData1 import input_properties_case_AssaelEx8_1Pg230
# from Data_pkg.PhysicalChemistryData_pkg.zData2 import input_properties_case_AssaelEx8_2Pg231
# from Data_pkg.PhysicalChemistryData_pkg.zData3 import input_properties_case_ElliottEx15_8Pg598
# from Data_pkg.PhysicalChemistryData_pkg.zData4 import input_properties_case_WalasEx6_6Pg326
# from Data_pkg.PhysicalChemistryData_pkg.zData5 import input_properties_case_SandlerPg500
# from Data_pkg.PhysicalChemistryData_pkg.zData6 import input_properties_case_whitson_problem_18_PR
# from Data_pkg.PhysicalChemistryData_pkg.zData7 import input_properties_case_noel_nevers_exF5_pg344
# from Data_pkg.PhysicalChemistryData_pkg.zData8 import input_properties_case_artigo_moises_2013___POE_ISO_5, comp1, \
#     comp2, kij, experimental_data_case_artigo_moises_2013___AB_ISO_5
# from Data_pkg.PhysicalChemistryData_pkg.zData9 import input_properties_case_artigo_moises_2013___POE_ISO_7, comp1, comp2, kij, \
                #    experimental_data_case_artigo_moises_2013___POE_ISO_7
# from Data_pkg.PhysicalChemistryData_pkg.zData10 import input_properties_case_REFPROP___ButaneOctane, comp1, comp2, kij, refprop_data


from Data_pkg.PhysicalChemistryData_pkg.zData11 import input_properties_case_R134a___POE_ISO_10, comp1, comp2, kij
'''GET INPUT PROPERTIES'''
props = input_properties_case_R134a___POE_ISO_10(comp1, comp2, kij)
# exp_data = refprop_data()
#exp_data = experimental_data_case_artigo_moises_2013___POE_ISO_7()


