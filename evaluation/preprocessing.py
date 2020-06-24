import os
import SimpleITK as sitk
import argparse
import pandas as pd
import numpy as np

train_ids = ['HGG/Brats17_2013_11_1/Brats17_2013_11_1',
             'HGG/Brats17_2013_13_1/Brats17_2013_13_1',
             'HGG/Brats17_2013_17_1/Brats17_2013_17_1',
             'HGG/Brats17_2013_19_1/Brats17_2013_19_1',
             'HGG/Brats17_2013_21_1/Brats17_2013_21_1',
             'HGG/Brats17_2013_23_1/Brats17_2013_23_1',
             'HGG/Brats17_2013_26_1/Brats17_2013_26_1',
             'HGG/Brats17_2013_2_1/Brats17_2013_2_1',
             'HGG/Brats17_2013_4_1/Brats17_2013_4_1',
             'HGG/Brats17_2013_7_1/Brats17_2013_7_1',
             'HGG/Brats17_CBICA_AAG_1/Brats17_CBICA_AAG_1',
             'HGG/Brats17_CBICA_AAL_1/Brats17_CBICA_AAL_1',
             'HGG/Brats17_CBICA_AAP_1/Brats17_CBICA_AAP_1',
             'HGG/Brats17_CBICA_ABE_1/Brats17_CBICA_ABE_1',
             'HGG/Brats17_CBICA_ABN_1/Brats17_CBICA_ABN_1',
             'HGG/Brats17_CBICA_ABY_1/Brats17_CBICA_ABY_1',
             'HGG/Brats17_CBICA_ALU_1/Brats17_CBICA_ALU_1',
             'HGG/Brats17_CBICA_AME_1/Brats17_CBICA_AME_1',
             'HGG/Brats17_CBICA_ANG_1/Brats17_CBICA_ANG_1',
             'HGG/Brats17_CBICA_ANP_1/Brats17_CBICA_ANP_1',
             'HGG/Brats17_CBICA_AOD_1/Brats17_CBICA_AOD_1',
             'HGG/Brats17_CBICA_AOO_1/Brats17_CBICA_AOO_1',
             'HGG/Brats17_CBICA_AOZ_1/Brats17_CBICA_AOZ_1',
             'HGG/Brats17_CBICA_APY_1/Brats17_CBICA_APY_1',
             'HGG/Brats17_CBICA_AQA_1/Brats17_CBICA_AQA_1',
             'HGG/Brats17_CBICA_AQG_1/Brats17_CBICA_AQG_1',
             'HGG/Brats17_CBICA_AQN_1/Brats17_CBICA_AQN_1',
             'HGG/Brats17_CBICA_AQP_1/Brats17_CBICA_AQP_1',
             'HGG/Brats17_CBICA_AQR_1/Brats17_CBICA_AQR_1',
             'HGG/Brats17_CBICA_AQU_1/Brats17_CBICA_AQU_1',
             'HGG/Brats17_CBICA_AQY_1/Brats17_CBICA_AQY_1',
             'HGG/Brats17_CBICA_ARF_1/Brats17_CBICA_ARF_1',
             'HGG/Brats17_CBICA_ARZ_1/Brats17_CBICA_ARZ_1',
             'HGG/Brats17_CBICA_ASA_1/Brats17_CBICA_ASA_1',
             'HGG/Brats17_CBICA_ASE_1/Brats17_CBICA_ASE_1',
             'HGG/Brats17_CBICA_ASH_1/Brats17_CBICA_ASH_1',
             'HGG/Brats17_CBICA_ASN_1/Brats17_CBICA_ASN_1',
             'HGG/Brats17_CBICA_ASU_1/Brats17_CBICA_ASU_1',
             'HGG/Brats17_CBICA_ASV_1/Brats17_CBICA_ASV_1',
             'HGG/Brats17_CBICA_ASW_1/Brats17_CBICA_ASW_1',
             'HGG/Brats17_CBICA_ASY_1/Brats17_CBICA_ASY_1',
             'HGG/Brats17_CBICA_ATB_1/Brats17_CBICA_ATB_1',
             'HGG/Brats17_CBICA_ATF_1/Brats17_CBICA_ATF_1',
             'HGG/Brats17_CBICA_ATV_1/Brats17_CBICA_ATV_1',
             'HGG/Brats17_CBICA_ATX_1/Brats17_CBICA_ATX_1',
             'HGG/Brats17_CBICA_AUN_1/Brats17_CBICA_AUN_1',
             'HGG/Brats17_CBICA_AUQ_1/Brats17_CBICA_AUQ_1',
             'HGG/Brats17_CBICA_AUR_1/Brats17_CBICA_AUR_1',
             'HGG/Brats17_CBICA_AVJ_1/Brats17_CBICA_AVJ_1',
             'HGG/Brats17_CBICA_AWG_1/Brats17_CBICA_AWG_1',
             'HGG/Brats17_CBICA_AWI_1/Brats17_CBICA_AWI_1',
             'HGG/Brats17_CBICA_AXJ_1/Brats17_CBICA_AXJ_1',
             'HGG/Brats17_CBICA_AXL_1/Brats17_CBICA_AXL_1',
             'HGG/Brats17_CBICA_AXN_1/Brats17_CBICA_AXN_1',
             'HGG/Brats17_CBICA_AXQ_1/Brats17_CBICA_AXQ_1',
             'HGG/Brats17_CBICA_AYA_1/Brats17_CBICA_AYA_1',
             'HGG/Brats17_CBICA_AYI_1/Brats17_CBICA_AYI_1',
             'HGG/Brats17_CBICA_AYU_1/Brats17_CBICA_AYU_1',
             'HGG/Brats17_CBICA_AZD_1/Brats17_CBICA_AZD_1',
             'HGG/Brats17_CBICA_AZH_1/Brats17_CBICA_AZH_1',
             'HGG/Brats17_CBICA_BFB_1/Brats17_CBICA_BFB_1',
             'HGG/Brats17_CBICA_BHB_1/Brats17_CBICA_BHB_1',
             'HGG/Brats17_CBICA_BHM_1/Brats17_CBICA_BHM_1',
             'HGG/Brats17_TCIA_111_1/Brats17_TCIA_111_1',
             'HGG/Brats17_TCIA_117_1/Brats17_TCIA_117_1',
             'HGG/Brats17_TCIA_121_1/Brats17_TCIA_121_1',
             'HGG/Brats17_TCIA_133_1/Brats17_TCIA_133_1',
             'HGG/Brats17_TCIA_138_1/Brats17_TCIA_138_1',
             'HGG/Brats17_TCIA_149_1/Brats17_TCIA_149_1',
             'HGG/Brats17_TCIA_151_1/Brats17_TCIA_151_1',
             'HGG/Brats17_TCIA_165_1/Brats17_TCIA_165_1',
             'HGG/Brats17_TCIA_167_1/Brats17_TCIA_167_1',
             'HGG/Brats17_TCIA_168_1/Brats17_TCIA_168_1',
             'HGG/Brats17_TCIA_179_1/Brats17_TCIA_179_1',
             'HGG/Brats17_TCIA_184_1/Brats17_TCIA_184_1',
             'HGG/Brats17_TCIA_186_1/Brats17_TCIA_186_1',
             'HGG/Brats17_TCIA_190_1/Brats17_TCIA_190_1',
             'HGG/Brats17_TCIA_192_1/Brats17_TCIA_192_1',
             'HGG/Brats17_TCIA_198_1/Brats17_TCIA_198_1',
             'HGG/Brats17_TCIA_201_1/Brats17_TCIA_201_1',
             'HGG/Brats17_TCIA_205_1/Brats17_TCIA_205_1',
             'HGG/Brats17_TCIA_211_1/Brats17_TCIA_211_1',
             'HGG/Brats17_TCIA_218_1/Brats17_TCIA_218_1',
             'HGG/Brats17_TCIA_221_1/Brats17_TCIA_221_1',
             'HGG/Brats17_TCIA_226_1/Brats17_TCIA_226_1',
             'HGG/Brats17_TCIA_234_1/Brats17_TCIA_234_1',
             'HGG/Brats17_TCIA_242_1/Brats17_TCIA_242_1',
             'HGG/Brats17_TCIA_247_1/Brats17_TCIA_247_1',
             'HGG/Brats17_TCIA_257_1/Brats17_TCIA_257_1',
             'HGG/Brats17_TCIA_274_1/Brats17_TCIA_274_1',
             'HGG/Brats17_TCIA_277_1/Brats17_TCIA_277_1',
             'HGG/Brats17_TCIA_278_1/Brats17_TCIA_278_1',
             'HGG/Brats17_TCIA_280_1/Brats17_TCIA_280_1',
             'HGG/Brats17_TCIA_283_1/Brats17_TCIA_283_1',
             'HGG/Brats17_TCIA_296_1/Brats17_TCIA_296_1',
             'HGG/Brats17_TCIA_309_1/Brats17_TCIA_309_1',
             'HGG/Brats17_TCIA_314_1/Brats17_TCIA_314_1',
             'HGG/Brats17_TCIA_319_1/Brats17_TCIA_319_1',
             'HGG/Brats17_TCIA_321_1/Brats17_TCIA_321_1',
             'HGG/Brats17_TCIA_322_1/Brats17_TCIA_322_1',
             'HGG/Brats17_TCIA_331_1/Brats17_TCIA_331_1',
             'HGG/Brats17_TCIA_335_1/Brats17_TCIA_335_1',
             'HGG/Brats17_TCIA_343_1/Brats17_TCIA_343_1',
             'HGG/Brats17_TCIA_368_1/Brats17_TCIA_368_1',
             'HGG/Brats17_TCIA_372_1/Brats17_TCIA_372_1',
             'HGG/Brats17_TCIA_375_1/Brats17_TCIA_375_1',
             'HGG/Brats17_TCIA_378_1/Brats17_TCIA_378_1',
             'HGG/Brats17_TCIA_394_1/Brats17_TCIA_394_1',
             'HGG/Brats17_TCIA_401_1/Brats17_TCIA_401_1',
             'HGG/Brats17_TCIA_406_1/Brats17_TCIA_406_1',
             'HGG/Brats17_TCIA_409_1/Brats17_TCIA_409_1',
             'HGG/Brats17_TCIA_412_1/Brats17_TCIA_412_1',
             'HGG/Brats17_TCIA_425_1/Brats17_TCIA_425_1',
             'HGG/Brats17_TCIA_430_1/Brats17_TCIA_430_1',
             'HGG/Brats17_TCIA_437_1/Brats17_TCIA_437_1',
             'HGG/Brats17_TCIA_448_1/Brats17_TCIA_448_1',
             'HGG/Brats17_TCIA_455_1/Brats17_TCIA_455_1',
             'HGG/Brats17_TCIA_460_1/Brats17_TCIA_460_1',
             'HGG/Brats17_TCIA_469_1/Brats17_TCIA_469_1',
             'HGG/Brats17_TCIA_471_1/Brats17_TCIA_471_1',
             'HGG/Brats17_TCIA_474_1/Brats17_TCIA_474_1',
             'HGG/Brats17_TCIA_479_1/Brats17_TCIA_479_1',
             'HGG/Brats17_TCIA_498_1/Brats17_TCIA_498_1',
             'HGG/Brats17_TCIA_499_1/Brats17_TCIA_499_1',
             'HGG/Brats17_TCIA_603_1/Brats17_TCIA_603_1',
             'HGG/Brats17_TCIA_606_1/Brats17_TCIA_606_1',
             'HGG/Brats17_TCIA_608_1/Brats17_TCIA_608_1',
             'LGG/Brats17_2013_15_1/Brats17_2013_15_1',
             'LGG/Brats17_2013_1_1/Brats17_2013_1_1',
             'LGG/Brats17_2013_28_1/Brats17_2013_28_1',
             'LGG/Brats17_2013_6_1/Brats17_2013_6_1',
             'LGG/Brats17_2013_9_1/Brats17_2013_9_1',
             'LGG/Brats17_TCIA_103_1/Brats17_TCIA_103_1',
             'LGG/Brats17_TCIA_130_1/Brats17_TCIA_130_1',
             'LGG/Brats17_TCIA_152_1/Brats17_TCIA_152_1',
             'LGG/Brats17_TCIA_177_1/Brats17_TCIA_177_1',
             'LGG/Brats17_TCIA_241_1/Brats17_TCIA_241_1',
             'LGG/Brats17_TCIA_249_1/Brats17_TCIA_249_1',
             'LGG/Brats17_TCIA_254_1/Brats17_TCIA_254_1',
             'LGG/Brats17_TCIA_261_1/Brats17_TCIA_261_1',
             'LGG/Brats17_TCIA_276_1/Brats17_TCIA_276_1',
             'LGG/Brats17_TCIA_282_1/Brats17_TCIA_282_1',
             'LGG/Brats17_TCIA_298_1/Brats17_TCIA_298_1',
             'LGG/Brats17_TCIA_307_1/Brats17_TCIA_307_1',
             'LGG/Brats17_TCIA_312_1/Brats17_TCIA_312_1',
             'LGG/Brats17_TCIA_325_1/Brats17_TCIA_325_1',
             'LGG/Brats17_TCIA_330_1/Brats17_TCIA_330_1',
             'LGG/Brats17_TCIA_351_1/Brats17_TCIA_351_1',
             'LGG/Brats17_TCIA_393_1/Brats17_TCIA_393_1',
             'LGG/Brats17_TCIA_408_1/Brats17_TCIA_408_1',
             'LGG/Brats17_TCIA_413_1/Brats17_TCIA_413_1',
             'LGG/Brats17_TCIA_428_1/Brats17_TCIA_428_1',
             'LGG/Brats17_TCIA_449_1/Brats17_TCIA_449_1',
             'LGG/Brats17_TCIA_462_1/Brats17_TCIA_462_1',
             'LGG/Brats17_TCIA_470_1/Brats17_TCIA_470_1',
             'LGG/Brats17_TCIA_480_1/Brats17_TCIA_480_1',
             'LGG/Brats17_TCIA_490_1/Brats17_TCIA_490_1',
             'LGG/Brats17_TCIA_615_1/Brats17_TCIA_615_1',
             'LGG/Brats17_TCIA_620_1/Brats17_TCIA_620_1',
             'LGG/Brats17_TCIA_621_1/Brats17_TCIA_621_1',
             'LGG/Brats17_TCIA_623_1/Brats17_TCIA_623_1',
             'LGG/Brats17_TCIA_624_1/Brats17_TCIA_624_1',
             'LGG/Brats17_TCIA_625_1/Brats17_TCIA_625_1',
             'LGG/Brats17_TCIA_629_1/Brats17_TCIA_629_1',
             'LGG/Brats17_TCIA_632_1/Brats17_TCIA_632_1',
             'LGG/Brats17_TCIA_634_1/Brats17_TCIA_634_1',
             'LGG/Brats17_TCIA_639_1/Brats17_TCIA_639_1',
             'LGG/Brats17_TCIA_642_1/Brats17_TCIA_642_1',
             'LGG/Brats17_TCIA_645_1/Brats17_TCIA_645_1',
             'LGG/Brats17_TCIA_653_1/Brats17_TCIA_653_1',
             'LGG/Brats17_TCIA_654_1/Brats17_TCIA_654_1']

valid_ids = ['HGG/Brats17_2013_22_1/Brats17_2013_22_1',
             'HGG/Brats17_2013_25_1/Brats17_2013_25_1',
             'HGG/Brats17_CBICA_ABM_1/Brats17_CBICA_ABM_1',
             'HGG/Brats17_CBICA_ALX_1/Brats17_CBICA_ALX_1',
             'HGG/Brats17_CBICA_AMH_1/Brats17_CBICA_AMH_1',
             'HGG/Brats17_CBICA_AQV_1/Brats17_CBICA_AQV_1',
             'HGG/Brats17_CBICA_ARW_1/Brats17_CBICA_ARW_1',
             'HGG/Brats17_CBICA_ASK_1/Brats17_CBICA_ASK_1',
             'HGG/Brats17_CBICA_ATP_1/Brats17_CBICA_ATP_1',
             'HGG/Brats17_CBICA_AVG_1/Brats17_CBICA_AVG_1',
             'HGG/Brats17_CBICA_AXW_1/Brats17_CBICA_AXW_1',
             'HGG/Brats17_CBICA_BFP_1/Brats17_CBICA_BFP_1',
             'HGG/Brats17_TCIA_150_1/Brats17_TCIA_150_1',
             'HGG/Brats17_TCIA_199_1/Brats17_TCIA_199_1',
             'HGG/Brats17_TCIA_203_1/Brats17_TCIA_203_1',
             'HGG/Brats17_TCIA_222_1/Brats17_TCIA_222_1',
             'HGG/Brats17_TCIA_231_1/Brats17_TCIA_231_1',
             'HGG/Brats17_TCIA_300_1/Brats17_TCIA_300_1',
             'HGG/Brats17_TCIA_390_1/Brats17_TCIA_390_1',
             'HGG/Brats17_TCIA_436_1/Brats17_TCIA_436_1',
             'HGG/Brats17_TCIA_444_1/Brats17_TCIA_444_1',
             'HGG/Brats17_TCIA_607_1/Brats17_TCIA_607_1',
             'LGG/Brats17_2013_24_1/Brats17_2013_24_1',
             'LGG/Brats17_TCIA_202_1/Brats17_TCIA_202_1',
             'LGG/Brats17_TCIA_346_1/Brats17_TCIA_346_1',
             'LGG/Brats17_TCIA_628_1/Brats17_TCIA_628_1',
             'LGG/Brats17_TCIA_637_1/Brats17_TCIA_637_1',
             'LGG/Brats17_TCIA_640_1/Brats17_TCIA_640_1']

test_ids = ['HGG/Brats17_2013_10_1/Brats17_2013_10_1',
            'HGG/Brats17_2013_12_1/Brats17_2013_12_1',
            'HGG/Brats17_2013_14_1/Brats17_2013_14_1',
            'HGG/Brats17_2013_18_1/Brats17_2013_18_1',
            'HGG/Brats17_2013_20_1/Brats17_2013_20_1',
            'HGG/Brats17_2013_27_1/Brats17_2013_27_1',
            'HGG/Brats17_2013_3_1/Brats17_2013_3_1',
            'HGG/Brats17_2013_5_1/Brats17_2013_5_1',
            'HGG/Brats17_CBICA_AAB_1/Brats17_CBICA_AAB_1',
            'HGG/Brats17_CBICA_ABB_1/Brats17_CBICA_ABB_1',
            'HGG/Brats17_CBICA_ABO_1/Brats17_CBICA_ABO_1',
            'HGG/Brats17_CBICA_ALN_1/Brats17_CBICA_ALN_1',
            'HGG/Brats17_CBICA_ANI_1/Brats17_CBICA_ANI_1',
            'HGG/Brats17_CBICA_ANZ_1/Brats17_CBICA_ANZ_1',
            'HGG/Brats17_CBICA_AOH_1/Brats17_CBICA_AOH_1',
            'HGG/Brats17_CBICA_AOP_1/Brats17_CBICA_AOP_1',
            'HGG/Brats17_CBICA_APR_1/Brats17_CBICA_APR_1',
            'HGG/Brats17_CBICA_APZ_1/Brats17_CBICA_APZ_1',
            'HGG/Brats17_CBICA_AQD_1/Brats17_CBICA_AQD_1',
            'HGG/Brats17_CBICA_AQJ_1/Brats17_CBICA_AQJ_1',
            'HGG/Brats17_CBICA_AQO_1/Brats17_CBICA_AQO_1',
            'HGG/Brats17_CBICA_AQQ_1/Brats17_CBICA_AQQ_1',
            'HGG/Brats17_CBICA_AQT_1/Brats17_CBICA_AQT_1',
            'HGG/Brats17_CBICA_AQZ_1/Brats17_CBICA_AQZ_1',
            'HGG/Brats17_CBICA_ASG_1/Brats17_CBICA_ASG_1',
            'HGG/Brats17_CBICA_ASO_1/Brats17_CBICA_ASO_1',
            'HGG/Brats17_CBICA_ATD_1/Brats17_CBICA_ATD_1',
            'HGG/Brats17_CBICA_AVV_1/Brats17_CBICA_AVV_1',
            'HGG/Brats17_CBICA_AWH_1/Brats17_CBICA_AWH_1',
            'HGG/Brats17_CBICA_AXM_1/Brats17_CBICA_AXM_1',
            'HGG/Brats17_CBICA_AXO_1/Brats17_CBICA_AXO_1',
            'HGG/Brats17_CBICA_AYW_1/Brats17_CBICA_AYW_1',
            'HGG/Brats17_CBICA_BHK_1/Brats17_CBICA_BHK_1',
            'HGG/Brats17_TCIA_105_1/Brats17_TCIA_105_1',
            'HGG/Brats17_TCIA_113_1/Brats17_TCIA_113_1',
            'HGG/Brats17_TCIA_118_1/Brats17_TCIA_118_1',
            'HGG/Brats17_TCIA_131_1/Brats17_TCIA_131_1',
            'HGG/Brats17_TCIA_135_1/Brats17_TCIA_135_1',
            'HGG/Brats17_TCIA_147_1/Brats17_TCIA_147_1',
            'HGG/Brats17_TCIA_162_1/Brats17_TCIA_162_1',
            'HGG/Brats17_TCIA_171_1/Brats17_TCIA_171_1',
            'HGG/Brats17_TCIA_180_1/Brats17_TCIA_180_1',
            'HGG/Brats17_TCIA_208_1/Brats17_TCIA_208_1',
            'HGG/Brats17_TCIA_235_1/Brats17_TCIA_235_1',
            'HGG/Brats17_TCIA_265_1/Brats17_TCIA_265_1',
            'HGG/Brats17_TCIA_290_1/Brats17_TCIA_290_1',
            'HGG/Brats17_TCIA_328_1/Brats17_TCIA_328_1',
            'HGG/Brats17_TCIA_332_1/Brats17_TCIA_332_1',
            'HGG/Brats17_TCIA_338_1/Brats17_TCIA_338_1',
            'HGG/Brats17_TCIA_361_1/Brats17_TCIA_361_1',
            'HGG/Brats17_TCIA_370_1/Brats17_TCIA_370_1',
            'HGG/Brats17_TCIA_374_1/Brats17_TCIA_374_1',
            'HGG/Brats17_TCIA_377_1/Brats17_TCIA_377_1',
            'HGG/Brats17_TCIA_396_1/Brats17_TCIA_396_1',
            'HGG/Brats17_TCIA_411_1/Brats17_TCIA_411_1',
            'HGG/Brats17_TCIA_419_1/Brats17_TCIA_419_1',
            'HGG/Brats17_TCIA_429_1/Brats17_TCIA_429_1',
            'HGG/Brats17_TCIA_473_1/Brats17_TCIA_473_1',
            'HGG/Brats17_TCIA_478_1/Brats17_TCIA_478_1',
            'HGG/Brats17_TCIA_491_1/Brats17_TCIA_491_1',
            'HGG/Brats17_TCIA_605_1/Brats17_TCIA_605_1',
            'LGG/Brats17_2013_0_1/Brats17_2013_0_1',
            'LGG/Brats17_2013_16_1/Brats17_2013_16_1',
            'LGG/Brats17_2013_29_1/Brats17_2013_29_1',
            'LGG/Brats17_2013_8_1/Brats17_2013_8_1',
            'LGG/Brats17_TCIA_101_1/Brats17_TCIA_101_1',
            'LGG/Brats17_TCIA_109_1/Brats17_TCIA_109_1',
            'LGG/Brats17_TCIA_141_1/Brats17_TCIA_141_1',
            'LGG/Brats17_TCIA_175_1/Brats17_TCIA_175_1',
            'LGG/Brats17_TCIA_255_1/Brats17_TCIA_255_1',
            'LGG/Brats17_TCIA_266_1/Brats17_TCIA_266_1',
            'LGG/Brats17_TCIA_299_1/Brats17_TCIA_299_1',
            'LGG/Brats17_TCIA_310_1/Brats17_TCIA_310_1',
            'LGG/Brats17_TCIA_387_1/Brats17_TCIA_387_1',
            'LGG/Brats17_TCIA_402_1/Brats17_TCIA_402_1',
            'LGG/Brats17_TCIA_410_1/Brats17_TCIA_410_1',
            'LGG/Brats17_TCIA_420_1/Brats17_TCIA_420_1',
            'LGG/Brats17_TCIA_442_1/Brats17_TCIA_442_1',
            'LGG/Brats17_TCIA_451_1/Brats17_TCIA_451_1',
            'LGG/Brats17_TCIA_466_1/Brats17_TCIA_466_1',
            'LGG/Brats17_TCIA_493_1/Brats17_TCIA_493_1',
            'LGG/Brats17_TCIA_618_1/Brats17_TCIA_618_1',
            'LGG/Brats17_TCIA_630_1/Brats17_TCIA_630_1',
            'LGG/Brats17_TCIA_633_1/Brats17_TCIA_633_1',
            'LGG/Brats17_TCIA_644_1/Brats17_TCIA_644_1',
            'LGG/Brats17_TCIA_650_1/Brats17_TCIA_650_1']


def get_brain_mask(t1):
    brain_mask = sitk.GetImageFromArray((sitk.GetArrayFromImage(t1) > 0).astype(np.uint8))
    brain_mask.CopyInformation(t1)
    brain_mask = sitk.Cast(brain_mask, sitk.sitkUInt8)
    return brain_mask


def z_score_normalisation(channel, brain_mask, cutoff_percentiles=(5., 95.), cutoff_below_mean=True):
    low, high = np.percentile(channel[brain_mask.astype(np.bool)], cutoff_percentiles)
    norm_mask = np.logical_and(brain_mask, np.logical_and(channel > low, channel < high))
    if cutoff_below_mean:
        norm_mask = np.logical_and(norm_mask, channel > np.mean(channel))
    masked_channel = channel[norm_mask]
    normalised_channel = (channel - np.mean(masked_channel)) / np.std(masked_channel)
    return normalised_channel


def fix_segmentation_labels(seg):
    array = sitk.GetArrayFromImage(seg)
    array[array == 4] = 3
    new_seg = sitk.GetImageFromArray(array)
    new_seg.CopyInformation(seg)
    return new_seg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',
                        required=True,
                        type=str,
                        help='Path to input directory.')
    parser.add_argument('--output-dir',
                        required=True,
                        type=str,
                        help='Path to output directory.')

    parse_args, unknown = parser.parse_known_args()
    output_dataframe = pd.DataFrame()
    for subdir_0 in ['HGG', 'LGG']:
        for subdir_1 in os.listdir(os.path.join(parse_args.input_dir, subdir_0)):
            id_ = os.path.join(subdir_0, subdir_1) + '/' + subdir_1
            print(id_)
            seg = fix_segmentation_labels(sitk.ReadImage(os.path.join(parse_args.input_dir, id_) + '_seg.nii.gz'))
            output_path = os.path.join(parse_args.output_dir, id_) + f'_seg.nii.gz'
            output_dataframe.loc[id_, 'seg'] = output_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(seg, output_path)

            t1 = sitk.ReadImage(os.path.join(parse_args.input_dir, id_) + '_t1.nii.gz')
            brain_mask = get_brain_mask(t1)
            output_path = os.path.join(parse_args.output_dir, id_) + f'_brain_mask.nii.gz'
            output_dataframe.loc[id_, 'sampling_mask'] = output_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(brain_mask, output_path)

            for suffix in ['flair', 't1', 't1ce', 't2']:
                channel = sitk.ReadImage(os.path.join(parse_args.input_dir, id_) + f'_{suffix:s}.nii.gz')
                channel_array = sitk.GetArrayFromImage(channel)
                normalised_channel_array = z_score_normalisation(channel_array, sitk.GetArrayFromImage(brain_mask))
                normalised_channel = sitk.GetImageFromArray(normalised_channel_array)
                normalised_channel.CopyInformation(channel)
                output_path = os.path.join(parse_args.output_dir, id_) + f'_{suffix:s}.nii.gz'
                output_dataframe.loc[id_, suffix] = output_path
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                sitk.WriteImage(normalised_channel, output_path)
    output_dataframe.index.name = 'id'
    os.makedirs('assets/BraTS2017_data', exist_ok=True)
    train_index = output_dataframe.loc[train_ids]
    train_index.to_csv('assets/BraTS2017_data/data_index_train.csv')
    valid_index = output_dataframe.loc[valid_ids]
    valid_index.to_csv('assets/BraTS2017_data/data_index_valid.csv')
    test_index = output_dataframe.loc[test_ids]
    test_index.to_csv('assets/BraTS2017_data/data_index_test.csv')
