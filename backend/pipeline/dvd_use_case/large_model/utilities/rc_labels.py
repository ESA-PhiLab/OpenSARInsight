"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
rc_labels.py

Tool: Filtered Vessel Label Lists for RC and SLC Imagery [used in Filtering the dataset]

Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2026-04-08

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""

labels_visible_on_rc_and_align: list[str] = [
    'VD_9_VV', 'VD_9_VH', 'VD_11_VV', 'VD_11_VH', 'VD_14_VH', 'VD_14_VV', 'VD_15_VH', 'VD_15_VV', 'VD_45_VH', 'VD_45_VV',
    'VD_49_VH', 'VD_49_VV', 'VD_138_VV', 'VD_138_VH', 'VD_153_VV', 'VD_153_VH', 'VD_169_VV', 'VD_169_VH', 'VD_175_VV', 'VD_175_VH',
    'VD_176_VV', 'VD_176_VH', 'VD_177_VV', 'VD_177_VH', 'VD_181_VV', 'VD_181_VH', 'VD_201_VV', 'VD_201_VH', 'VD_209_VV', 'VD_209_VH',
    'VD_213_VV', 'VD_213_VH', 'VD_214_VV', 'VD_214_VH', 'VD_218_VV', 'VD_218_VH', 'VD_222_VV', 'VD_222_VH', 'VD_227_VV', 'VD_227_VH',
    'VD_231_VV', 'VD_231_VH', 'VD_232_VV', 'VD_232_VH', 'VD_235_VV', 'VD_235_VH', 'VD_236_VV', 'VD_236_VH', 'VD_237_VV', 'VD_237_VH',
    'VD_238_VV', 'VD_238_VH', 'VD_241_VV', 'VD_241_VH', 'VD_242_VV', 'VD_242_VH', 'VD_243_VV', 'VD_243_VH', 'VD_245_VV', 'VD_245_VH',
    'VD_248_VV', 'VD_248_VH', 'VD_277_VV', 'VD_277_VH', 'VD_285_VV', 'VD_285_VH', 'VD_292_VV', 'VD_292_VH', 'VD_300_VV', 'VD_300_VH',
    'VD_310_VV', 'VD_310_VH', 'VD_328_VV', 'VD_328_VH', 'VD_329_VV', 'VD_329_VH', 'VD_341_VV', 'VD_341_VH', 'VD_342_VV', 'VD_342_VH',
    'VD_344_VV', 'VD_344_VH', 'VD_356_VV', 'VD_356_VH', 'VD_367_VV', 'VD_367_VH', 'VD_376_VV', 'VD_376_VH', 'VD_388_VV', 'VD_383_VH',
    'VD_385_VV', 'VD_385_VH', 'VD_397_VV', 'VD_397_VH', 'VD_409_VV', 'VD_409_VH', 'VD_414_VV', 'VD_414_VH', 'VD_420_VV', 'VD_420_VH',
    'VD_435_VV', 'VD_435_VH', 'VD_441_VV', 'VD_441_VH', 'VD_442_VV', 'VD_442_VH', 'VD_449_VV', 'VD_449_VH', 'VD_450_VV', 'VD_450_VH',
    'VD_455_VV', 'VD_455_VH', 'VD_464_VV', 'VD_464_VH', 'VD_470_VV', 'VD_470_VH', 'VD_472_VV', 'VD_472_VH', 'VD_473_VV', 'VD_473_VH',
    'VD_477_VV', 'VD_477_VH', 'VD_481_VV', 'VD_481_VH', 'VD_482_VV', 'VD_482_VH', 'VD_486_VV', 'VD_486_VH', 'VD_487_VV', 'VD_487_VH',
    'VD_491_VV', 'VD_491_VH', 'VD_511_VV', 'VD_511_VH', 'VD_514_VV', 'VD_514_VH', 'VD_516_VV', 'VD_516_VH', 'VD_532_VV', 'VD_532_VH',
    'VD_565_VV', 'VD_565_VH', 'VD_570_VV', 'VD_570_VH', 'VD_606_VV', 'VD_606_VH', 'VD_613_VV', 'VD_613_VH', 'VD_661_VV', 'VD_661_VH',
    'VD_692_VV', 'VD_692_VH', 'VD_693_VV', 'VD_693_VH', 'VD_696_VV', 'VD_696_VH', 'VD_699_VV', 'VD_699_VH', 'VD_705_VV', 'VD_705_VH',
    'VD_720_VV', 'VD_720_VH', 'VD_723_VV', 'VD_723_VH', 'VD_729_VV', 'VD_729_VH', 'VD_733_VV', 'VD_733_VH', 'VD_759_VV', 'VD_759_VH',
    'VD_765_VV', 'VD_765_VH', 'VD_769_VV', 'VD_769_VH', 'VD_786_VV', 'VD_786_VH', 'VD_787_VV', 'VD_787_VH', 'VD_807_VV', 'VD_807_VH',
    'VD_810_VV', 'VD_810_VH', 'VD_811_VV', 'VD_811_VH', 'VD_814_VV', 'VD_814_VH', 'VD_815_VV', 'VD_815_VH', 'VD_817_VV', 'VD_817_VH',
    'VD_818_VV', 'VD_818_VH', 'VD_820_VV', 'VD_820_VH', 'VD_821_VV', 'VD_821_VH', 'VD_823_VV', 'VD_823_VH', 'VD_824_VV', 'VD_824_VH',
    'VD_827_VV', 'VD_827_VH', 'VD_832_VV', 'VD_832_VH', 'VD_834_VV', 'VD_834_VH', 'VD_843_VV', 'VD_843_VH', 'VD_845_VV', 'VD_845_VH',
    'VD_843_VV', 'VD_843_VH', 'VD_854_VV', 'VD_854_VH', 'VD_867_VV', 'VD_867_VH', 'VD_874_VV', 'VD_874_VH', 'VD_875_VV', 'VD_875_VH',
    'VD_893_VV', 'VD_893_VH', 'VD_900_VV', 'VD_900_VH', 'VD_938_VV', 'VD_938_VH', 'VD_948_VV', 'VD_948_VH', 'VD_953_VV', 'VD_953_VH',
    'VD_957_VV', 'VD_957_VH', 'VD_966_VV', 'VD_966_VH', 'VD_985_VV', 'VD_985_VH', 'VD_987_VV', 'VD_987_VH', 'VD_991_VV', 'VD_991_VH',
    'VD_1010_VV', 'VD_1010_VH', 'VD_1021_VV', 'VD_1021_VH','VD_1071_VV', 'VD_1071_VH', 'VD_1080_VV', 'VD_1080_VH', 'VD_1091_VV',
    'VD_1091_VH', 'VD_1101_VV', 'VD_1101_VH','VD_1106_VV', 'VD_1106_VH', 'VD_1113_VV', 'VD_1113_VH', 'VD_1119_VV', 'VD_1119_VH',
    'VD_1120_VV', 'VD_1120_VH', 'VD_1124_VV', 'VD_1124_VH', 'VD_1125_VV', 'VD_1125_VH', 'VD_1128_VV', 'VD_1128_VH', 'VD_1142_VV',
    'VD_1142_VH', 'VD_1145_VV', 'VD_1145_VH', 'VD_1146_VV', 'VD_1146_VH', 'VD_1149_VV', 'VD_1149_VH', 'VD_1156_VV', 'VD_1156_VH',
    'VD_1158_VV', 'VD_1158_VH', 'VD_1159_VV', 'VD_1159_VH', 'VD_1202_VV', 'VD_1202_VH', 'VD_1205_VV', 'VD_1205_VH', 'VD_1212_VV',
    'VD_1212_VH', 'VD_1213_VV', 'VD_1213_VH', 'VD_1221_VV', 'VD_1221_VH', 'VD_1249_VV', 'VD_1249_VH', 'VD_1257_VV', 'VD_1257_VH',
    'VD_1271_VV', 'VD_1271_VH', 'VD_1289_VV', 'VD_1289_VH', 'VD_1307_VV', 'VD_1307_VH', 'VD_1321_VV', 'VD_1321_VH', 'VD_1345_VV',
    'VD_1345_VH', 'VD_1380_VV', 'VD_1380_VH', 'VD_1388_VV', 'VD_1388_VH', 'VD_1434_VV', 'VD_1434_VH', 'VD_1474_VV', 'VD_1474_VH',
    'VD_1478_VV', 'VD_1478_VH', 'VD_1479_VV', 'VD_1479_VH', 'VD_1482_VV', 'VD_1482_VH', 'VD_1484_VV', 'VD_1484_VH', 'VD_1489_VV',
    'VD_1489_VH', 'VD_1490_VV', 'VD_1490_VH', 'VD_1493_VV', 'VD_1493_VH', 'VD_1495_VV', 'VD_1495_VH', 'VD_1496_VV', 'VD_1496_VH',
    'VD_1498_VV', 'VD_1498_VH', 'VD_1499_VV', 'VD_1499_VH', 'VD_1502_VV', 'VD_1502_VH', 'VD_1505_VV', 'VD_1505_VH', 'VD_3052_VV',
    'VD_3052_VH', 'VD_3057_VV', 'VD_3057_VH', 'VD_3058_VV', 'VD_3058_VH', 'VD_3059_VV', 'VD_3059_VH', 'VD_3060_VV', 'VD_3060_VH',
    'VD_3062_VV', 'VD_3062_VH', 'VD_3067_VV', 'VD_3067_VH', 'VD_3068_VV', 'VD_3068_VH', 'VD_3072_VV', 'VD_3072_VH', 'VD_3075_VV',
    'VD_3075_VH', 'VD_3076_VV', 'VD_3076_VH', 'VD_3093_VV', 'VD_3093_VH', 'VD_3103_VV', 'VD_3130_VH', 'VD_3107_VV', 'VD_3107_VH',
    'VD_3110_VV', 'VD_3110_VH', 'VD_3111_VV', 'VD_3111_VH', 'VD_3116_VV', 'VD_3116_VH', 'VD_3118_VV', 'VD_3118_VH', 'VD_3119_VV',
    'VD_3119_VH', 'VD_3123_VV', 'VD_3123_VH', 'VD_3128_VV', 'VD_3128_VH', 'VD_3130_VV', 'VD_3130_VH', 'VD_3133_VV', 'VD_3133_VH',
    'VD_3135_VV', 'VD_3135_VH', 'VD_3136_VV', 'VD_3136_VH', 'VD_3139_VV', 'VD_3139_VH', 'VD_3141_VV', 'VD_3141_VH', 'VD_3142_VV',
    'VD_3142_VH', 'VD_3144_VV', 'VD_3144_VH', 'VD_3145_VV', 'VD_3145_VH', 'VD_3146_VV', 'VD_3146_VH', 'VD_3159_VV', 'VD_3159_VH',
    'VD_3167_VV', 'VD_3167_VH', 'VD_3175_VV', 'VD_3175_VH', 'VD_3180_VV', 'VD_3180_VH', 'VD_3197_VV', 'VD_3197_VH', 'VD_3198_VV',
    'VD_3198_VH', 'VD_3240_VV', 'VD_3240_VH', 'VD_3249_VV', 'VD_3249_VH', 'VD_3258_VV', 'VD_3258_VH', 'VD_3266_VV', 'VD_3266_VH',
    'VD_3280_VV', 'VD_3280_VH', 'VD_3340_VV', 'VD_3340_VH', 'VD_3518_VV', 'VD_3518_VH', 'VD_3649_VV', 'VD_3649_VH', 'VD_3652_VV',
    'VD_3652_VH', 'VD_3657_VV', 'VD_3657_VH', 'VD_3661_VV', 'VD_3661_VH', 'VD_3662_VV', 'VD_3662_VH', 'VD_3669_VV', 'VD_3669_VH',
    'VD_3672_VV', 'VD_3672_VH', 'VD_3691_VV', 'VD_3691_VH', 'VD_3704_VV', 'VD_3704_VH', 'VD_3719_VV', 'VD_3719_VH', 'VD_3724_VV',
    'VD_3724_VH', 'VD_3729_VV', 'VD_3729_VH', 'VD_3732_VV', 'VD_3732_VH', 'VD_3734_VV', 'VD_3734_VH', 'VD_3753_VV', 'VD_3753_VH',
    'VD_3758_VV', 'VD_3758_VH', 'VD_3935_VV', 'VD_3935_VH', 'VD_3936_VV', 'VD_3936_VH', 'VD_3940_VV', 'VD_3940_VH', 'VD_3941_VV',
    'VD_3941_VH', 'VD_3942_VV', 'VD_3942_VH', 'VD_3944_VV', 'VD_3944_VH', 'VD_3949_VV', 'VD_3949_VH', 'VD_3954_VV', 'VD_3954_VH',
    'VD_3958_VV', 'VD_3958_VH', 'VD_3960_VV', 'VD_3960_VH', 'VD_3963_VV', 'VD_3963_VH', 'VD_3965_VV', 'VD_3965_VH', 'VD_3967_VV',
    'VD_3967_VH', 'VD_3968_VV', 'VD_3968_VH', 'VD_3975_VV', 'VD_3975_VH', 'VD_3976_VV', 'VD_3976_VH', 'VD_3982_VV', 'VD_3982_VH',
    'VD_3986_VV', 'VD_3986_VH', 'VD_3988_VV', 'VD_3988_VH', 'VD_3989_VV', 'VD_3989_VH', 'VD_3990_VV', 'VD_3990_VH', 'VD_4004_VV',
    'VD_4004_VH', 'VD_4007_VV', 'VD_4007_VH', 'VD_4008_VV', 'VD_4008_VH', 'VD_4086_VV', 'VD_4086_VH', 'VD_4251_VV', 'VD_4251_VH',
    'VD_4260_VV', 'VD_4260_VH', 'VD_4269_VV', 'VD_4269_VH', 'VD_4273_VV', 'VD_4273_VH', 'VD_4287_VV', 'VD_4287_VH', 'VD_4297_VV',
    'VD_4297_VH', 'VD_4299_VV', 'VD_4299_VH', 'VD_4305_VV', 'VD_4305_VH', 'VD_4313_VV', 'VD_4313_VH', 'VD_4315_VV', 'VD_4315_VH',
    'VD_4316_VV', 'VD_4316_VH', 'VD_4320_VV', 'VD_4320_VH', 'VD_4324_VV', 'VD_4324_VH', 'VD_4337_VV', 'VD_4337_VH', 'VD_4346_VV',
    'VD_4346_VH', 'VD_4354_VV', 'VD_4354_VH', 'VD_4366_VV', 'VD_4366_VH', 'VD_4370_VV', 'VD_4370_VH', 'VD_4373_VV', 'VD_4373_VH',
    'VD_4375_VV', 'VD_4375_VH', 'VD_4379_VV', 'VD_4379_VH', 'VD_4382_VV', 'VD_4382_VH', 'VD_4392_VV', 'VD_4392_VH', 'VD_4394_VV',
    'VD_4394_VH', 'VD_4404_VV', 'VD_4404_VH', 'VD_4408_VV', 'VD_4408_VH', 'VD_4414_VV', 'VD_4414_VH', 'VD_4422_VV', 'VD_4422_VH',
    'VD_4432_VV', 'VD_4432_VH', 'VD_4449_VV', 'VD_4449_VH', 'VD_4450_VV', 'VD_4450_VH', 'VD_4455_VV', 'VD_4455_VH', 'VD_4456_VV',
    'VD_4456_VH', 'VD_4465_VV', 'VD_4465_VH', 'VD_4467_VV', 'VD_4467_VH', 'VD_4472_VV', 'VD_4472_VH', 'VD_4479_VV', 'VD_4479_VH',
    'VD_4482_VV', 'VD_4482_VH', 'VD_4488_VV', 'VD_4488_VH', 'VD_4491_VV', 'VD_4491_VH', 'VD_4492_VV', 'VD_4492_VH', 'VD_4781_VV',
    'VD_4781_VH', 'VD_4788_VV', 'VD_4788_VH', 'VD_4822_VV', 'VD_4822_VH', 'VD_4843_VV', 'VD_4843_VH', 'VD_4867_VV', 'VD_4867_VH',
    'VD_4922_VV', 'VD_4922_VH', 'VD_4933_VV', 'VD_4933_VH', 'VD_4941_VV', 'VD_4941_VH', 'VD_4955_VV', 'VD_4955_VH', 'VD_4967_VV',
    'VD_4967_VH', 'VD_4968_VV', 'VD_4968_VH', 'VD_4969_VV', 'VD_4969_VH', 'VD_4972_VV', 'VD_4972_VH', 'VD_4982_VV', 'VD_4982_VH',
    'VD_4983_VV', 'VD_4983_VH', 'VD_4994_VV', 'VD_4994_VH', 'VD_4997_VV', 'VD_4997_VH', 'VD_5000_VV', 'VD_5000_VH', 'VD_5001_VV',
    'VD_5001_VH', 'VD_5003_VV', 'VD_5003_VH', 'VD_5009_V', 'VD_5009_VH', 'VD_5013_VV', 'VD_5013_VH', 'VD_5014_VV', 'VD_5014_VH',
    'VD_5015_VV', 'VD_5015_VH', 'VD_5016_VV', 'VD_5016_VH', 'VD_5017_VV', 'VD_5017_VH', 'VD_5018_VV', 'VD_5018_VH', 'VD_5019_VV',
    'VD_5019_VH', 'VD_5020_VV', 'VD_5020_VH', 'VD_5021_VV', 'VD_5021_VH', 'VD_5284_VV', 'VD_5284_VH', 'VD_5294_VV', 'VD_5294_VH',
    'VD_5304_VV', 'VD_5304_VH', 'VD_5307_VV', 'VD_5307_VH', 'VD_5316_VV', 'VD_5316_VH', 'VD_5320_VV', 'VD_5320_VH', 'VD_5325_VV',
    'VD_5325_VH', 'VD_5335_VV', 'VD_5335_VH', 'VD_5345_VV', 'VD_5345_VH', 'VD_5358_VV', 'VD_5358_VH', 'VD_5362_VV', 'VD_5362_VH',
    'VD_5374_VV', 'VD_5374_VH', 'VD_5383_VV', 'VD_5383_VH', 'VD_5423_VV', 'VD_5423_VH', 'VD_5429_VV', 'VD_5429_VH', 'VD_5441_VV',
    'VD_5441_VH', 'VD_5451_VV', 'VD_5451_VH', 'VD_5454_VV', 'VD_5454_VH', 'VD_5455_VV', 'VD_5455_VH', 'VD_5456_VV', 'VD_5456_VH',
    'VD_5458_VV', 'VD_5458_VH', 'VD_5459_VV', 'VD_5459_VH', 'VD_5463_VV', 'VD_5463_VH', 'VD_5471_VV', 'VD_5471_VH', 'VD_5474_VV',
    'VD_5474_VH', 'VD_5487_VV', 'VD_5487_VH', 'VD_5494_VV', 'VD_5494_VH', 'VD_5520_VV', 'VD_5520_VH', 'VD_5726_VV', 'VD_5726_VH',
    'VD_5750_VV', 'VD_5750_VH', 'VD_5758_VV', 'VD_5758_VH', 'VD_5772_VV', 'VD_5772_VH', 'VD_5789_VV', 'VD_5789_VH', 'VD_5798_VV',
    'VD_5798_VH', 'VD_5808_VV', 'VD_5808_VH', 'VD_5853_VV', 'VD_5853_VH', 'VD_5870_VV', 'VD_5870_VH',



    
]

labels_visible_on_rc_and_align_partly: list[str] = [
    'VD_8', 'VD_8', 'VD_9_VV', 'VD_9_VH', 'VD_24_VH', 'VD_24_VV', 'VD_28_VH', 'VD_28_VV', 'VD_34_VH', 'VD_34_VV','VD_35_VH', 'VD_35_VV',
    'VD_38_VH', 'VD_38_VV', 'VD_42_VH', 'VD_42_VV', 'VD_48_VH', 'VD_48_VV', 'VD_55_VH', 'VD_55_VV', 'VD_60_VH', 'VD_60_VV', 'VD_64_VH', 
    'VD_64_VV', 'VD_80_VH', 'VD_80_VV', 'VD_76_VV', 'VD_81_VH', 'VD_81_VV', 'VD_83_VH', 'VD_83_VV', 'VD_104_VV', 'VD_104_VH', 'VD_114_VV',
    'VD_114_VH', 'VD_117_VV', 'VD_117_VH', 'VD_122_VV', 'VD_122_VH', 'VD_126_VV', 'VD_126_VH', 'VD_134_VV', 'VD_134_VH', 'VD_155_VV',
    'VD_155_VH', 'VD_164_VV', 'VD_164_VH', 'VD_187_VV', 'VD_187_VH', 'VD_192_VV', 'VD_192_VH', 'VD_199_VV', 'VD_199_VH', 'VD_202_VV',
    'VD_202_VH', 'VD_246_VV', 'VD_246_VH', 'VD_249_VV', 'VD_249_VH', 'VD_269_VV', 'VD_269_VH', 'VD_275_VV', 'VD_275_VH', 'VD_293_VV',
    'VD_293_VH', 'VD_317_VV', 'VD_317_VH', 'VD_340_VV', 'VD_340_VH', 'VD_358_VV', 'VD_358_VH', 'VD_363_VV', 'VD_363_VH', 'VD_381_VV',
    'VD_381_VH', 'VD_394_VV', 'VD_394_VH', 'VD_395_VV', 'VD_395_VH', 'VD_1036_VV', 'VD_1036_VH', 'VD_1463_VV', 'VD_1463_VH', 'VD_3045_VV',
    'VD_3045_VH', 'VD_3048_VV', 'VD_3048_VH', 'VD_3108_VV', 'VD_3108_VH', 'VD_4439_VV', 'VD_4439_VH', 'VD_5331_VV', 'VD_5331_VH', 'VD_5361_VV',
    'VD_5361_VH', 'VD_5420_VV', 'VD_5420_VH', 'VD_5751_VV', 'VD_5751_VH',
]


total_label_count: int = 4047
all_processed = set(
    labels_visible_on_rc_and_align
    + labels_visible_on_rc_and_align_partly
)
total_unique_processed = len(all_processed)
useable: int = int(len(labels_visible_on_rc_and_align)) + int(len(labels_visible_on_rc_and_align_partly))
percentage_usable: float = (useable / total_label_count) * 100
print('='*50)
print(f'Total Label Count: {total_label_count}')
print(f'Perfet matches: {len(labels_visible_on_rc_and_align)}')
print(f'Partial Matches: {len(labels_visible_on_rc_and_align_partly)}')
print(f'Total Unique Labels Processed: {total_unique_processed} ({total_unique_processed / total_label_count * 100:.2f}%)')
print('='*50)
print(f'Total Labels useable on RC: {useable}\n% of the total label: {percentage_usable:.2f}%')
print('='*50)