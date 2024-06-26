import pandas as pd
import pandas
import numpy as np
from pandas import ExcelWriter

egrid_data_xlsx = '../Data/GridInputData/2022Final/eGRID2021_data_real_original.xlsx'
egrid_unt = pandas.read_excel(egrid_data_xlsx, 'UNT'+'21', skiprows=[0])
egrid_unt['orispl_unit'] = egrid_unt.ORISPL.map(str) + '_' + egrid_unt.UNITID.map(str)
egrid_gen = pandas.read_excel(egrid_data_xlsx, 'GEN'+'21', skiprows=[0])
egrid_plnt = pandas.read_excel(egrid_data_xlsx, 'PLNT'+'21', skiprows=[0])
egrid_unt['orispl_unit'] = egrid_unt.ORISPL.map(str) + '_' + egrid_unt.UNITID.map(str)
egrid_gen['orispl_unit'] = egrid_gen['ORISPL'].map(str) + '_' + egrid_gen['GENID'].map(str)

wecc_orispl = np.unique(egrid_plnt[egrid_plnt['NERC']=='WECC']['ORISPL'].values)
orispl_to_check = egrid_gen.loc[(egrid_gen['FUELG1'].isin(['NG', 'DFO',  'SUB', 'LIG', 'LFG', 'WO',
       'COG', 'BLQ', 'BIT',  'WDS', 'OBG', 'OBS', 'RC', 'OG',
       'PUR', 'MSW', 'WH', 'JF', 'PRG', 'PG', 'PC', 'SLW', 'AB', 'KER',
       'RFO', 'WDL', 'SGP', 'SGC', 'OTH', 'TDF', 'OBL', 'BFG', 'WC']))&(egrid_gen['ORISPL'].isin(wecc_orispl))]['ORISPL'].unique()

# gen to unt mapping
orispl_to_fix = {141:{'AF1':'1', 'AF2':'2', 'AF3':'3'},
                 160:{'ST1':'1', 'ST2':'2', 'ST3':'3', 'GT4':'4'},
                 56948:{str(k):'CT'+str(k) for k in np.arange(1, 13)},
                 6177:{'CO1':'U1B', 'CO2':'U2B'},
                 124:{'GT2':'GT1'},
                 55129:{'CTG1':'DBG1', 'CTG2':'DBG2'},
                 60768:{'ST10':'CTG3'},
                 55306:{'CTG1':'1CTGA', 'CTG2':'1CTGB', 'CTG3':'2CTGA', 'CTG4':'2CTGB',
                        'CTG5':'3CTGA', 'CTG6':'3CTGB', 'CTG7':'4CTGA', 'CTG8':'4CTGB'},
                 55124:{'CTG1':'P1', 'CTG2':'P2'},
                 126:{'ST3':'3'},
                 147:{'KY7':'K-7'},
                55481:{'GT1':'1', 'GT2':'2', 'GT3':'5', 'GT6':'6'},
                56616:{'G3':'4PB'},
                 55455:{'CT1A':'CC1A', 'CT1B':'CC1B', 'CT2A':'CC2A', 'CT2B':'CC2B'},
                 118:{'GE1':'CT3'},
                 8068:{'ST'+str(k):str(k) for k in ['5A', '5B', '6A']},
                 55522:{'CT'+str(k):'CT0'+str(k) for k in np.arange(1, 10)},
                 117:{'C4-1':'CC4', 'C5-1':'CC5A', 'C5-2':'CC5B'},
                 120:{'ST1':'1', 'GT5':'CT5', 'GT6':'CT6'},
                 54694:{'GEN1':'4101'},
                 62115:{'1S':'HRSG1A'},
                 62116:{'1S':'HRSG1A'},
                 55951:{'AMP1':'AMP-1'},
                 57564:{'CTG':'8'},
                 56706:{'AMPC':'SD200'},
                 10684:{'TG5':'BLR22', 'TG8':'BLR25', 'TG9':'BLR26'},
                 10650:{'GEN1':'GT1'},
                 56474:{'1':'CT1'},
                 10649:{'GEN1':'GT1'},
                 10764:{'GEN1':'Zurn'},
                 55295:{'CT1':'1', 'CT2':'2'},
                 10652:{'GEN1':'BLR1'},
                 302:{'CEC '+str(k):str(k) for k in np.arange(6, 11)},
                 55508:{'CPP2':'GT-1', '8':'GT-2'},
                 55510:{'CPP4':'GT-1'},
                 55513:{'CPP7':'GT-1'},
                 55499:{'CPP1':'GT-1'},
                 57027:{'CPP'+str(k):str(k) for k in np.arange(1, 5)},
                 10169:{'GEN1':'D1'},
                 56475:{'1':'CT1'},
                 55540:{'GEN1':'1A'},
                 10623:{'GEN1':'1', 'GEN2':'2'},
                 50131:{'K100':'1'},
                 10661:{'GEN5':'4'},
                 56532:{'A':'CT1', 'B':'CT2'},
                 10840:{'GEN1':'U1', 'GEN2':'U2'},
                 50632:{'GEN1':'1'},
                 55625:{'CT1':'UNIT1'},
                 55512:{'CPP6':'GT-1'},
                 55333:{'CTG'+str(k):str(k) for k in [1,2,3]},
                 10300:{'GEN1':'BLR1'},
                 60100:{'G-1':'B-1'},
                 56026:{'CTG1':'PCT1', 'CTG1':'PCT2'},
                 260:{'CT1A':'1A', 'CT1B':'2A', 'CT2A':'3A', 'CT2B':'4A'},
                 389:{'2A':'2-2', '31':'3-1', '32':'3-2'},
                 56707:{'AMPC':'SD200'},
                 55400:{'CTG1':'CTG-1', 'CTG2':'CTG-2'},
                 50530:{'GEN3':'H43'},
                 10052:{'GEN1':'BLR1'},
                 55847:{'CTG1':'UNIT1'},
                 56476:{'A':'GT1', 'B':'GT2'},
                 55810:{'S3':'S-3', 'S4':'S-4', 'S5':'S-5'},
                 11034:{'GEN1':'S-100'},
                 54749:{'CTG':'1'},
                 55627:{'CT1':'UNIT1'},
                 10349:{'GEN1':'2'},
                 399:{'10A':'**10A', '10B':'**10B', 'GT10':'10', 'GT11':'11', 'GT12':'12', 'GT13':'13', 'GT14':'14'},
                 50541:{'ST1':'HR200', 'ST2':'HRSG1'},
                 10777:{'GEN1':'ZURN1'},
                 50049:{'GEN1':'BLRA', 'GEN2':'BLRB', 'GEN3':'BLRC'},
                 55541:{'CTG'+str(k):str(k) for k in [1,2,3]},
                 10294:{'STG':'2'},
                 10405:{'GEN1':'1'},
                 55151:{'GEN'+str(k):'CTG-'+str(k) for k in [1,2,3,4]},
                 7987:{'L1':'01'},
                 55626:{'CT1':'UNIT1'},
                 55542:{'CTG1':'1', 'CTG2':'2'},
                 54768:{'GEN1':'GT1'},
                 341:{'CT'+str(k):str(k) for k in [1,2,3,4]},
                 55217:{'CTG1':'X724', 'CTG2':'X725'},
                 10342:{'TG3':'EIHRSG'},
                 57483:{'CTG'+str(k):'GT-'+str(k) for k in [1,2,3,4]},
                 54912:{'STG1':'HRSG1'},
                 56471:{'1':'CT1'},
                 50612:{'GEN1':'GT1'},
                 55393:{'CTG1':'1', 'CTG2':'2'},
                 56473:{'1':'CT1'},
                 358:{'MV3A':'3-1', 'MV3B':'3-2', 'MV4A':'4-1', 'MV4B':'4-2'},
                 54626:{'TG01':'BL01'},
                 6013:{'O1':'01', 'O2':'02'},
                 55345:{'1-01':'CTG-1', '1-02':'CTG-2'},
                 50560:{'GEN1':'B101'},
                 55656:{'CT01':'CT001', 'CT02':'CT002', 'CT04':'CT004'},
                 10472:{'GEN1':'B300'},
                 10767:{'GEN1':'CFB'},
                 10772:{'GEN1':'CFB'},
                 55963:{'CTG1':'1'},
                 56469:{'1':'FW-1'},
                 56298:{'0001':'CT001','0002':'CT002'},
                 56467:{'CTG1':'CT1', 'CTG2':'CT2'},
                 50865:{'K100':'1'},
                 50864:{'K100':'1'},
                 7551:{'CT1A':'1A', 'CT1B':'1B', 'CT1C':'1C'},
                 57482:{'CTG'+str(k):str(k) for k in np.arange(1, 9)},
                 50110:{'GEN1':'BLR1'},
                 10144:{'GEN4':'BLR3'},
                 50112:{'GEN3':'BLR1'},
                 50837:{'GEN1':'UNIT1'},
                 7552:{'CCCT':'1'},
                 59658:{'GEN1':'BLR1'},
                 60698:{'GT1':'D1', 'GT2':'D7'},
                 54238:{'STG':'BIOMS1'},
                 55182:{'X718':'CTG1', 'X719':'CTG2'},
                 50881:{'GEN'+str(k):'BLR'+str(k) for k in [1,2,3]},
                 55855:{'CTG1':'UNIT1'},
                 7266:{'NA1':'1'},
                 10836:{'GEN1':'BLR1'},
                 55200:{'UN5':'CT5', 'UN6':'CT6'},
                 55645:{'CT01':'CT-01', 'CT02':'CT-02'},
                 10003:{'GEN1':'BLR1', 'GEN2':'BLR2'},
                 6021:{'1':'C1', '2':'C2', '3':'C3'},
                 55453:{'S'+str(k):str(k) for k in [1,2,3,4,5,6]},
                 525:{'1':'H1', '2':'H2'},
                 50707:{'LMA':'S001', 'LMB':'S002', 'LMC':'S003', 'LMD':'S004', 'LME':'S005'},
                 55127:{'UN1':'CT1', 'UN2':'CT2'},
                 56998:{'GT1':'CT01', 'GT2':'CT02', '4':'CT04', '5':'CT05', '6':'CT06', '7':'CT07'},
                 6761:{'1':'101'},
                 8219:{'GT1':'2', 'GT2':'3'},
                 55835:{'CTG1':'1', 'CTG2':'2'},
                 56445:{'GEN1':'CT-01', 'GEN2':'CT-02'},
                 55207:{'UN7':'CT7', 'UN8':'CT8'},
                 10504:{'1500':'SB1', '2500':'SB2'},
                 55744:{'1':'CT01'},
                 50637:{'GEN1':'1PWR', 'GEN2':'2PWR', 'GEN3':'4PWR', 'GEN4':'5REC'},
                 7953:{'1':'CT1', '2':'CT2', '3':'CT3'},
                 57028:{'GTG':'CT1'},
                 55179:{'CTG1':'CTGEN1'},
                 10784:{'GEN1':'CBLR'},
                 56908:{'1':'1A', '2':'2A', '3':'3A'},
                 55749:{'UNT1':'U1'},
                 58284:{'0001':'1'},
                 55514:{'CTG1':'CTG01', 'CTG2':'CTG02'},
                 55322:{'CTG1':'CTG-1', 'CTG2':'CTG-2', 'CTG3':'CTG-3', 'CTG4':'CTG-4'},
                 2322:{str(k):str(k)+'A' for k in np.arange(11, 23)},
                 55077:{'ED01':'EDE1', 'ED02':'EDE2'},
                 7082:{'GT1':'**3', 'GT4':'**4', '5':'**5', '6':'**6'},
                 55687:{'A01':'BHG1', 'A02':'BHG2'},
                 10761:{'GEN'+str(k):str(k) for k in [1,2,3,4,5]},
                 54350:{'STM':'AUX'},
                 54349:{'STM':'AUX'},
                 54271:{'STG':'HRSG1'},
                 55841:{'CT1':'A01', 'CT2':'A03'},
                 54854:{'5657':'3', '5658':'4', '5659':'5'},
                 10869:{'GE':'NORTH', 'GE-2':'SOUTH'},
                 58503:{'GEN1':'CTEU1'},
                 50650:{'GEN1':'BLR1'},
                 7350:{'1':'CTG1', '2':'CTG2'},
                 7931:{'2':'SG02'},
                 50396:{'GEN'+str(k):'BLR'+str(k) for k in [1, 2, 6]},
                 58109:{'TG1':'REC1', 'TG2':'REC2'},
                 56192:{'1':'FBB'},
                 54761:{'GEN1':'1', 'GEN2':'2'},
                 55328:{'CTG1':'CTG-1', 'CTG2':'CTG-2'},
                 50191:{'TG4':'REC4'},
                 56227:{'1':'PWEU1'},
                 55478:{'0001':'CT1'},
                 3456:{'5CT1':'**4', '5CT2':'**5', 'CT1':'GT-6A', 'CT2':'GT-6B'},
                 7790:{'1':'1-1'},
                 56102:{'CT1A':'CTG1A', 'CT1B':'CTG1B'},
                 6481:{'1':'1SGA', '2':'2SGA'},
                 56253:{'MC1':'MC-1', 'MC2':'MC-2'},
                 56177:{'GT1':'U1'},
                 50951:{'GEN1':'1'},
                 50185:{'TG1':'1RB', 'TG2':'2RB'},
                 7870:{'CTG1':'CT1', 'CTG2':'CT2', 'CTG3':'CT3'},
                 54537:{'CT1A':'CT-1A', 'CT1B':'CT-1B'},
                 607:{'3':'CT3', '4':'CT4'},
                 55818:{'FICT':'F1CT'},
                 55482:{'G1':'CT-1'},
                 7999:{'CT1':'1', 'CT2':'2'},
                 55882:{'GEN1':'BLR1'},
                 50886:{'GEN1':'1-1A'},
                 54476:{'GEN1':'CT-1'},
                 3845:{'2':'BW22'},
                 57703:{'01A':'CT01', '01B':'CT02', '02A':'CT03'},
                 4158:{str(k):'BW4'+str(k) for k in [1,2,3,4]},
                 54318:{'TG1':'C', 'TG2':'D'},
                 57915:{str(k):str(k)+'BLR' for k in [1,2,3,4,5,6]},
                 8066:{str(k):'BW7'+str(k) for k in [1,2,3,4]},
                 55477:{'GT2':'CT2'},
                 7504:{'2':'001'},
                 55479:{'0001':'001'},
                 56319:{'0001':'001'},
                 56596:{'5':'001'},
                 6101:{'1':'BW91'}}

egrid_gen_copy = egrid_gen.copy()
for key1, d1 in orispl_to_fix.items():
    for key2, val in d1.items():
        egrid_gen_copy.loc[egrid_gen_copy.loc[(egrid_gen_copy['ORISPL']==key1)&(egrid_gen_copy['GENID']==key2)].index, 'GENID'] = val


with ExcelWriter(
    "../Data/GridInputData/2022Final/eGRID2021_data.xlsx",
    mode="a",
    engine="openpyxl",
    if_sheet_exists="replace",
) as writer:
    egrid_gen_copy.to_excel(writer, sheet_name="GEN21")  
