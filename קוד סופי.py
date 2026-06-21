# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 09:59:29 2026

@author: ilann
"""
# %%

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

output_dir = r"C:\Users\ilann\OneDrive\Desktop\final"

warnings.filterwarnings("ignore")
# %%
# ─────────────────────────────────────────────────────────
# 1. רשימת מניות (זהה לקוד המקורי)
# ─────────────────────────────────────────────────────────
tickers_usa = [
    "NVDA","GOOGL","GOOG","AAPL","MSFT","AMZN","AVGO","META","TSLA","WMT","BRK.B",
    "LLY","JPM","MU","AMD","XOM","V","JNJ","INTC","ORCL","COST","CSCO",
    "MA","CAT","CVX","NFLX","ABBV","BAC","UNH","KO","LRCX","PG","AMAT",
    "PLTR","MS","HD","PM","GE","GS","MRK","TXN","GEV","RTX","LIN",
    "KLAC","WFC","QCOM","AXP","IBM","C","TMUS","ADI","PEP","PANW","MCD",
    "VZ","NEE","DIS","ANET","BLK","AMGN","BA","T","TJX","STX","APP",
    "TMO","UNP","GILD","SCHW","WDC","CRWD","ISRG","DELL","GLW","ABT","UBER",
    "DE","COP","WELL","APH","ETN","CRM","PFE","BX","HON","PLD","VRT",
    "CB","SPGI","MO","CVS","LOW","LMT","SBUX","BKNG","SYK","PGR","NEM",
    "BMY","COF","DHR","INTU","VRTX","CME","ACN","PWR","PH","NOW","SO",
    "EQIX","ADBE","HWM","TT","MDT","DUK","SNPS","CDNS","WMB","MAR","CEG",
    "HCA","BK","CMI","MCK","FTNT","GD","WM","ADP","CMCSA","ICE","FDX",
    "FCX","MNST","KKR","CSX","PNC","ELV","SLB","JCI","BSX","USB","AMT",
    "UPS","ABNB","MMM","MDLZ","NOC","MCO","APO","VLO","SPG","EOG","ORLY",
    "CI","MPC","KMI","DDOG","SHW","EMR","NXPI","MPWR","CVNA","HLT","PSX",
    "CL","NSC","ITW","COHR","DASH","ECL","CTAS","AON","AEP","CRH","MSI",
    "ROST","WBD","RCL","DLR","TDG","RSG","GM","BKR","APD","TRV","REGN",
    "NKE","AFL","GWW","D","OXY","URI","OKE","SRE","TRGP","PCAR","TFC",
    "TEL","KEYS","LHX","FANG","O","DVN","ALL","TGT","AZO","CTVA","CARR",
    "AJG","MET","NDAQ","PSA","F","AME","NUE","ADSK","COR","EBAY","FAST",
    "EA","TER","MCHP","ETR","COIN","XEL","ROK","EW","CAH","VST","DAL",
    "EXC","TTWO","WAB","HPE","GRMN","FITB","CMG","VTR","IDXX","ON","STT",
    "MSCI","ODFL","AMP","YUM","KR","AIG","ARES","KDP","ED","CCI","BDX",
    "PYPL","ADM","DHI","EME","LYV","HSY","IBKR","CBOE","PEG","CBRE","HIG",
    "TKO","IRM","HUM","EQT","PRU","HAL","JBL","WEC","SYY","PCG","VMC",
    "CCL","PAYX","ROP","MLM","ACGL","LVS","KVUE","STLD","WAT","ZTS","CPRT",
    "AXON","WDAY","KMB","A","CASY","HBAN","VICI","EXR","NTRS","FISV","MTB",
    "RJF","UAL","ATO","AEE","RMD","DTE","EL","IQV","CNC","TDY","DOV",
    "BIIB","GEHC","VRSN","DOW","KHC","FOXA","FICO","IR","OTIS","CNP","WRB",
    "TPL","TPR","NRG","EIX","ROL","PPL","FOX","CINF","AVB","CFG","EXPE",
    "XYL","FE","ES","STZ","EQR","DXCM","FSLR","HUBB","JBHT","AWK","CTSH",
    "WTW","BG","LYB","SYF","NTAP","EXE","TSN","DG","PPG","RF","CHD",
    "KEY","CPAY","VRSK","FIS","NI","CMS","L","DRI","PFG","TROW","AKAM",
    "MTD","SBAC","WST","FFIV","VLTO","PHM","DGX","LH","ULTA","STE","OMC",
    "ALB","LEN","EXPD","CHRW","DD","WSM","BRO","RL","SW","CHTR","EFX",
    "CF","VTRS","HPQ","MRNA","INCY","EVRG","SNA","IFF","GPN","LUV","PKG",
    "LNT","SMCI","ESS","FTV","GIS","DLTR","LII","BR","AMCR","INVH","PTC",
    "TSCO","BEN","WY","ZBH","IP","KIM","TXT","LDOS","IEX","NDSN","NVR",
    "MAA","HST","NWS","GNRC","BALL","GEN","REG","NWSA","APA","EG","LULU",
    "DOC","CSGP","TYL","DECK","J","UDR","CDW","HAS","SOLV","MAS","HII",
    "TRMB","GPC","DVA","AIZ","MKC","ZBRA","GL","BBY","IVZ","GDDY","PNW",
    "BF.B","AVY","COO","PNR","SWK","ERIE","ALGN","CLX","APTV","HRL","SJM",
    "ALLE","CPT","BXP","RVTY","SWKS","PODD","TTD","IT","AES","UHS","DPZ",
    "FRT","JKHY","WYNN","MGM","BAX","HSIC","FDS","ARE","TAP","AOS","BLDR",
    "CRL","NCLH","TECH","MOS","POOL","CAG","CPB","EPAM","Q","AA","AAL",
    "ACI","ACM","AFG","AGCO","ALK","ALLY","ALV","AMG","AMH","AN","ARMK",
    "ARW","AVTR","BAH","BBWI","BIO","BJ","BMRN","BRKR","BURL","BWA","CAR",
    "CCK","CG","CLF","CMC","CNH",
    "CMA","ZION","WAL","FNF","FAF","UNM","MKL","RE","LPLA","LNC","SLM",
    "RDN","COOP","SF","SNV","FLO","WU","MOH","HOLX","EHC","XRAY","MASI",
    "PRGO","MRO","NOV","HP","NFG","SWN","RRC","CHK","OVV","SM","CRC",
    "CTRA","PR","MTDR","ET","WES","M","KSS","GPS","PVH","VFC","HBI",
    "ONON","RH","ETSY","TXRH","EAT","NWL","ELF","PARA","IPG","NYT","SIRI",
    "RBLX","IAC","SFM","XRX","WEX","GNTX","SAIC","PEGA","MANH","TWLO","ZM",
    "DOCU","SNOW","MDB","ESTC","NET","ZS","OKTA","CFLT","HUBS","RHP","SUI",
    "ELS","COLD","STAG","REXR","EGP","CUBE","GLPI","OHI","WPC","NNN","FR",
    "EPRT","PK","IDA","SR","UGI","SWX","OGE","R","FLS","TRN","GATX",
    "GXO","XPO","SAIA","WERN","KNX","LSTR","RXO","DINO",
]

tickers_europe = [
    "AZN.L","HSBA.L","BP.L","SHEL.L","RIO.L","GLEN.L","VOD.L","BT-A.L",
    "BA.L","RR.L","GSK.L","AHT.L","EXPN.L","CPG.L","BATS.L","IMB.L",
    "DGE.L","ULVR.L","RKT.L","PRU.L","LGEN.L","LLOY.L","NWG.L","BARC.L",
    "STAN.L","REL.L","WPP.L","SGE.L","IHG.L","LAND.L","BDEV.L","PSN.L",
    "TW.L","SVT.L","UU.L","SN.L","JD.L","HWDN.L","MNG.L","HLMA.L",
    "RTO.L","IMI.L","WEIR.L","SMT.L","NG.L","SSE.L","CNA.L","RMV.L",
    "AUTO.L","MNDI.L","AV.L","TSCO.L","SBRY.L","MKS.L","KGF.L","NXT.L",
    "IAG.L","SMIN.L","CCH.L","OTB.L","ABF.L","BME.L","EDV.L","AAL.L","FRES.L",
    "MC.PA","OR.PA","RMS.PA","KER.PA","EL.PA","SAN.PA","AIR.PA","BN.PA",
    "BNP.PA","GLE.PA","ACA.PA","AXA.PA","SAF.PA","SU.PA","LR.PA","RI.PA",
    "PUB.PA","VIV.PA","ORA.PA","TTE.PA","CAP.PA","DSY.PA","SGO.PA","AF.PA",
    "RCO.PA","VIE.PA","DG.PA","ML.PA","RNO.PA","EN.PA","HO.PA","ALO.PA",
    "CS.PA","GFC.PA","SW.PA","AM.PA","EDEN.PA","ENG.PA","KEP.PA","COFA.PA",
    "ICAD.PA","NK.PA","RUI.PA",
    "SIE.DE","ALV.DE","MUV2.DE","DBK.DE","VOW3.DE","BMW.DE","MBG.DE",
    "BAS.DE","BAYN.DE","MRK.DE","SAP.DE","HNR1.DE","CON.DE","DTE.DE",
    "DPW.DE","DB1.DE","RWE.DE","EON.DE","LHA.DE","FME.DE","FRE.DE",
    "HEN3.DE","BEI.DE","MTX.DE","SRT3.DE","QIA.DE","ENR.DE","CBK.DE","ZAL.DE",
    "VNA.DE","SY1.DE","HEI.DE","RHM.DE","PUM.DE","WIE.DE","FNTN.DE","BOSS.DE",
    "SHL.DE","FRA.DE","EVK.DE","LEG.DE","NDA.DE","AFX.DE",
    "NESN.SW","ROG.SW","NOVN.SW","ABBN.SW","ZURN.SW","GIVN.SW","SGSN.SW",
    "GEBN.SW","LONN.SW","SIKA.SW","CFR.SW","SCMN.SW","SOON.SW","STMN.SW",
    "PGHN.SW","BARN.SW","TEMN.SW","ADEN.SW","BALN.SW","HELN.SW","SCHP.SW",
    "EMS.SW","CLTN.SW","VACN.SW","TKNA.SW","UBSG.SW","SRENH.SW","BAER.SW",
    "LOGN.SW","ALC.SW","UHR.SW","KNIN.SW","PARTN.SW","VATN.SW","SFSN.SW","PSPN.SW",
    "ASML.AS","HEIA.AS","INGA.AS","NN.AS","RAND.AS","AKZA.AS","WKL.AS",
    "PHG.AS","AD.AS","ADYEN.AS","IMCD.AS","ASM.AS","BESI.AS","LIGHT.AS",
    "ABN.AS","URW.AS","DSM.AS","KPN.AS","PRX.AS","EXO.AS","MT.AS","FLOW.AS",
    "AALB.AS","TKWY.AS","CPT.AS","GLPG.AS","BOKA.AS","CRBN.AS",
    "BBVA.MC","SAN.MC","TEF.MC","IBE.MC","REP.MC","AMS.MC","ELE.MC",
    "ENG.MC","RED.MC","MAP.MC","CLNX.MC","IDR.MC","ITX.MC","AENA.MC",
    "FER.MC","CABK.MC","ACS.MC","BKT.MC","GRF.MC","SAB.MC","ANA.MC",
    "VIS.MC","COL.MC","NTGY.MC",
    "ENI.MI","ENEL.MI","ISP.MI","UCG.MI","G.MI","PRY.MI","RACE.MI","MB.MI",
    "AMP.MI","MONC.MI","REC.MI","DIA.MI","TERNA.MI","TIT.MI","SPM.MI",
    "STLA.MI","LDO.MI","SRG.MI","TRN.MI","INW.MI","BPE.MI","BPM.MI",
    "PIRC.MI","BMED.MI","CPR.MI","IGG.MI","BZZ.MI","FCG.MI","TEN.MI",
    "HM-B.ST","ERIC-B.ST","VOLV-B.ST","SAND.ST","ATCO-A.ST","ATCO-B.ST",
    "SHB-A.ST","SWED-A.ST","SEB-A.ST","NDA-SE.ST","SKF-B.ST","ALFA.ST",
    "ASSA-B.ST","HEXA-B.ST","INVE-B.ST","EVO.ST","EPIA.ST","TEL2-B.ST",
    "TELIA.ST","SECU-B.ST","SSAB-A.ST","BOKI.ST","HUSQ-B.ST","NIBE-B.ST",
    "KINV-B.ST","SINCH.ST","EPI-A.ST","INDU-C.ST","BALD-B.ST","GETI-B.ST",
    "SBB-B.ST","CAST.ST","AXFO.ST","AAK.ST","FABG.ST",
    "NOVO-B.CO","ORSTED.CO","MAERSK-B.CO","CARL-B.CO","COLO-B.CO","DSV.CO",
    "DEMANT.CO","GMAB.CO","NZYM-B.CO","CHR.CO","TRYG.CO","ROCK-B.CO",
    "FLS.CO","AMBU-B.CO","GN.CO","RBREW.CO","DANSKE.CO","VWS.CO","LUN.CO",
    "JYSK.CO","SYDB.CO","ISS.CO","NTG.CO","TOP.CO","SIM.CO","SDF.CO",
    "ALK-B.CO","SPNO.CO",
    "DNB.OL","TEL.OL","EQNR.OL","NHY.OL","ORK.OL","YAR.OL","AKRBP.OL",
    "SUBC.OL","SALM.OL","MHG.OL","LSG.OL","BWLPG.OL","PGS.OL","AUSS.OL",
    "STB.OL","TOM.OL","SCHA.OL","MOWI.OL","GJFS.OL","NOD.OL","VRY.OL",
    "DNO.OL","FRO.OL","NSK.OL","BAKKA.OL","KOG.OL",
    "UPM.HE","FORTUM.HE","NOKIA.HE","STERV.HE","KNEBV.HE","NESTE.HE",
    "WRT1V.HE","CGCBV.HE","ORNBV.HE","ELISA.HE","KESBV.HE","OUT1V.HE",
    "TYRES.HE","METSO.HE","NDA-FI.HE","SAMPO.HE","HUH1V.HE","VALMT.HE",
    "KGCES.HE","KEMIRA.HE","TIETO.HE","FSV1V.HE","CITYCON.HE","YIT.HE",
    "ABI.BR","UCB.BR","SOLB.BR","ARGX.BR","GBLB.BR","COLR.BR","KBC.BR",
    "PROX.BR","WDP.BR","TNET.BR","SOFB.BR","EKTA.BR","AGS.BR","AED.BR",
    "COFB.BR","ELI.BR","GLPG.BR","MELE.BR","KIN.BR","BEKB.BR","BPOST.BR","TESB.BR",
    "ERST.VI","OMV.VI","VOE.VI","VER.VI","RBI.VI","ANDR.VI","BAWAG.VI",
    "EDP.LS","EDPR.LS","GALP.LS","BCP.LS","JMT.LS","SON.LS","NOS.LS",
    
    ##more
    "HLN.L", "CRDA.L", "SGRO.L", "ENT.L", "PHNX.L", "WTB.L", "BKG.L", "SDR.L", "INF.L", "SPX.L",
    "BUR.L", "OCDO.L", "ANTO.L", "DCC.L", "RS1.L", "SNN.L", "MRO.L", "PENN.L", "UTG.L", "HL.L",
    "FCIT.L", "AC.PA", "FR.PA", "GET.PA", "MF.PA", "GTT.PA", "UBI.PA", "TEP.PA", "BVI.PA", "DIM.PA",
    "NXI.PA", "POM.PA", "VONN.DE", "PAH3.DE", "DTG.DE", "FPE3.DE", "G1A.DE", "KRN.DE", "AIXA.DE", "NEM.DE",
    "EVT.DE", "KGX.DE", "RTL.DE", "G24.DE", "SIG.SW", "TECN.SW", "BKW.SW", "GALN.SW", "SPSN.SW", "DKSH.SW",
    "HOLN.SW", "SUN.SW", "ARCAD.AS", "PNL.AS", "SBM.AS", "HEIAM.AS", "AMG.AS", "FUR.AS", "COR.AS", "ALFEN.AS",
    "MEL.MC", "FDR.MC", "ROVI.MC", "SRE.MC", "LOG.MC", "PST.MI", "NEXI.MI", "A2A.MI", "ERG.MI", "IRE.MI",
    "AZM.MI", "FBK.MI", "IP.MI", "THULE.ST", "LATO-B.ST", "INDT.ST", "HOLM-B.ST", "DOM.ST", "MYCR.ST", "BAVA.CO",
    "NETC.CO", "AKER.OL", "VAR.OL"
    
    #even more "BRBY.L", "SMDS.L", "JMAT.L", "HIK.L", "BAB.L", "DRX.L", "TATE.L", "IGP.L", "BVIC.L", "VTY.L",
    "CRST.L", "GNC.L", "INCH.L", "BEZ.L", "MONY.L", "PSH.L", "SFR.L", "OSL.L", "BOY.L", "ELM.L",
    
    # צרפת
    "SO.PA", "IPS.PA", "SPIE.PA", "WLN.PA", "RUB.PA", "NEX.PA", "FDJ.PA", "ADP.PA", "ALD.PA", "VRLA.PA",
    "TFI.PA", "ERO.PA", "AMP.PA", "NEO.PA", "DBV.PA",
    
    # גרמניה
    "LXS.DE", "WAF.DE", "UN01.DE", "FIE.DE", "PSM.DE", "NDX1.DE", "DHER.DE", "SDF.DE", "O2D.DE", "MOR.DE",
    "HFG.DE", "GXI.DE", "KCO.DE", "SOW.DE", "1COV.DE",
    
    # איטליה
    "BRE.MI", "CNHI.MI", "HER.MI", "AMPL.MI", "OVS.MI", "MAI.MI", "IVS.MI", "BPSO.MI", "SFER.MI", "ENAV.MI",
    
    # שווייץ
    "AMS.SW", "JUL.SW", "VIFN.SW", "IDIA.SW", "DOCM.SW", "BOSS.SW", "BUREN.SW", "COMN.SW", "EFGN.SW", "FORN.SW",
    
    # שוודיה
    "LUND-B.ST", "JM.ST", "LOOMIS.ST", "NCC-B.ST", "PEAB-B.ST", "WALL-B.ST", "WIHL.ST", "VITR.ST", "BILL.ST", "HUFV-A.ST",
    
    # נורווגיה
    "NEL.OL", "VEI.OL", "SCHB.OL", "GOGL.OL", "ENTRA.OL",
    
    # הולנד
    "ASRNL.AS", "OCI.AS", "VOPK.AS", "POST.AS", "SLIGR.AS",
    
    # ספרד
    "ALM.MC", "EBRO.MC", "CIE.MC", "PHM.MC", "SACYR.MC",
    
    # דנמרק
    "MATAS.CO", "ZEAL.CO", "DFL.CO", "NKT.CO", "TIV.CO"
]

# %%
# ─────────────────────────────────────────────────────────
# 2. הורדת נתונים
# ─────────────────────────────────────────────────────────
START_DATE = "2004-01-01"                        # תאריך התחלה להורדת נתונים
END_DATE   = "2025-12-31"                        # תאריך סיום להורדת נתונים
USA_BENCH  = "^GSPC"                             # מדד הייחוס האמריקאי - S&P 500
EU_BENCH   = "^STOXX"                            # מדד הייחוס האירופאי - Euro Stoxx
RF_TICKER  = "^IRX"                              # סימן הריבית חסרת הסיכון (13-Week Treasury Bill)

all_stocks = list(set(tickers_usa + tickers_europe))  # מאחד את שתי הרשימות ומסיר כפילויות

print("Downloading stock prices...")             # הדפסת הודעת סטטוס למשתמש
raw    = yf.download(all_stocks, start=START_DATE, end=END_DATE,
                     interval="1d",             # מורד מחירים יומיים
                     auto_adjust=True,          # מתאים מחירים לדיבידנדים ופיצולים
                     progress=False,            # מסתיר פס התקדמות
                     threads=True)              # מוריד במקביל לחיסכון בזמן
prices = raw["Close"].resample("ME").last()     # שומר רק את מחיר הסגירה ומדגם לסוף כל חודש

print("Downloading benchmarks & RF...")         # הדפסת הודעה על הורדת מדדי ייחוס
aux    = yf.download([USA_BENCH, EU_BENCH, RF_TICKER], start=START_DATE, end=END_DATE,
                     interval="1d", auto_adjust=True, progress=False, threads=True)
                                                 # מוריד נתונים של מדדי הייחוס והריבית חסרת הסיכון
aux    = aux["Close"].resample("ME").last()     # שומר סגירה חודשית של המדדים

stock_ret  = prices.pct_change()                # מחשב תשואה חודשית לכל מניה: (מחיר_חדש - מחיר_ישן) / מחיר_ישן
usa_mkt    = aux[USA_BENCH].pct_change()        # תשואה חודשית של S&P 500
eu_mkt     = aux[EU_BENCH].pct_change()         # תשואה חודשית של Euro Stoxx
rf_monthly = aux[RF_TICKER] / 100 / 12          # ריבית חסרת סיכון חודשית (ממירה מאחוזים שנתיים לחודשיים)
# %%
# ─────────────────────────────────────────────────────────
# 3. בניית פאנל נתונים
# ─────────────────────────────────────────────────────────
print("Building panel...")                      # הודעת סטטוס
eu_set  = set(tickers_europe)                   # הופך את רשימת המניות האירופאיות ל-set לחיפוש מהיר
records = []                                     # רשימה ריקה שתאסוף את כל שורות הנתונים

for i, date in enumerate(stock_ret.index[12:], start=12):  # לולאה מחודש 13 ואילך (כדי שיהיה חלון של 12 חודשים)
    fwd_i = i + 1                               # אינדקס חודש קדימה (התשואה שנרצה לחזות)
    if fwd_i >= len(stock_ret):                 # אם הגענו לסוף הנתונים - עצור
        continue

    rf_val = rf_monthly.iloc[fwd_i]             # ריבית חסרת הסיכון לחודש הבא

    for ticker in stock_ret.columns:            # לולאה על כל מניה
        col    = stock_ret[ticker]              # עמודת התשואות של מניה זו
        window = col.iloc[i - 12 : i - 1].dropna()  # חלון של 11 חודשים אחרונים (לא כולל החודש הנוכחי)
        if len(window) < 9:                     # אם יש פחות מ-9 תצפיות - דלג (נתונים חסרים)
            continue

        ret_12_1       = (1 + window).prod() - 1    # מומנטום: תשואה מצטברת של 11 חודשים (12-1)
        forward_return = col.iloc[fwd_i]             # תשואת החודש הבא (משתנה תלוי)
        current_return = col.iloc[i]                 # תשואת החודש הנוכחי

        if pd.isna(forward_return) or pd.isna(current_return):  # דלג אם יש ערכים חסרים
            continue

        region  = "Europe" if ticker in eu_set else "USA"       # מגדיר אם המניה אירופאית או אמריקאית
        mkt_fwd = eu_mkt.iloc[fwd_i] if region == "Europe" else usa_mkt.iloc[fwd_i]
                                                                  # בוחר את תשואת המדד המתאימה לאזור

        records.append({                        # מוסיף שורת נתונים לרשימה
            "Date":      date,                  # תאריך
            "Ticker":    ticker,                # סימן המניה
            "Region":    region,                # אזור גיאוגרפי
            "RET_12_1":  ret_12_1,             # תשואת מומנטום (12 חודשים אחורה, מינוס החודש הנוכחי)
            "Curr_Ret":  current_return,        # תשואת החודש הנוכחי
            "Fwd_Ret":   forward_return,        # תשואת החודש הבא (מה שרוצים לחזות)
            "Exc_Stock": forward_return - rf_val,    # עודף תשואה של המניה מעל ריבית חסרת סיכון
            "Exc_Mkt":   mkt_fwd - rf_val,          # עודף תשואה של המדד מעל ריבית חסרת סיכון
            "Mkt_Ret":   mkt_fwd,              # תשואת המדד הרלוונטי
            "RF":        rf_val,               # ריבית חסרת סיכון
        })

df = pd.DataFrame(records).dropna()            # הופך את הרשימה לטבלה ומסיר שורות עם ערכים חסרים

# ── Winsorize ─────────────────────────────────────────
for col in ["Fwd_Ret", "RET_12_1", "Exc_Stock", "Exc_Mkt"]:  # עבור כל עמודת תשואה
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)   # מחשב אחוזון 1 ו-99
    df[col] = df[col].clip(lo, hi)                             # חותך ערכים קיצוניים לגבולות אלו 
df["Year"] = df["Date"].dt.year                # מוסיף עמודת שנה לסינון לפי תקופות

# ── Z-Score חודשי ─────────────────────────────────────
def add_zscore(df):                             # פונקציה לנירמול המומנטום
    df = df.copy()                              # עובד על עותק כדי לא לשנות את המקור
    df["MOM_Z"] = (
        df.groupby(["Date", "Region"])["RET_12_1"]   # מקבץ לפי חודש ואזור
        .transform(lambda x: (x - x.mean()) / x.std())  # מחשב Z-Score: (ערך - ממוצע) / סטיית תקן
    )
    return df

df = add_zscore(df)                             # מפעיל את פונקציית הנירמול על הטבלה

print(f"Panel ready: {len(df):,} observations") # מדפיס כמה שורות יש בסך הכל
# %%
#################שמירת טבלאות############3
df.to_csv(
    rf"{output_dir}\momentum_panel_full.csv",
    index=False,
    encoding="utf-8-sig"
)

# פאנל לרגרסיות בלבד
reg_df = df[[
    "Date",
    "Ticker",
    "Region",
    "Exc_Stock",
    "Exc_Mkt",
    "MOM_Z",
    "Year"
]].copy()

reg_df.to_csv(
    rf"{output_dir}\momentum_regression_panel.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"Full panel saved: {output_dir}\\momentum_panel_full.csv")
print(f"Regression panel saved: {output_dir}\\momentum_regression_panel.csv")
fm_df = df[[
    "Date",
    "Ticker",
    "Region",
    "Year",
    "MOM_Z",
    "Fwd_Ret"
]].copy()

fm_df.to_csv(
    rf"{output_dir}\fama_macbeth_panel.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"Fama-MacBeth panel saved: {len(fm_df):,} observations")
# %%
# ─────────────────────────────────────────────────────────
# 4. רגרסיות
# ─────────────────────────────────────────────────────────
SUB_PERIODS = [(2005, 2009), (2010, 2014), (2015, 2019), (2020, 2025)]  # תתי-תקופות לניתוח

def regressions(data, label):                   # פונקציה שמריצה את כל הרגרסיות עבור תת-קבוצה
    print(f"\n{'─'*60}")
    print(f"  {label}  (n={len(data):,})")     # מדפיס כותרת עם גודל המדגם
    print(f"{'─'*60}")

    # 1. Fama-MacBeth
    betas = []                                  # רשימה לאחסון מקדמי הרגרסיה מכל חודש
    for _, g in data.groupby("Date"):           # לולאה על כל חודש בנפרד
        if len(g) < 20:                         # דלג אם יש פחות מ-20 מניות באותו חודש
            continue
        res = sm.OLS(g["Fwd_Ret"],
                     sm.add_constant(g["MOM_Z"])).fit()  # רגרסיה OLS: תשואה עתידית ~ קבוע + מומנטום
        betas.append(res.params["MOM_Z"])       # שומר את מקדם המומנטום
    if betas:
        b = pd.Series(betas)                    # הופך לסדרה סטטיסטית
        t = b.mean() / (b.std() / np.sqrt(len(b)))  # מחשב t-statistic: ממוצע / שגיאה סטנדרטית
        print(f"[1] Fama-MacBeth (z-score)   coeff={b.mean():.4f}  t={t:.2f}")  # מדפיס תוצאה

    # 2. CAPM
    m2 = smf.ols("Exc_Stock ~ Exc_Mkt", data=data).fit()  # רגרסיית CAPM: עודף תשואה ~ עודף שוק
    print(f"[2] CAPM            alpha={m2.params['Intercept']:.4f}"
          f"(p={m2.pvalues['Intercept']:.3f})  "
          f"beta_mkt={m2.params['Exc_Mkt']:.4f}")   # מדפיס: alpha (תשואה עודפת), beta (רגישות לשוק)

    # 3. CAPM + Momentum
    m3 = smf.ols("Exc_Stock ~ Exc_Mkt + MOM_Z", data=data).fit()  # CAPM עם פקטור מומנטום
    print(f"[3] CAPM+Mom(z)     alpha={m3.params['Intercept']:.4f}"
          f"(p={m3.pvalues['Intercept']:.3f})  "
          f"beta_mkt={m3.params['Exc_Mkt']:.4f}  "
          f"beta_mom={m3.params['MOM_Z']:.4f}"
          f"(p={m3.pvalues['MOM_Z']:.3f})")          # מדפיס: alpha, beta שוק, beta מומנטום

for region in ["USA", "Europe"]:                # לולאה על שני האזורים
    sub = df[df["Region"] == region]            # מסנן נתונים לפי אזור
    print(f"\n{'═'*60}")
    print(f"  REGION: {region}")
    print(f"{'═'*60}")
    regressions(sub, "Full Sample 2005-2025")   # מריץ רגרסיה על כל התקופה
    for s, e in SUB_PERIODS:                    # לולאה על תתי-התקופות
        regressions(sub[(sub["Year"] >= s) & (sub["Year"] <= e)], f"{s}-{e}")  # מריץ רגרסיה לכל תת-תקופה
# %%
# ─────────────────────────────────────────────────────────
# 5. תיק Long/Short
# ─────────────────────────────────────────────────────────
def long_short(data, label):                    # פונקציה לבניית תיק מומנטום Long/Short
    rows = []                                   # רשימה לאחסון תוצאות חודשיות
    for date, g in data.groupby("Date"):        # לולאה על כל חודש
        if len(g) < 20:                         # דלג אם יש פחות מ-20 מניות
            continue
        p10 = g["RET_12_1"].quantile(0.10)     # אחוזון 10 של מומנטום (המפסידנות)
        p90 = g["RET_12_1"].quantile(0.90)     # אחוזון 90 של מומנטום (המנצחות)
        win = g[g["RET_12_1"] >= p90]["Fwd_Ret"].mean()  # תשואה עתידית ממוצעת של 10% המניות עם מומנטום הגבוה
        los = g[g["RET_12_1"] <= p10]["Fwd_Ret"].mean()  # תשואה עתידית ממוצעת של 10% המניות עם מומנטום הנמוך
        rows.append({
            "Date": date,
            "Win":  win,                        # תשואת פוזיציית הלונג (קניה)
            "Los":  los,                        # תשואת פוזיציית השורט (מכירה בחסר)
            "WML":  win - los,                  # הפרש: Winners Minus Losers (רווח תיק הL/S)
            "Mkt":  g["Mkt_Ret"].iloc[0],      # תשואת השוק באותו חודש
        })
    if not rows:                                # אם אין נתונים - החזר None
        return None

    r = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)  # ממיין לפי תאריך

    ann_ret = r["WML"].mean() * 12              # תשואה שנתית: ממוצע חודשי × 12
    ann_vol = r["WML"].std()  * np.sqrt(12)    # תנודתיות שנתית: סטיית תקן חודשית × √12
    t_stat  = r["WML"].mean() / (r["WML"].std() / np.sqrt(len(r)))  # t-statistic לבדיקת מובהקות
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0  # יחס שארפ: תשואה עודפת / סיכון

    print(f"\n{'─'*50}")
    print(f"Long/Short — {label}")
    print(f"  Monthly WML mean : {r['WML'].mean():.4f}")   # ממוצע הפרש חודשי
    print(f"  Ann. Return      : {ann_ret:.2%}")            # תשואה שנתית באחוזים
    print(f"  Ann. Volatility  : {ann_vol:.2%}")            # תנודתיות שנתית
    print(f"  Sharpe Ratio     : {sharpe:.2f}")             # יחס שארפ
    print(f"  t-stat           : {t_stat:.2f}")             # t-statistic
    return r                                    # מחזיר את טבלת התוצאות


ls_usa = long_short(df[df["Region"] == "USA"],    "USA")     # מחשב Long/Short לארה"ב
ls_eu  = long_short(df[df["Region"] == "Europe"], "Europe")  # מחשב Long/Short לאירופה
# %%
# ─────────────────────────────────────────────────────────
# 6. גרפים
# ─────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")               # מגדיר סגנון גרפים עם רשת לבנה

def cum_compound(s):                            # פונקציה לחישוב תשואה מצטברת עם השקעה מחדש
    """תשואה מצטברת עם השקעה מחדש — $1 בתחילה"""
    return (1 + s).cumprod() - 1               # מכפיל: (1+r1)×(1+r2)×... ומחסיר 1

# ── 6a. גרף Long/Short vs שוק ──────────────────────────
def plot_ls(res, label, market_label):          # פונקציה לציור גרף ביצועי L/S
    fig, ax = plt.subplots(figsize=(13, 6))    # יוצר מסגרת גרף בגודל 13×6 אינץ'
    ax.plot(res["Date"], cum_compound(res["Win"]),
            color="#2ca02c", lw=2,   label="Long (Winners)")   # ציור קו תשואת המנצחות בירוק
    ax.plot(res["Date"], cum_compound(res["Los"]),
            color="#d62728", lw=2,   label="Short (Losers)")   # ציור קו תשואת המפסידנות באדום
    ax.plot(res["Date"], cum_compound(res["WML"]),
            color="#1f77b4", lw=2.5, label="W−L (Net)", ls="--")  # ציור קו הפרש L/S בכחול מקווקו
    ax.plot(res["Date"], cum_compound(res["Mkt"]),
            color="black",   lw=1.5, label=market_label, ls=":")   # ציור קו תשואת השוק בשחור מנוקד
    ax.axhline(0, color="black", lw=1, alpha=0.4)              # קו אופקי ב-0%
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))  # פורמט ציר Y באחוזים
    ax.set_title(f"Momentum Long/Short vs Market — {label}\n"
                 f"(Compound Cumulative Return, $1 invested)",
                 fontsize=13, fontweight="bold")                # כותרת הגרף
    ax.set_xlabel("Date")                      # תווית ציר X
    ax.set_ylabel("Cumulative Return (compound)")  # תווית ציר Y
    ax.legend(fontsize=11)                     # הוספת מקרא
    plt.tight_layout()                         # מכוון אוטומטית שולות
    plt.show()                                 # מציג את הגרף

plot_ls(ls_usa, "USA",    "S&P 500")           # מציג גרף L/S לארה"ב
plot_ls(ls_eu,  "Europe", "Euro Stoxx 600")    # מציג גרף L/S לאירופה


# ── 6c. השוואה ישירה ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))         # גרף השוואה בין ארה"ב לאירופה
ax.plot(ls_usa["Date"], cum_compound(ls_usa["WML"]),
        color="#1f77b4", lw=2.5, label="USA  W−L")    # תשואת L/S ארה"ב בכחול
ax.plot(ls_eu["Date"],  cum_compound(ls_eu["WML"]),
        color="#d62728", lw=2.5, label="Europe W−L")  # תשואת L/S אירופה באדום
ax.axhline(0, color="black", lw=1, alpha=0.4)         # קו ייחוס ב-0
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_title("Head-to-Head: Momentum W−L — USA vs Europe\n"
             "(Compound Cumulative Return)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return (compound)")
ax.legend(fontsize=12)
plt.tight_layout()
plt.show()
# %%
# ─────────────────────────────────────────────────────────
# 7. ניתוח עשירונים
# ─────────────────────────────────────────────────────────
def decile_analysis(data, label):               # פונקציה לניתוח תשואות לפי עשירוני מומנטום
    rows = []
    for date, g in data.groupby("Date"):        # לולאה על כל חודש
        if len(g) < 30:                         # דלג אם יש פחות מ-30 מניות
            continue
        g = g.copy()                            # עותק של הנתונים
        try:
            g["D"] = pd.qcut(g["RET_12_1"], q=10,
                              labels=False, duplicates="drop") + 1  # מחלק למניות ל-10 עשירונים לפי מומנטום
        except ValueError:                      # טיפול בשגיאה אם לא ניתן לחלק לעשירונים
            continue
        row = {"Date": date}
        for d in range(1, 11):                  # לולאה על 10 עשירונים
            vals    = g[g["D"] == d]["Fwd_Ret"] # תשואות עתידיות של מניות בעשירון d
            row[f"D{d}"] = vals.mean() if len(vals) > 0 else np.nan  # ממוצע תשואה לעשירון
        rows.append(row)

    if not rows:
        print(f"Not enough data: {label}")
        return None

    dec = pd.DataFrame(rows).set_index("Date").sort_index()  # טבלת עשירונים ממוינת לפי תאריך

    print(f"\n{'═'*60}")
    print(f"  Decile Summary — {label}")
    print(f"{'═'*60}")
    print(f"  {'Decile':<16} {'Ann.Ret':>9} {'Ann.Vol':>9} {'Sharpe':>8} {'Cum.Ret':>10}")
    print(f"  {'─'*52}")
    ann_rets = []
    for d in range(1, 11):                      # לולאה על כל עשירון
        col    = dec[f"D{d}"].dropna()          # נתוני העשירון ללא חסרים
        ar     = col.mean() * 12                # תשואה שנתית
        av     = col.std()  * np.sqrt(12)       # תנודתיות שנתית
        sh     = ar / av if av > 0 else np.nan  # שארפ
        cr     = (1 + col).prod() - 1           # תשואה מצטברת (compound)
        ann_rets.append(ar)
        dlabel = "D1 (Losers)" if d == 1 else ("D10 (Winners)" if d == 10 else f"D{d}")  # שם העשירון
        print(f"  {dlabel:<16} {ar:>9.2%} {av:>9.2%} {sh:>8.2f} {cr:>10.2%}")  # מדפיס שורת סיכום

    # גרף עמודות לפי עשירון
    dlabels = ["D1\n(Losers)"] + [f"D{d}" for d in range(2, 10)] + ["D10\n(Winners)"]  # תוויות עשירונים
    colors  = ["#d62728" if d <= 2 else ("#2ca02c" if d >= 9 else "#aec7e8")
               for d in range(1, 11)]           # צבעים: אדום=הפסד, ירוק=רווח, כחול=ביניים

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(dlabels, [r * 100 for r in ann_rets],
                  color=colors, edgecolor="black", lw=0.6)  # ציור עמודות
    for bar, v in zip(bars, ann_rets):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.3 if v >= 0 else -1.3),
                f"{v:.1%}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")  # כתיבת ערך מעל/מתחת לכל עמודה
    ax.axhline(0, color="black", lw=1)          # קו ייחוס ב-0
    ax.set_title(f"Annualized Return by Momentum Decile — {label}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Momentum Decile  (D1 = Past Losers  →  D10 = Past Winners)")
    ax.set_ylabel("Annualized Return (%)")
    plt.tight_layout()
    plt.show()

    # גרף קווים מצטבר לכל עשירון
    cmap = plt.cm.RdYlGn                        # סקאלת צבעים: אדום-צהוב-ירוק
    fig, ax = plt.subplots(figsize=(13, 7))
    for d in range(1, 11):                      # לולאה על עשירונים
        col   = dec[f"D{d}"].dropna()
        color = cmap((d - 1) / 9)              # צבע הולך מאדום לירוק לפי עשירון
        lw    = 2.5 if d in (1, 10) else 1.1   # קו עבה יותר לעשירונים קיצוניים
        ls    = "--" if d == 1 else ("-" if d == 10 else ":")  # סגנון קו שונה לקיצוניים
        dlbl  = "D1 (Losers)" if d == 1 else ("D10 (Winners)" if d == 10 else f"D{d}")
        ax.plot(col.index, cum_compound(col),
                label=dlbl, color=color, lw=lw, ls=ls)  # ציור קו לכל עשירון
    ax.axhline(0, color="black", lw=1, alpha=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title(f"Cumulative Return by Momentum Decile — {label}\n(Compound)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (compound)")
    ax.legend(fontsize=9, loc="upper left", ncol=2)  # מקרא בשתי עמודות
    plt.tight_layout()
    plt.show()

    return dec                                  # מחזיר את טבלת העשירונים


dec_usa = decile_analysis(df[df["Region"] == "USA"],    "USA")     # ניתוח עשירונים לארה"ב
dec_eu  = decile_analysis(df[df["Region"] == "Europe"], "Europe")  # ניתוח עשירונים לאירופה


# %%







import statsmodels.formula.api as smf

def capm_wml(ls_df, region_name):

    if ls_df is None or len(ls_df) == 0:
        return

    # הוספת ריבית חסרת סיכון חודשית
    rf = (
        df[df["Region"] == region_name]
        .groupby("Date")["RF"]
        .first()
        .reset_index()
    )

    ls_df = ls_df.merge(rf, on="Date", how="left")

    # תשואות עודפות
    ls_df["Exc_WML"] = ls_df["WML"] - ls_df["RF"]
    ls_df["Exc_Mkt"] = ls_df["Mkt"] - ls_df["RF"]

    # CAPM
    model = smf.ols(
        "Exc_WML ~ Exc_Mkt",
        data=ls_df
    ).fit()

    print(f"\n{'─'*60}")
    print(f"CAPM on WML Portfolio — {region_name}")
    print(f"{'─'*60}")

    print(
        f"alpha={model.params['Intercept']:.4f} "
        f"(p={model.pvalues['Intercept']:.3f})  "
        f"beta={model.params['Exc_Mkt']:.4f}"
    )

    return model


# ארה"ב
capm_wml(ls_usa, "USA")

# אירופה
capm_wml(ls_eu, "Europe")
# %%

















































