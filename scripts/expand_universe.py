#!/usr/bin/env python3
"""
Expand universe.yaml with tradeable stocks from public sources.
Fetches S&P 500, NASDAQ-100, Russell 2000 and more.
"""

import yaml
import pandas as pd
from collections import defaultdict

def fetch_sp500():
    """S&P 500 components (static list - updated Jan 2025)."""
    print("Fetching S&P 500...")
    sp500 = [
        # Technology
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "CSCO", "ACN", "ADBE", "IBM",
        "QCOM", "TXN", "INTU", "AMD", "NOW", "AMAT", "ADI", "LRCX", "MU", "KLAC",
        "SNPS", "CDNS", "MCHP", "APH", "MSI", "FTNT", "TEL", "GLW", "ANSS", "KEYS",
        "ON", "FSLR", "ENPH", "MPWR", "SWKS", "TER", "ZBRA", "JNPR", "FFIV", "AKAM",
        # Healthcare
        "UNH", "JNJ", "LLY", "MRK", "ABBV", "TMO", "PFE", "ABT", "DHR", "BMY",
        "AMGN", "MDT", "ISRG", "GILD", "CVS", "ELV", "VRTX", "SYK", "CI", "BSX",
        "REGN", "MCK", "HCA", "ZTS", "BDX", "DXCM", "IQV", "IDXX", "EW", "A",
        "MTD", "RMD", "CAH", "GEHC", "HOLX", "WAT", "ILMN", "ALGN", "TECH", "COO",
        "CNC", "MOH", "BIIB", "VTRS", "HSIC", "DGX", "LH", "CRL", "INCY", "MRNA",
        # Financials
        "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "SPGI",
        "BLK", "C", "PGR", "SCHW", "CB", "CME", "ICE", "AON", "MMC", "MCO",
        "USB", "PNC", "TFC", "AIG", "MET", "ALL", "AFL", "PRU", "TRV", "MSCI",
        "AMP", "BK", "STT", "TROW", "NTRS", "DFS", "FRC", "CFG", "HBAN", "FDS",
        "WRB", "L", "GL", "RJF", "CMA", "ZION", "KEY", "RF", "SIVB", "FITB",
        # Consumer Discretionary
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG",
        "ORLY", "AZO", "ROST", "MAR", "HLT", "GM", "F", "DHI", "LEN", "PHM",
        "GRMN", "POOL", "NVR", "BBY", "ULTA", "DRI", "YUM", "EBAY", "ETSY", "LVS",
        "WYNN", "CCL", "RCL", "NCLH", "MGM", "CZR", "EXPE", "HAS", "DG", "DLTR",
        "KMX", "APTV", "BWA", "LEA", "TPR", "VFC", "RL", "PVH", "HBI", "GPS",
        # Consumer Staples
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "EL",
        "KMB", "GIS", "K", "HSY", "CAG", "SJM", "CPB", "HRL", "MKC", "CHD",
        "CLX", "KR", "SYY", "WBA", "TGT", "STZ", "ADM", "TAP", "TSN", "BG",
        # Industrials
        "CAT", "RTX", "HON", "UNP", "UPS", "DE", "BA", "LMT", "GE", "GD",
        "NOC", "MMM", "FDX", "CSX", "NSC", "WM", "ETN", "ITW", "EMR", "PH",
        "ROK", "CMI", "PCAR", "CARR", "OTIS", "JCI", "TT", "LHX", "SWK", "IR",
        "CPRT", "CTAS", "RSG", "FAST", "GWW", "AME", "ODFL", "XYL", "IEX", "SNA",
        "J", "DOV", "HII", "TDG", "WAB", "GNRC", "EFX", "ROP", "PWR", "VRSK",
        # Energy
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PXD", "VLO", "PSX", "OXY",
        "WMB", "HES", "DVN", "HAL", "KMI", "BKR", "FANG", "CTRA", "MRO", "APA",
        "OKE", "TRGP", "EQT", "DTE", "EIX", "AEE",
        # Utilities
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "WEC", "ES",
        "ED", "PEG", "AWK", "PPL", "CMS", "LNT", "CNP", "NI", "EVRG", "ATO",
        "FE", "NRG", "CEG", "PCG", "PNW",
        # Real Estate
        "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
        "EQR", "VTR", "SBAC", "ARE", "MAA", "UDR", "ESS", "INVH", "WY", "SUI",
        "EXR", "PEAK", "CPT", "REG", "KIM", "HST", "BXP", "VNO", "FRT", "IRM",
        # Materials
        "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "DD", "NUE", "DOW", "PPG",
        "VMC", "MLM", "CTVA", "ALB", "LYB", "CF", "FMC", "MOS", "IFF", "CE",
        "EMN", "PKG", "AVY", "WRK", "SEE", "IP", "BALL", "AMCR",
        # Communication Services
        "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "CHTR", "TMUS",
        "ATVI", "EA", "TTWO", "MTCH", "WBD", "PARA", "FOX", "FOXA", "NWS", "NWSA",
        "OMC", "IPG", "LYV", "DISH",
    ]
    print(f"  Found {len(sp500)} S&P 500 stocks")
    return sp500

def fetch_nasdaq100():
    """NASDAQ-100 components (static list - updated Jan 2025)."""
    print("Fetching NASDAQ-100...")
    nasdaq100 = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "COST",
        "PEP", "CSCO", "NFLX", "ADBE", "AMD", "TMUS", "INTC", "TXN", "CMCSA", "AMGN",
        "HON", "QCOM", "INTU", "AMAT", "BKNG", "ISRG", "SBUX", "ADP", "MDLZ", "ADI",
        "VRTX", "GILD", "MU", "REGN", "PYPL", "LRCX", "PANW", "SNPS", "CDNS", "ASML",
        "KLAC", "CSX", "MELI", "CTAS", "MNST", "MAR", "ORLY", "ABNB", "MRVL", "FTNT",
        "NXPI", "WDAY", "AZN", "ADSK", "DXCM", "PCAR", "PAYX", "CHTR", "KDP", "CPRT",
        "MRNA", "AEP", "KHC", "ROST", "MCHP", "IDXX", "LULU", "EXC", "SGEN", "BIIB",
        "ODFL", "CEG", "EA", "DLTR", "CRWD", "CSGP", "WBD", "ZS", "XEL", "ILMN",
        "VRSK", "FAST", "GFS", "DDOG", "TEAM", "FANG", "ANSS", "BKR", "ALGN", "WBA",
        "ZM", "LCID", "JD", "PDD", "BIDU", "EBAY", "ENPH", "SPLK", "SIRI", "DOCU",
    ]
    print(f"  Found {len(nasdaq100)} NASDAQ-100 stocks")
    return nasdaq100

def fetch_russell2000_sample():
    """Fetch Russell 2000 sample - small cap stocks."""
    print("Fetching Russell 2000 sample...")
    # Russell 2000 isn't easily scraped, use known small caps
    small_caps = [
        # High volatility small caps
        "AMC", "BB", "BBBY", "CLOV", "WISH", "SOFI", "PLTR", "LCID", "RIVN",
        "SPCE", "OPEN", "UPST", "AFRM", "HOOD", "COIN", "RBLX", "SNOW", "DDOG",
        # Biotech small caps
        "SRNE", "INO", "OCGN", "ATOS", "SNDL", "TLRY", "CGC", "ACB", "HEXO",
        # Tech small caps
        "FUBO", "SKLZ", "BARK", "BODY", "LMND", "ROOT", "GOEV", "RIDE", "WKHS",
        # SPACs and recent IPOs
        "DNA", "JOBY", "LILM", "EVTL", "ARBE", "OUST", "LAZR", "VLDR", "INVZ",
        # Mining and materials
        "UUUU", "DNN", "CCJ", "URG", "NXE", "LEU", "LTBR", "SMR",
        # Clean energy
        "PLUG", "FCEL", "BLDP", "BE", "CHPT", "EVGO", "BLNK", "CLSK",
        # Fintech
        "SQ", "PYPL", "AFRM", "FOUR", "BILL", "TOST", "MARQ",
        # Gaming/Entertainment
        "DKNG", "PENN", "GENI", "RSI", "BETZ",
        # Healthcare
        "HIMS", "TALK", "AMWL", "TDOC", "DOCS",
        # Real Estate Tech
        "OPEN", "RDFN", "COMP", "EXPI",
        # AI/ML
        "AI", "PATH", "S", "SNOW", "MDB", "CRWD", "ZS", "NET", "OKTA",
        # Semiconductors
        "SOXL", "SOXS", "SMH", "NVDA", "AMD", "INTC", "MU", "MRVL", "ON",
        # EV/Auto
        "TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "FSR", "NKLA", "HYLN",
        # Space
        "RKLB", "RDW", "ASTS", "BKSY", "PL", "MNTS",
        # Crypto related
        "MARA", "RIOT", "HUT", "BITF", "HIVE", "BTBT", "CAN", "CIFR", "IREN",
        "MSTR", "COIN", "SI",
        # Quantum computing
        "IONQ", "RGTI", "QUBT",
        # 3D Printing
        "DDD", "SSYS", "XONE", "DM", "NNDM",
        # Drones/Robotics
        "AVAV", "KTOS", "UAVS",
        # Streaming/Media
        "ROKU", "SPOT", "PARA", "WBD", "LYV",
        # Travel
        "ABNB", "BKNG", "EXPE", "TRIP",
        # Food/Beverage
        "BYND", "OTLY", "TTCF",
        # Retail
        "CHWY", "ETSY", "W", "PTON", "PRPL",
    ]
    print(f"  Found {len(small_caps)} small cap stocks")
    return small_caps

def fetch_additional_tickers():
    """Additional popular/volatile tickers not in major indices."""
    print("Fetching additional volatile stocks...")
    additional = [
        # Leveraged ETFs (high volatility)
        "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA", "LABU", "LABD",
        "FNGU", "FNGD", "TECL", "TECS", "FAS", "FAZ", "NUGT", "DUST",
        "JNUG", "JDST", "ERX", "ERY", "YINN", "YANG", "BOIL", "KOLD",
        # Sector ETFs
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLB", "XLY", "XLP", "XLU", "XLRE",
        "QQQ", "SPY", "IWM", "DIA", "VTI", "VOO", "VT",
        # Commodity ETFs
        "GLD", "SLV", "USO", "UNG", "WEAT", "CORN", "DBA",
        # Bond ETFs (for diversification signals)
        "TLT", "IEF", "SHY", "LQD", "HYG", "JNK",
        # International
        "EEM", "EFA", "FXI", "EWJ", "EWZ", "RSX",
        # VIX related
        "VXX", "UVXY", "SVXY", "VIXY",
        # Popular meme/retail stocks
        "GME", "AMC", "BBBY", "KOSS", "EXPR", "NAKD", "CLOV", "WISH", "SDC",
        # Cannabis
        "TLRY", "CGC", "ACB", "SNDL", "HEXO", "OGI", "CRON", "VFF", "GRWG",
        # Chinese ADRs
        "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI", "BILI", "TME",
        "IQ", "VIPS", "TAL", "EDU", "DIDI", "FUTU", "TIGR",
        # High short interest
        "CVNA", "UPST", "AFRM", "PTON", "BYND", "NKLA", "RIDE",
        # Recent IPOs 2023-2024
        "ARM", "CART", "KVYO", "BIRK", "VFS", "ONON",
        # REITs
        "O", "VICI", "AMT", "PLD", "SPG", "EQIX", "DLR", "PSA", "AVB",
        # Dividend stocks (for different strategy signals)
        "T", "VZ", "IBM", "MMM", "WBA", "DOW",
        # Utilities (often inverse to market)
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
        # Defense
        "LMT", "RTX", "NOC", "GD", "BA", "HII",
        # Banks
        "BAC", "WFC", "C", "USB", "PNC", "TFC", "SCHW",
    ]
    print(f"  Found {len(additional)} additional stocks")
    return additional

def fetch_midcap_stocks():
    """Mid-cap stocks with good volatility potential."""
    print("Fetching mid-cap stocks...")
    midcaps = [
        # Mid-cap tech
        "CFLT", "GTLB", "DOCN", "ESTC", "BRZE", "BILL", "HUBS", "VEEV", "TWLO", "OKTA",
        "ZI", "COUP", "APPN", "FSLY", "NCNO", "ASAN", "FROG", "SUMO", "CLDR", "PD",
        "TDC", "EVBG", "NEWR", "BSY", "JAMF", "RPD", "PRFT", "DBX", "BOX", "ZEN",
        # Mid-cap healthcare
        "EXAS", "NTRA", "HZNP", "RARE", "SGEN", "NBIX", "IONS", "ALNY", "BMRN", "SAREPTA",
        "UTHR", "MEDP", "KRYS", "FOLD", "ARDX", "PRTA", "RCKT", "CRNX", "APLS", "IRTC",
        "TGTX", "RGEN", "LGND", "CDNA", "XENE", "ABCL", "TWST", "PACB", "ARWR", "DRNA",
        # Mid-cap financials
        "LPLA", "IBKR", "HOOD", "COIN", "MKTX", "VIRT", "CBOE", "NDAQ", "LAZ", "EVR",
        "PJT", "MC", "GHL", "HLI", "PIPR", "JEF", "SF", "ALLY", "SYF", "CACC",
        # Mid-cap consumer
        "DECK", "CROX", "SHAK", "BROS", "WING", "TXRH", "PLAY", "CAKE", "EAT", "BLMN",
        "JACK", "DINE", "BJRI", "RUTH", "ARCO", "LOCO", "FRGI", "PZZA", "WEN", "NDLS",
        "SBUX", "DPZ", "QSR", "MCD", "CMG", "CAVA", "CARG", "CVNA", "AN", "ABG",
        # Mid-cap industrials
        "AXON", "TTC", "SITE", "WSC", "PRIM", "MYRG", "DY", "MTZ", "STRL", "ROAD",
        "GVA", "PRIM", "AGX", "TPC", "ROCK", "CW", "DCI", "NPO", "FELE", "RBC",
        # Mid-cap energy
        "PR", "SM", "MTDR", "PDCE", "CNX", "RRC", "AR", "SWN", "CHK", "GPOR",
        "CIVI", "NOG", "MGY", "CPE", "ESTE", "REPX", "VTLE", "CHRD", "OVV", "CLR",
        # Biotech/Pharma mid-caps
        "RXRX", "FATE", "BEAM", "CRSP", "EDIT", "NTLA", "VERV", "BLUE", "SRPT", "EXEL",
        "HALO", "IMVT", "SWTX", "BPMC", "ARVN", "KYMR", "ACLX", "OLPX", "RLAY", "PTGX",
        # Software mid-caps
        "MDB", "SNOW", "DDOG", "NET", "CRWD", "ZS", "S", "CFLT", "ESTC", "PLTR",
        "PATH", "AI", "BBAI", "UPST", "AFRM", "SOFI", "LC", "OPEN", "RDFN", "REAL",
        # Semiconductors mid-caps
        "WOLF", "CRUS", "SLAB", "DIOD", "POWI", "VSH", "ACLS", "FORM", "KLIC", "ONTO",
        "MKSI", "NOVT", "AEHR", "RMBS", "AMKR", "UCTT", "AMBA", "SITM", "CEVA", "LSCC",
        # Clean energy mid-caps
        "RUN", "NOVA", "ARRY", "SHLS", "STEM", "FLNC", "ENVX", "QS", "MVST", "DCFC",
        "EVGO", "CHPT", "BLNK", "LEV", "PTRA", "GWH", "OUST", "LAZR", "VLDR", "AEVA",
    ]
    print(f"  Found {len(midcaps)} mid-cap stocks")
    return midcaps

def fetch_international_adrs():
    """International ADRs traded on US exchanges."""
    print("Fetching international ADRs...")
    adrs = [
        # European ADRs
        "ASML", "NVO", "AZN", "SHEL", "TTE", "BP", "UL", "GSK", "DEO", "BUD",
        "BTI", "RIO", "BHP", "VALE", "ABB", "SAP", "SNY", "NVS", "CS", "UBS",
        "ING", "DB", "HSBC", "LYG", "BCS", "SAN", "BBVA", "STM", "ERIC", "NOK",
        "SPOT", "SE", "GRAB", "GLBE", "MNDY", "WIX", "FVRR", "GLOB", "DOCS",
        # Asian ADRs
        "TSM", "SONY", "TM", "HMC", "NIO", "XPEV", "LI", "BABA", "JD", "PDD",
        "BIDU", "NTES", "TCEHY", "BABA", "BILI", "IQ", "TME", "DIDI", "TAL", "EDU",
        "ZTO", "YUMC", "HTHT", "QFIN", "LX", "VNET", "KC", "ATHM", "WB", "YMM",
        "BZUN", "HUYA", "DOYU", "TIGR", "FUTU", "FINV", "LKNCY", "NIU", "XP", "STNE",
        # Latin American ADRs
        "NU", "MELI", "GLOB", "BSBR", "ITUB", "BBD", "PBR", "ABEV", "SBS", "CIG",
        "ERJ", "TIMB", "PAGS", "GGB", "SID", "CBD", "AFYA", "VTRU", "ARCO",
        # Indian ADRs
        "INFY", "WIT", "HDB", "IBN", "RDY", "TTM", "SIFY", "WNS", "YTRA", "MFG",
        # Other
        "TEVA", "ESLT", "CYBR", "CHKP", "NICE", "SEDG", "FROG", "MNDY", "RKLB",
    ]
    print(f"  Found {len(adrs)} international ADRs")
    return adrs

def load_existing_universe():
    """Load existing universe.yaml."""
    with open("universe.yaml", "r") as f:
        return yaml.safe_load(f)

def get_existing_symbols(universe):
    """Get all symbols already in universe."""
    existing = set()
    scanner = universe.get("scanner_universe", {})
    for category, symbols in scanner.items():
        if isinstance(symbols, list):
            existing.update(symbols)

    # Also include proven_symbols and blacklist
    existing.update(universe.get("proven_symbols", []))
    existing.update(universe.get("blacklist", []))

    return existing

def expand_universe(target_count=1000):
    """Expand universe to target count of stocks."""
    # Load existing
    universe = load_existing_universe()
    existing = get_existing_symbols(universe)
    print(f"Existing symbols: {len(existing)}")

    # Fetch from all sources
    new_stocks = {
        "sp500_expanded": fetch_sp500(),
        "nasdaq100_expanded": fetch_nasdaq100(),
        "small_caps": fetch_russell2000_sample(),
        "etfs_and_popular": fetch_additional_tickers(),
        "midcap_volatile": fetch_midcap_stocks(),
        "international_adrs": fetch_international_adrs(),
    }

    # Add to scanner_universe, avoiding duplicates
    scanner = universe.get("scanner_universe", {})
    all_added = set()

    for category, symbols in new_stocks.items():
        new_symbols = []
        for sym in symbols:
            if sym not in existing and sym not in all_added:
                new_symbols.append(sym)
                all_added.add(sym)

        if new_symbols:
            if category not in scanner:
                scanner[category] = []
            scanner[category].extend(new_symbols)
            print(f"Added {len(new_symbols)} new to {category}")

    universe["scanner_universe"] = scanner

    # Save updated universe
    with open("universe.yaml", "w") as f:
        yaml.dump(universe, f, default_flow_style=False, sort_keys=False)

    # Count total
    total = sum(len(v) for v in scanner.values() if isinstance(v, list))
    print(f"\nTotal symbols in scanner_universe: {total}")

    return total

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Just count, don't modify")
    args = parser.parse_args()

    if args.dry_run:
        print("\n=== DRY RUN - Counting available stocks ===\n")
        sources = [
            fetch_sp500(),
            fetch_nasdaq100(),
            fetch_russell2000_sample(),
            fetch_additional_tickers(),
            fetch_midcap_stocks(),
            fetch_international_adrs(),
        ]

        universe = load_existing_universe()
        existing = get_existing_symbols(universe)

        all_new = set()
        for lst in sources:
            all_new.update(lst)

        unique_new = all_new - existing
        print(f"\nExisting: {len(existing)}")
        print(f"Total from all sources: {len(all_new)}")
        print(f"New unique symbols to add: {len(unique_new)}")
        print(f"Final total would be: {len(existing) + len(unique_new)}")
    else:
        expand_universe()
