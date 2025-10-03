import os, re, numpy as np,matplotlib.pyplot as plt, pandas as pd,requests
from collections import defaultdict, namedtuple
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup as BSP
from io import StringIO

from functools import partial

from collections import namedtuple

from tqdm.notebook import tqdm

from dataclasses import dataclass

from itertools import product

from adjustText import adjust_text

from typing import Any, Generator

import sec_edgar_downloader
sec_edgar_downloader.__version__
# see if namedtuple has "in"

RexGroup = namedtuple('RexGroup', 'rex groupnum'.split())

ReportAttributes = namedtuple('ReportAttributes','date type name cik'.split())

Location = namedtuple('Location','path attributes'.split())

class Reports(object):

    @staticmethod
    def find_file_leaves(dr) -> Generator[tuple[Any, list[Any], list[Any]], Any, None]:
        # a generator to find the files under 
        #  the top directory skipping
        #  directories without files
        for root,drs,fns in os.walk(dr):
            if fns:
                yield root,drs,fns

    @staticmethod
    def get_report_attributes(rexgroup,text):
        # search the text for identifying metadata, like date
        try:
            return re.search(rexgroup.rex,text,re.MULTILINE)[rexgroup.groupnum]
        except TypeError as e:
            msg = f'Problems finding {rexgroup}'
            raise TypeError(msg) from e

    @classmethod
    def get_locations(kls,reports: list(tuple[Any]),
                     attributerexs: list[RexGroup]) -> list[Location]:
        locs = []
        for root,_,fns in reports:
            for fn in fns:
                path = os.path.join(root,fn)
                with open(path) as f:
                    header = f.read(2000)
                    attributes = []
                    for rexgroup in attributerexs:
                        try:
                            attributes.append(kls.get_report_attributes(rexgroup,header))
                        except TypeError as e:
                            msg = f"Problems with {rexgroup} and {path}"
                            raise TypeError(msg) from e
                    try: locs.append(Location(path,ReportAttributes(*attributes)))
                    except TypeError as e:
                        print(f"Problem with {rexgroup} and {path}")
        return locs
    
    @staticmethod
    def get_location_dict(locs:list[Location],
                          fundnamedict:dict,
                          loc_test = lambda name,loc: name in loc.attributes.name):
        flocs = {}
        for shortname,name in fundnamedict.items():
            flocs[shortname] = [loc for loc in locs if loc_test(name,loc)]
        return flocs

    def __init__(self,
                 reportdr='Downloads/recent',
                 attributes = None,
                fundnamedict=None,
                loc_test = None):
        self.reportdr = reportdr
        self.reports = list(self.find_file_leaves(reportdr))
        self.attributes = attributes
        self.fundnamedict = fundnamedict
        self.locs = self.get_locations(self.reports,attributes)
        self.locd = self.get_location_dict(self.locs,fundnamedict,loc_test)

def get_loc(test,locs, idx = None):
    sel_locs = [loc for loc in locs if test(loc)]
    return sel_locs if idx is None else sel_locs[idx]

class ReportMaker(object):

    def __init__(self,positionrex,rowmaker):
        self.positionrex=positionrex
        self.rowmaker = rowmaker

    def make_dataframe_from_parser(self,parser):
        try:
            nodes = parser.soup.find_all(self.positionrex)
            return pd.DataFrame([self.rowmaker(node) for
                             node in nodes])
        except TypeError as e:
            msg = f'Soup from {parser.location} has problems'
            raise TypeError(msg)
        
# get_invesco_report_attributes = partial(get_report_attributes,secreportattributes)


hfund_rowmaker = lambda node: dict(
            cusip = node.cusip.string,
            nameOfIssuer = node.nameOfIssuer.string,
            titleOfClass = node.titleOfClass.string,
            value = float(node.value.string),
            investmentDiscretion = node.investmentDiscretion.string,
            putCall = putCall.string if (putCall := node.putCall) else None,
            otherManager = otherManager.string if (otherManager := node.otherManager) else None,
            shares_principal = int((sorp := node.shrsOrPrnAmt).find("sshPrnamt").string),
            shares_type = sorp.find("sshPrnamtType").string,
            sole_vote = int((vauth := node.votingAuthority).find('Sole').string),
            shared_vote = int(vauth.find("Shared").string),
            no_vote = int(vauth.find("None").string)
)

mfund_rowmaker = lambda node: dict(
            cusip = node.cusip.string,
            nameOfIssuer = node.find("name").string,
            value = float(val.string) if (val := node.valUSD) else None,
            pctValue = float(pcval.string) if (pcval := node.pctVal) else None
)

mfundreportmaker = ReportMaker('invstOrSec',mfund_rowmaker)
hfundreportmaker = ReportMaker('infoTable',hfund_rowmaker)

#convenience method
def loc2df(loc: Location,maker) -> pd.DataFrame:
    return maker.make_dataframe_from_parser(ReportParser(loc).get_soup(True))

def parsers2df(reportmaker,parsers):
    dfs = [reportmaker.make_dataframe_from_parser(parser) for parser in parsers]
    return pd.concat(dfs,keys = [parser.location.attributes.date for parser in parsers])

clean_df = lambda df: pd.concat([df.query('cusip != "N/A"'),
                                 df.query('cusip == "N/A" and nameOfIssuer != "N/A"')\
                                    .assign(cusip = 'not_set')])\
                                        .drop_duplicates()

config_df = lambda df: df.query("nameOfIssuer != 'N/A' or cusip != 'N/A'")\
    .dropna().sort_values('value')