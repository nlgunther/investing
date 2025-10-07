import os, re, numpy as np,matplotlib.pyplot as plt, pandas as pd,requests
from collections import defaultdict, namedtuple
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup as BSP
import lxml
from io import StringIO

from functools import partial

from collections import namedtuple

from tqdm.notebook import tqdm

from dataclasses import dataclass

from itertools import product

from adjustText import adjust_text

from typing import Any, Generator,NamedTuple

import sec_edgar_downloader
sec_edgar_downloader.__version__
# see if namedtuple has "in"

class RexGroup(NamedTuple): # for default values
    rex: str
    groupnum: int  = 0 

ReportAttributes = namedtuple('ReportAttributes','date type name cik'.split())

Location = namedtuple('Location','path attributes'.split())

class Reports(object):

    @staticmethod
    def find_file_leaves(dr) -> Generator[tuple[Any, list[Any], list[Any]], Any, None]:
        '''Given a directory, yield the leaves of the file tree
        skipping directories without files'''
        for root,drs,fns in os.walk(dr):
            if fns:
                yield root,drs,fns

    @staticmethod
    def search_report_text_for_attributes(rexgroup,text):
        '''Given a RexGroup and a text, return the attribute
        found by searching the text with the rex in the RexGroup.
        This method is a wrapper around re.search()'''
        m = re.search(rexgroup.rex,text,re.MULTILINE)
        if m:
            try:
                return m[rexgroup.groupnum]
            except TypeError as e:
                msg = f'Problems finding {rexgroup.rex} with group {rexgroup.groupnum}'
                raise TypeError(msg) from e
    
    @classmethod
    def get_report_attributes_from_rexgroups(kls, rexgroups, path,num=-1):
        '''Given a list of RexGroups and a file path to a report, return 
         return a list of report attributes by reading the first #num characters
        of the file and searching for each RegGroup rex'''
        with open(path) as f:
            header = f.read(num)
            attributes = []
            for rexgroup in rexgroups:
                try:
                    attribute = kls.search_report_text_for_attributes(rexgroup,header)
                    if attribute: attributes.append(attribute)
                except TypeError as e:
                    msg = f"Problems with {rexgroup} and {path}"
                    raise TypeError(msg) from e
        return attributes

    @classmethod
    def get_locations(kls,reports: list(tuple[Any]),
                     attributerexs: list[RexGroup]) -> list[Location]:
        '''Given a list of file leaves (root,dirs,files) from find_file_leaves
        and a list of RexGroups specifying report attributes, return a list of
        Locations with the attributes filled in.
        Get the locations of the reports with their attributes
        by reading the first 2000 characters of each file
        '''

        test = partial(kls.get_locations_from_attribute_rexs, attributerexs)
        return [loc for root,_,fns in reports
                    for fn in fns
                        if (loc := test(os.path.join(root,fn)))]
    
    @classmethod
    def get_locations_from_attribute_rexs(kls,rexgroups, path):
        attributes = kls.get_report_attributes_from_rexgroups(rexgroups,path,num=2000)
        if attributes:
            try: 
                return Location(path,ReportAttributes(*attributes))
            except TypeError as e:
                print(f"Problem with {rexgroup} and {path}: {e}")
    
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

class ReportParser(object):

    EXCERPT_LENGTH = 2000
    delimitertags_rex =  r'(<XML>)|(</XML>)'
    
    @staticmethod
    def read(path):
        with open(path) as f:
            text = f.read()
        return text

    @staticmethod
    def find_xml(text):
        xmlintext = list(re.finditer(ReportParser.delimitertags_rex,text))
        try: 
            assert (numtags := len(xmlintext)) % 2 == 0
        except AssertionError as e:
            msg = f"There are an odd number {numtags} of occurrences matching {ReportParser.delimitertags_rex} in the text"
            raise AssertionError(msg) from e
        if numtags > 2:
            print(f"There are {numtags/2} pairs of tags matching {ReportParser.delimitertags_rex} in the text;taking the last")
            opentags = [xmlintext[i] for i in range(0,numtags,2)]
        return [xmlintext[i].span()[j] for i,j in ((-2,0),(-1,1))]
        '''
        get final pair - a crude hack.
        should find the biggest span, or check if a key subtag, such as <informationTable> for
        hedge funds, exists
        '''
        # TODO match even and odd indices and take largest included text; make into its own object

    @classmethod
    def set_delimitertags_rex(kls,tags):
        kls.delimitertags_rex = tags
    
    def __init__(self,location: Location):
        self.location = location
        self.soup = None

    def get_soup(self,return_parser=False):
        text = self.read(self.location.path) # loose coupling with location object
        self.start,self.stop = self.find_xml(text)
        # STORE limited excerpts of the text
        self.header, self.fund_info = text[:self.start],text[self.start:self.start + self.EXCERPT_LENGTH]
        soup = BSP(text[self.start:self.stop],'lxml-xml')
        if return_parser: # wrap Parser around soup to include context data
            self.soup = soup
            return self
        else:
            return soup


class ReportMaker(object):

    def __init__(self,positionrex,rowmaker):
        self.positionrex=positionrex
        self.rowmaker = rowmaker

    def make_dataframe_from_parser(self,soup):
        try:
            nodes = soup.find_all(self.positionrex)
            return pd.DataFrame([self.rowmaker(node) for
                             node in nodes])
        except TypeError as e:
            msg = f'Soup from {parser.location} has problems'
            raise TypeError(msg)
        
# get_invesco_report_attributes = partial(search_report_text_for_attributes,secreportattributes)


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
    return maker.make_dataframe_from_parser(ReportParser(loc).get_soup())

def parsers2df(reportmaker,parsers):
    dfs = [reportmaker.make_dataframe_from_parser(parser) for parser in parsers]
    return pd.concat(dfs,keys = [parser.location.attributes.date for parser in parsers])

def no_repeats(s):
    cs = []
    no_reps = ''
    for e in s:
        if e in cs: continue
        no_reps+=e
        cs.append(e)
    return no_reps    

def remove_vowels_down(s_):
    s = re.sub(r'inc$|co$|corp$|ltd$|lp$|llc$','',s_.lower())
    s = re.sub('(?i)[aeiou]','',s)
    s = re.sub(r'\s+','_',s)
    s = re.sub(r'\W','',s).strip()
    s = re.sub(r'([a-zA-Z])\1+', r'\1',s)
    s = re.sub(r'_+$','',s)
    return s#no_repeats(s)

clean_df = lambda df: pd.concat([df.query('cusip != "N/A"'),
                                 df.query('cusip == "N/A" and nameOfIssuer != "N/A"')\
                                    .assign(cusip = df.nameOfIssuer.apply(remove_vowels_down))])\
                                        .drop_duplicates()

config_df = lambda df: df.query("nameOfIssuer != 'N/A' or cusip != 'N/A'")\
    .dropna().sort_values('value')

# Exposure for Hedge Funds with multiple long/short positions in same cusip
short_str = r'(?i)(\d\.?)+\s*x'
short_rex = re.compile(r'(?i)(\d\.?)+\s*x')
is_short = lambda s: short_rex.search(s)
column_is_short = lambda col: r'%s.str.contains(%s)' % (col,short_rex)
qShort = r'sign == -1 and putCall=="Put" and %s.str.contains(r"(?i)(\d\.?)+\s*x")' % 'titleOfClass'

class Exposure(object):
    
    @staticmethod
    def get_exposure(df):
        numdf = df.assign(num = df.titleOfClass.apply(lambda r:float(m[1]) if (m := is_short(r)) else 1))
        signdf = numdf.assign(sign = numdf.apply(lambda r: (-1 if (short_rex.search(r.titleOfClass) and re.search('(?i)short',r.titleOfClass)) else 1) *  (-1 if (r.putCall and r.putCall.lower() == 'put') else 1),axis=1))
        # signdf.query('sign == -1 and putCall=="Put" and titleOfClass.str.contains(r"(?i)(\d\.?)+\s*x") ')
        # signdf.query(qShort)
        return signdf.assign(mod_value = signdf.value * signdf.num*signdf.sign)

    def __init__(self,df):
        self.df = df
        self.df = self.get_exposure(df.copy())

    def aggregate(self):
        self.df = self.df.groupby('cusip')['mod_value'.split()].sum()

    def get_weights(self):
        self.df = self.df.assign(weights = self.df.mod_value/self.df.mod_value.abs().sum())


def make_comparison_plot(df, x, y, n = 15,fundname = 'Fund', figsize=None,yeqx=False):
    figsize = (6,6) if figsize is None else figsize
    fig,ax = plt.subplots(1,1,figsize= figsize)
    df.iloc[:-n].plot.scatter(x=x,y=y, ax = ax)
    lims = (-0.003,0.06)
    df.iloc[-n:].plot.scatter(x=x,y=y,c='red',ax=ax,grid=True, title=f'Most Recent {fundname} Holdings vs Market')
    if yeqx: ax.plot(lims,lims,c='red')
    # ax.set_aspect('equal')
    annotations = [ax.annotate(text,(x,y),size=8) for text,x,y in df.iloc[-n:][('nameOfIssuer %s %s' % (x,y)).split()].values]
    adjust_text(annotations)
    return fig,ax