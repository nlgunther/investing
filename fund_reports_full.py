#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

Report = namedtuple(
    'Report',
    'name type cik'.split()
)


class Download(object):

    def __init__(self,form: str,limit: int = 5):
        self.dl = Downloader("Nick Gunther", "nlgbus2631@gmail.com",r"C:\Users\nlgun\Downloads\recent")
        self.limit = limit
        print(f"Saving to {self.dl.download_folder} with default limit {limit}")

    def download(self,cikdict,limit=None):
        limit = self.limit if limit is None else limit
        print(f"Saving to {self.dl.download_folder} with default limit {limit}")

        for cik,nameform in cikdict.items():
            name,form =nameform
            print(f"Downloading registrant {name}, form {form} and cik {cik}")
            self.dl.get(form, cik,limit=limit)

# Example of cikdict
ciks = "0001423053 0001037389 0001350694 0000036405".split()
# ciks = "S000002848".split()
names = "Citadel Renaissance Bridgewater Vanguard_Total_Market".split()
types = ['13F-HR']*3 + ['NPORT-P']
cikdict = dict(zip(ciks,zip(names,types)))
tuple_data = zip(names,types,ciks)

class pdict(dict):

    def __init__(self,tuple_data,numkeys=1):
        # tuple_data is a zip of names, types and ciks
        # the first numkeys of each tuple will be mapped to the
        # the last entry, which will be mapped to the entire tuple
        for tpl in tuple_data:
            self[tpl[-1]] = Report(*tpl) # store the full named tuple
            # at the cik then map the other designated attributes
            # to the cik
            for e in tpl[:numkeys]: self[e] = tpl[-1]
        self.ks = self.keys()

    def pget(self,key,default=None):
        # p[ath]get: finds the first time a value
        # isn't itself a key and returns that value
        if self[key] not in self.ks: return  self[key]
        else: return self.pget(self[key]) # follow chains of keys



# In[7]:

def find_file_leaves(dr):
    # a generator to find the files under 
    #  the top directory skipping
    #  directories without files
    for root,drs,fns in os.walk(dr):
        if fns:
            yield root,drs,fns

def make_paths(basefolder,localfolder,d,k):
    tpl = report_dict.pget(k)
    return os.path.join(basefolder,localfolder,tpl.cik,tpl.type)
# fileiter = iter(find_file_leaves(make_secpath(report_dict['Citadel'])))
make_secpath = partial(make_paths,dl.download_folder,'sec-edgar-filings',report_dict)


def file2soup(path):
    with open(path) as f:
        soup = BSP( f.read(),'xml')
    return soup

kdatepathdict = {}
DEBUG=False
from functools import reduce
import copy
for k, souppaths in souppathdict.items():
    if DEBUG: print(k)
    for souppath in souppaths:
        if re.search('(?i)_short',souppath): continue
        if DEBUG: print(souppath)
        with open(souppath) as f:
            initial_text = f.read(2000)
        if (m:=re.search(r'^FILED.*DATE:\s+(\d{8})$',initial_text,re.MULTILINE)):
            if DEBUG: print(m[1],souppath)
            tempd = {m[1]:souppath}
            if k in cikdatepathdict:
                cikdatepathdict[k].update(tempd)
            else: cikdatepathdict[k]=tempd


# In[47]:


cikdatepathdict = {}
DEBUG=False
from functools import reduce
import copy
for k, souppaths in souppathdict.items():
    if DEBUG: print(k)
    for souppath in souppaths:
        if re.search('(?i)_short',souppath): continue
        if DEBUG: print(souppath)
        with open(souppath) as f:
            initial_text = f.read(2000)
        if (m:=re.search(r'^FILED.*DATE:\s+(\d{6})\d*$',initial_text,re.MULTILINE)):
            if DEBUG: print(m[1],souppath)
            tempd = {m[1]:souppath}
            if k in cikdatepathdict:
                cikdatepathdict[k].update(tempd)
            else: cikdatepathdict[k]=tempd


# In[16]:


cikdatepathdict


# In[17]:


datesoupgen = ((cik,((date,file2soup(path)) for date,path in datepathdict.items())) for cik, datepathdict in cikdatepathdict.items())


# In[20]:


cik,soup = next(iter(datesoupgen))


# In[21]:


report_dict[cik]


# In[44]:


next(iter(citdict.values()))


# ## Report function

# In[18]:


def report2df(soup):

    data = [dict(
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
    for node in soup.find_all('infoTable')]
    return pd.DataFrame(data)


# In[22]:


citdict= dict(soup)


# In[23]:


# citdict_ = citdict
citdict = {k[:6]:report2df(path) for k,path in citdict.items()}


# In[24]:


citdict.keys()#['202508']


# If title has, eg, 2x short, multiple value by coefficient and sign with 

# In[25]:


def number_times_value(row):
    number = re.search('\dx')

def short(row):
    # action dictinoary
    actd = {
        'short_title': number_times_value
        'call': get_value
        'put': minus_value
    }

    title_test(row.titleOfClass)


# In[ ]:


shorten_keys = lambda d,n: {k[:n]:v for k,v in d.items()}


# In[73]:


# citdict = shorten_keys(citdict,6)


# In[65]:


# testdf = citdict['20250814']
testdf = citdict['202505']


# In[27]:


num = testdf[testdf.titleOfClass.apply(lambda r: bool(re.search(r'(?i)\d\s*x',r)))].value.sum()
numtot = testdf.value.sum()
f"{num:,.2f}             {numtot:,.2f}"


# In[77]:


testdf[testdf.titleOfClass.apply(lambda r: bool(re.search(r'(?i)(\d\.?)+\s*x short',r)))]


# In[29]:


short_str = r'(?i)(\d\.?)+\s*x'
short_rex = re.compile(r'(?i)(\d\.?)+\s*x')
is_short = lambda s: short_rex.search(s)
column_is_short = lambda col: '%s.str.contains(%s)' % (col,short_rex)
qShort = 'sign == -1 and putCall=="Put" and %s.str.contains(r"(?i)(\d\.?)+\s*x")' % 'titleOfClass'


# In[76]:


numdf = testdf.assign(num = testdf.titleOfClass.apply(lambda r:float(m[1]) if (m := is_short(r)) else 1))
testdf.query('titleOfClass.str.contains(r"x short")')


# In[31]:


class Exposure(object):

    @staticmethod
    def get_exposure(df):
        numdf = df.assign(num = df.titleOfClass.apply(lambda r:float(m[1]) if (m := is_short(r)) else 1))
        signdf = numdf.assign(sign = numdf.apply(lambda r: (-1 if (short_rex.search(r.titleOfClass) and re.search('(?i)short',r.titleOfClass)) else 1) *  (-1 if (r.putCall and r.putCall.lower() == 'put') else 1),axis=1))
        # signdf.query('sign == -1 and putCall=="Put" and titleOfClass.str.contains(r"(?i)(\d\.?)+\s*x") ')
        # signdf.query(qShort)
        return signdf.assign(mod_value = signdf.value * signdf.num*signdf.sign)

    def __init__(self,df):
        self.df = self.get_exposure(df)

    def aggregate(self):
        self.df = self.df.groupby('cusip')['mod_value'.split()].sum()

    def get_weights(self):
        self.df = self.df.assign(weights = self.df.mod_value/self.df.mod_value.abs().sum())


# In[35]:


exp = Exposure(citdict['202505'])
exp.aggregate()
exp.get_weights()
exp.df.weights.abs().sum()


# In[36]:


exp5 = exp


# In[39]:


exp8.df


# In[51]:


testdf = pd.DataFrame({202508: exp8.df.weights,202505:exp5.df.weights})
(exp8.df.mod_value.sum()- (exp5exp := exp5.df.mod_value.sum()))/exp5exp


# In[63]:


# Leverage
tot = lambda df,s,k: df.query(s)[k].sum()
leverage = lambda df: df.mod_value.sum()/tot(df,'mod_value>0','mod_value')
propsize = lambda df1,df0: (df1.mod_value.sum() - df0.mod_value.sum())/df0.mod_value.sum()
# (exp8.df.query('mod_value > 0').mod_value.sum()), tot(exp8.df,'mod_value>0','mod_value'),tot(exp8.df,'mod_value<0','mod_value')
[leverage(df) for df in (exp8.df,exp5.df)],propsize(exp8.df,exp5.df)


# In[48]:


testdf = testdf.dropna().sort_values(202508).join(citdict['202508'].set_index('cusip')['nameOfIssuer mod_value itleOfClass'.split()].drop_duplicates())


# In[44]:


testdf_graph = testdf.dropna().assign(delta = testdf[202508] - testdf[202505])
testdf_graph = testdf_graph.assign(absdelta = testdf_graph.delta.abs()).reset_index()
testdf_graph.sort_values('absdelta',inplace=True)
testdf_graph


# In[52]:


from adjustText import adjust_text
ax = testdf_graph.dropna().iloc[:-10,:].plot.scatter(x=202505,y=202508, title='Changes in Citadel Holdings')
lims = (-0.06,0.02)
ax.plot(lims,lims,c='red')
(movers := testdf_graph.dropna().iloc[-10:,:]).plot.scatter(x=202505,y=202508,c='red',ax=ax,grid=True)

texts = [ax.annotate(text,(x,y)) for y,x,text in movers[[202508, 202505,'nameOfIssuer']].values]
adjust_text(texts)
ax.set_aspect('equal')
# for y,x,text in movers[[202508, 202505,'nameOfIssuer']].values:
#     ax.annotate(text,(x,y))


# In[38]:


get_exposure(testdf)


# In[43]:


(moddf := Out[38]).query(qShort)


# In[50]:


moddf[moddf.nameOfIssuer.str.contains('AB ACTIVE ETFS')]


# In[51]:


moddf.groupby('cusip nameOfIssuer titleOfClass'.split()).mod_value.sum()


# In[238]:


signdf = numdf.assign(sign = numdf.apply(lambda r: (-1 if (short_rex.search(r.titleOfClass) and re.search('(?i)short',r.titleOfClass)) else 1) *  (-1 if (r.putCall and r.putCall.lower() == 'put') else 1),axis=1))
# signdf.query('sign == -1 and putCall=="Put" and titleOfClass.str.contains(r"(?i)(\d\.?)+\s*x") ')
signdf.query(qShort)


# In[191]:


moddf = signdf.assign(mod_value = signdf.value * signdf.num*signdf.sign)


# In[40]:


Out[38].query('titleOfClass.str.contains(r"(?i)(\d\.?)+\s*x")')


# In[41]:


((grouped := Out[38].groupby("cusip").mod_value.sum())/grouped.sum()).hist()


# In[205]:


((grouped := moddf.groupby("cusip").mod_value.sum())/grouped.sum()).hist()


# In[214]:


formatdlr = lambda num: f"${num:,.0f}"


# In[216]:


a,b = grouped[grouped>0].sum(),grouped[grouped<0].sum()
[formatdlr(n) for n in (a,b)]


# In[67]:


sorted(citdict.keys())


# In[134]:


df = citdict['202508']['cusip value putCall titleOfClass'.split()]#.dropna()
df = df.assign(short = df.apply(lambda row:row.putCall if row.putCall else "short" if re.search(r'(?i)x.*short',row.titleOfClass) else "long",axis=1))


# In[135]:


df[df.short == 'short'].head(50)


# # CUSIP to Ticker

# In[78]:


converter = pd.read_csv(r'G:\My Drive\UCBerkeley\CountingFactors\Stock_Data\CUSIP.csv')


# In[82]:


converter.set_index('cusip',inplace=True)


# In[ ]:





# In[109]:


converter


# In[83]:


exp8.df.join(converter)


# In[115]:


sectors = pd.read_excel(r'G:\My Drive\UCBerkeley\CountingFactors\hedge_funds\tickers2sectors.xlsx',skiprows=[0,2])
sectors = sectors.dropna(how='all',axis=1).assign(symbol = sectors['Symbol '].apply(lambda s: s.strip())).set_index('symbol').drop('Symbol ',axis=1)
sectors


# In[122]:


exp8gic = Out[83].join(sectors.dropna(how='all',axis=1),on='symbol',how='inner').reset_index()#.groupby('GICS Sector','GICS Sub-Industry')['mod_value weights'.split()].sum()


# In[126]:


exp8gic.groupby(['GICS Sector','GICS Sub-Industry'])['mod_value weights'.split()].sum()


# In[128]:


exp5.df.join(converter).join(sectors.dropna(how='all',axis=1),on='symbol',how='inner').reset_index().groupby(['GICS Sector','GICS Sub-Industry'])['mod_value weights'.split()].sum()


# In[ ]:


(sectordelta := pd.DataFrame({202508: Out[126].weights,202505: Out[128].weights})).plot.scatter(y=202508,x=202505,grid=True)


# In[151]:


def cols2str(df):
    df.columns = [str(c) for c in df.columns]
    return df


# In[158]:


sectordelta = cols2str(sectordelta)
sectordelta.reset_index()['GICS Sub-Industry;202508;202505'.split(';')]


# In[169]:


fig,ax = plt.subplots(1,1,figsize=(8,8))
sectordelta.iloc[:-10,:].plot.scatter(y='202508',x='202505',ax=ax)
lims = (-0.010,0.05)
ax.plot(lims,lims,c='red')
sectordelta.iloc[-10:,:].plot.scatter(y='202508',x='202505',grid=True,c='red',ax=ax,title='Citadel Weight Change by Sector, Last Two Quarters')

texts = [ax.annotate(text,(x,y)) for text,y,x in sectordelta.reset_index()['GICS Sub-Industry;202508;202505'.split(';')].values[-10:]]
adjust_text(texts)
ax.set_aspect('equal')    


# In[133]:


sectordelta = sectordelta.assign(delta= sectordelta[202508] - sectordelta[202505])


# In[140]:


sectordelta = sectordelta.assign(absdelta = sectordelta.delta.abs())
sectordelta = sectordelta.sort_values('absdelta')
sectordelta


# In[141]:


sectordelta.index.get_level_values(1)


# In[74]:


citdict['202508']['cusip value putCall'.split()].groupby('cusip putCall'.split()).sum().xs('000360206',level='cusip')

citdict['202508']['cusip value putCall'.split()].groupby('cusip putCall'.split()).sum().xs(None,level='putCall')


# In[ ]:





# In[ ]:


for k,v in citdict.items():
    citdict[v]


# In[237]:


import pandas as pd
import xml.etree.ElementTree as ET

def parse_13f_xml(file_path):
    """
    Parses the SEC 13F information table XML file and converts it to a Pandas DataFrame.

    Parameters:
    - file_path (str): Path to the XML file containing the <informationTable> (e.g., the INFORMATION TABLE XML).

    Returns:
    - pd.DataFrame: DataFrame with the extracted data from each <infoTable>.
    """
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Define the namespace
    ns = {'ns': 'http://www.sec.gov/edgar/document/thirteenf/informationtable'}

    # List to hold the data
    data = []

    # Iterate over each <infoTable>
    for info_table in root.findall('ns:infoTable', ns):
        # Extract fields, handling optional ones
        put_call = info_table.find('ns:putCall', ns)
        put_call_text = put_call.text if put_call is not None else None

        entry = {
            'nameOfIssuer': info_table.find('ns:nameOfIssuer', ns).text,
            'titleOfClass': info_table.find('ns:titleOfClass', ns).text,
            'cusip': info_table.find('ns:cusip', ns).text,
            'value': int(info_table.find('ns:value', ns).text),
            'sshPrnamt': int(info_table.find('ns:shrsOrPrnAmt/ns:sshPrnamt', ns).text),
            'sshPrnamtType': info_table.find('ns:shrsOrPrnAmt/ns:sshPrnamtType', ns).text,
            'investmentDiscretion': info_table.find('ns:investmentDiscretion', ns).text,
            'putCall': put_call_text
        }
        data.append(entry)

    # Create DataFrame
    df = pd.DataFrame(data)

    return df


# In[242]:


df = parse_13f_xml(souppathdict['0001423053'][0])


# In[26]:


get_type = lambda cik: report_dict[cik].type
get_type('0001423053')
table_types = dict(zip('13F-HR NPORT-P'.split(),'informationTable invstOrSecs'.split()))
table_types


# In[45]:


tabledict = {}
for cik,soup in soupdict.items():
    tabledict[cik] = soupdict[cik][0].find(table_types[get_type(cik)])


# In[103]:


if False:
    from lxml import etree
    import pandas as pd

    parser = etree.XMLParser(encoding='utf-8', #Your encoding issue.
                                  recover=True, #I assume you probably still want to recover from bad xml, it's quite nice. If not, remove.
                                  )

    with open(souppathdict['0001423053'][0]) as f:
            xml_data = f.read()

    doc = parser.parse(xml_data)


# In[229]:


testsoup = copy.copy(tabledict['0001423053'])


# In[233]:


df1 = pd.read_xml(StringIO(str(testsoup)))


# In[287]:


df = pd.DataFrame(data)


# In[302]:


df[df.titleOfClass.str.contains(r'(?i)\dx\s*short')].value.sum()/df.value.sum()


# In[313]:


df[df.putCall.fillna(' ').str.contains(r'(?i)^put$')].sort_values('value',ascending=False).value.sum()/df.value.sum()


# In[315]:


df.query("putCall.fillna(' ').str.contains(r'(?i)^put$') or titleOfClass.str.contains('(?i)short')").sort_values('value',ascending=False).value.sum()/df.value.sum()


# In[317]:


0.38/0.62


# In[ ]:


shortdf = df.query("putCall.fillna(' ').str.contains(r'(?i)^put$') or titleOfClass.str.contains('(?i)short')").sort_values('value',ascending=False)


# In[150]:


df.query("titleOfClass.str.contains('X SHORT')")


# In[138]:


df.loc[~(df.index.isin(Out[136].index))]


# In[136]:


df.query("putCall.fillna(' ').str.contains(r'(?i)^put$') or titleOfClass.str.contains('(?i)short')").sort_values('value',ascending=False)


# In[151]:


df.query('titleOfClass.str.contains(r"(?i)\dx short")').head(50)


# In[327]:


dr = r'C:\Users\nlgun\Downloads\recent\sec-edgar-filings\0001423053\13F-HR\0001104659-25-078555'
dr1 = r'C:\Users\nlgun\Downloads\recent\sec-edgar-filings\0001423053\13F-HR\0000950123-24-011767'


# In[348]:


report_dict.pget('Citadel')


# In[351]:


path_dict['0001423053']


# In[334]:


df3


# In[336]:


with open(r'%s\full-submission.txt' % dr) as f:
    text = f.read()
text[:5000]


# In[339]:


print(text[:5000])


# # HERE

# In[372]:


soupdict


# In[373]:


citdict = soupdict['0001423053']


# In[340]:


BSP(text,'xml')


# In[341]:


Out[340].find('informationTable')


# In[384]:


with open(citdict['20250214']) as f:
    # df20250814 = report2df(BSP(f.read()), 'xml').informationTable)
    # df20250515 = report2df(BSP(f.read(), 'xml').informationTable)
    # df20250214 = report2df(BSP(f.read(), 'xml').informationTable)
    df20241114 = report2df(BSP(f.read(), 'xml').informationTable)


# In[381]:


df = df20250814
df.query("putCall.fillna(' ').str.contains(r'(?i)^put$') or titleOfClass.str.contains('(?i)short')").sort_values('value',ascending=False).value.sum()/df.value.sum()


# In[382]:


df = df20250515
df.query("putCall.fillna(' ').str.contains(r'(?i)^put$') or titleOfClass.str.contains('(?i)short')").sort_values('value',ascending=False).value.sum()/df.value.sum()


# In[383]:


df = df20250214
df.query("putCall.fillna(' ').str.contains(r'(?i)^put$') or titleOfClass.str.contains('(?i)short')").sort_values('value',ascending=False).value.sum()/df.value.sum()


# In[385]:


df = df20241114
df.query("putCall.fillna(' ').str.contains(r'(?i)^put$') or titleOfClass.str.contains('(?i)short')").sort_values('value',ascending=False).value.sum()/df.value.sum()


# In[390]:


for df in (df20250814,df20250515,df20250214,df20241114): #df.set_index('cusip',inplace=True)
    df['weight']=df.value/df.value.sum()


# In[393]:


dfcomb = pd.concat([df[['weight']].groupby(level=0).sum() for df in (df20250814,df20250515,df20250214,df20241114)],axis=1,keys='202508 202505 202502 202411'.split())


# In[400]:


dfcomb = dfcomb.swaplevel(axis=1).droplevel(0,axis=1).fillna(0)


# In[417]:


ax = dfcomb.plot.scatter(*'202508 202502'.split(),c=dfcomb['202505']- dfcomb['202502'],cmap='jet',grid=True)
ax.plot((0,0.15),(0,0.15),c='red')


# In[420]:


dfcomb.sort_values('202508',ascending=False)


# In[426]:


df20250814.loc[dfcomb.sort_values('202508',ascending=False).index.intersection(df20250814.index)]['nameOfIssuer'].drop_duplicates()


# In[ ]:


pd.concat(df20250814.set_index('cusip'))


# In[342]:


with open(r'%s\full-submission.txt' % dr1) as f:
    df4 = report2df(BSP(f.read(), 'xml'))


# In[343]:


with open(r'C:\Users\nlgun\Downloads\recent\sec-edgar-filings\0001423053\13F-HR\0000950123-25-005687\full-submission.txt') as f:
    df2 = report2df(BSP(f.read(), 'xml'))


# In[319]:


df2


# In[346]:


df4.query("putCall.fillna(' ').str.contains(r'(?i)^put$') or titleOfClass.str.contains('(?i)short')").sort_values('value',ascending=False).value.sum()/df2.value.sum()


# In[333]:


df3.query("putCall.fillna(' ').str.contains(r'(?i)^put$') or titleOfClass.str.contains('(?i)short')").sort_values('value',ascending=False)


# In[325]:


df3.query("putCall.fillna(' ').str.contains(r'(?i)^put$') or titleOfClass.str.contains('(?i)short')").sort_values('value',ascending=False)


# In[321]:


df.query("putCall.fillna(' ').str.contains(r'(?i)^put$') or titleOfClass.str.contains('(?i)short')").sort_values('value',ascending=False)


# In[273]:


Out[233].set_index('cusip').join(Out[266].set_index('cusip'))


# In[234]:


Out[233].columns


# In[279]:


print(str(testsoup)[:10000])


# In[143]:


testsoup.find('n1:informationTable')
def has_xmlns(tag):
    return tag.has_attr('xmlns')

testsoup.find(has_xmlns)


# In[150]:


ptag = BSP.(r'<n1:informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable" xmlns:n1="http://www.sec.gov/edgar/document/thirteenf/informationtable" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.sec.gov/edgar/document/thirteenf/informationtable eis_13FDocument.xsd">','html.parser')


# In[156]:


testsoup_ = testsoup.infoTable.wrap(testsoup.new_tag(r'informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable" xmlns:n1="http://www.sec.gov/edgar/document/thirteenf/informationtable" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.sec.gov/edgar/document/thirteenf/informationtable eis_13FDocument.xsd"'))


# In[216]:


it  = testsoup.find("infoTable")
X = it.wrap(BSP(str(it)).new_tag('informationTable', "http://www.sec.gov/edgar/document/thirteenf/informationtable","http://www.sec.gov/edgar/document/thirteenf/informationtable"))


# In[205]:


it  = testsoup.find("infoTable")
X = it.wrap(testsoup.new_tag('informationTable'))


# In[217]:


pd.read_xml(StringIO(str(X)))


# In[207]:


testsoup.new_tag('p')


# In[220]:


n = testsoup.find('infoTable')
n1 = n.find('sshPrnamt')


# In[225]:


n1.string,n.cusip.string


# In[88]:


for i in range(10):
    print(next(tsiter))


# In[227]:


pd.read_xml(StringIO(str(testsoup)))


# In[102]:


testsoup.find_next()


# In[59]:


tosio = lambda o: StringIO(str(o))


# In[80]:


pd.read_xml(StringIO(str(testsoup)), xpath = './/n1:infoTable',
                 namespaces={"edgar": "http://www.sec.gov/edgar/document/thirteenf/informationtable"})


# In[50]:


x = testsoup.find_all('shrsOrPrnAmt')


# In[54]:


y = [e1 for e in x for e1 in e.children]


# In[61]:


pd.read_xml(tosio(next(iter(x))),xpath="//edgar:shrsOrPrnAmt", 
                 namespaces={"edgar": "http://www.sec.gov/edgar/nport"})


# In[47]:


[next(testsoup.children) for _ in range(20)]


# In[134]:


s = r'<a>12</a>\n<a>34</a>'
sop = BSP(s,'lxml')


# In[135]:


pd.read_xml(StringIO(str(sop)))


# In[148]:


txt = '''<html><body><n1:informationtable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable" xmlns:n1="http://www.sec.gov/edgar/document/thirteenf/informationtable" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemalocation="http://www.sec.gov/edgar/document/thirteenf/informationtable eis_13FDocument.xsd">\n</informationtable>
<n1:nameOfIssuer>10X GENOMICS INC</n1:nameOfIssuer>
<n1:titleOfClass>CL A COM</n1:titleOfClass>
<n1:cusip>88025U109</n1:cusip>
<n1:value>501414</n1:value>
<n1:shrsOrPrnAmt>
<n1:sshPrnamt>43300</n1:sshPrnamt>
<n1:sshPrnamtType>SH</n1:sshPrnamtType>
</n1:shrsOrPrnAmt>
<n1:putCall>Put</n1:putCall>
<n1:investmentDiscretion>DFND</n1:investmentDiscretion>
<n1:otherManager>1</n1:otherManager>
<n1:votingAuthority>
<n1:Sole>43300</n1:Sole>
<n1:Shared>0</n1:Shared>
<n1:None>0</n1:None>
</n1:votingAuthority></body></html>'''


# In[145]:


tag = r'<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable" xmlns:n1="http://www.sec.gov/edgar/document/thirteenf/informationtable" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.sec.gov/edgar/document/thirteenf/informationtable eis_13FDocument.xsd">\n'
tag+='</informationTable>'
tagbs = BSP(tag,'lxml')
tagbs.extend(testsoup.find('infoTable'))
# pd.read_xml(StringIO(str(tagbs)))
tagbs


# In[150]:


pd.read_xml(StringIO(txt))


# In[122]:


testsoup.find('infoTable').parent.tag


# In[119]:


tss = str(testsoup)
tss[:1000]


# In[107]:


pd.read_xml(StringIO(str(testsoup)))


# In[33]:


k = '0001423053'
len(n:=tabledict[k])
type(n),table_types[get_type(k)]
posns_ = n.find_all('value')


# In[48]:


# for n in testsoup.find_all(re.compile('shrsOrPrnAmt|votingAuthority|otherManager')): n.decompose()
for n in testsoup.find_all(re.compile('shrsOrPrnAmt')):
    n.decompose()


# In[75]:


n = testsoup.find('infoTable')


# # Soup Tests

# In[208]:


path = souppathdict['0001423053'][0]
path = os.path.join(os.path.split(path)[0],'full-submission_short.txt')


# In[209]:


with open(path) as f:
    soup = BSP(f.read(),'xml')


# In[210]:


infotable = soup.find('informationTable')


# In[202]:


re.sub(r'\n','',str(infotable))[:500]


# In[223]:


pd.read_xml(StringIO(str(infotable)),xpath=r'.//*')


# In[219]:


pd.read_xml(StringIO(str(infotable)))


# In[217]:


df = Out[212]
df.iloc[:,-5:-3].dropna()


# In[211]:


infotable.find_all(recursive=False)


# In[200]:


str(infotable)[:500]


# In[189]:


infotable.find_all()[:3]


# In[178]:


xtrs = [n.extract() for n in infotable.find_all('shrsOrPrnAmt')]


# In[179]:


len(xtrs)


# In[181]:


xtrs[0]


# In[185]:


list(xtrs[0].find_all())


# In[ ]:





# In[76]:


# for e in n.children: print(e)
m = n.find('shrsOrPrnAmt')
n.extend(m.children)
m.decompose()
n


# In[89]:


# next(testsoup.children)
# next(testsoup.children)
str(testsoup)[:500]


# In[59]:


n.find('shrsOrPrnAmt').decompose()
n#.extend(n.find('shrsOrPrnAmt'))


# In[60]:


pd.read_xml(StringIO(str(n)))


# In[228]:


x = pd.read_xml(StringIO(str(testsoup)))


# In[53]:


x


# In[105]:


testsoup.head()


# In[87]:


len(posns),len(posns_)


# In[82]:


posns[:3]


# In[250]:


for k in ciks:
    print((basepath := os.path.join(r'C:\Users\nlgun\Downloads\recent\sec-edgar-filings',k,report_dict[k].type)))
    print([os.path.join(triple[0],fn) for triple in find_file_leaves(basepath) for fn in triple[-1]][:3])



# In[13]:


def schedule2df(path,extract_date = extract_date,tablerex = 'informationTable',with_info=True):
    assert os.path.splitext(path)[1] == '.txt'
    with open(path) as f: soup = BSP(f.read(),'xml')
    table = soup.find(tablerex)
    table = pd.read_xml(StringIO(str(table))) if with_info else None
    return table,extract_date(soup),extract_value(soup)


# In[98]:


os.path.basename('a/b')


# In[16]:


filing = namedtuple('filing','path date name dfs'.split())

d = defaultdict(list)
filerex = r'(?i)^\w+\W?submission.txt$'
namerex =  r'(?i)^[\w\s]+CONFORMED NAME:\s*([ \w,\.]+)\s*$'
daterex = r'(?i)^[\w\s]+AS OF DATE:[\s\w]+(\d{8})\s*$'
cikrex =  r'(?i)^[\w\s]+INDEX KEY:[\s\w]+(\d{10})\s*$'
name = re.search(namerex,text,re.MULTILINE)
date = re.search(daterex,text,re.MULTILINE)
cik = re.search(cikrex,text,re.MULTILINE)
# re.search(filerex,'full-submission.txt')
for root,fldrs,fns in tqdm(os.walk(r'Downloads\recent')):
    fldrs[:] = [fld for fld in fldrs if not re.search('36405',fld)]
    for fn in fns: 
        if re.search(filerex,fn):
            print(root,fn)
            with open(os.path.join(root,fn)) as f:
                text = f.read()
                path = os.path.join(root,fn)
                # with open(path) as f: text = f.read()
            if re.search("36405",cik[0]): continue
            if all((name,date,cik)):
                print(path)
                d[cik[1]].append(filing(path, pd.to_datetime(date[1].strip()),name[1].strip(),schedule2df(path,tablerex = 'informationTable')))


# In[220]:


filedir = make_secpath(report_dict['Renaissance'])


# In[227]:


citpaths = [os.path.join(filedir,fn,'full-submission.txt') for fn in os.listdir(filedir)]


# In[228]:


citpaths


# In[259]:


soup = file2soup(paths4soup[0]) ####HERE!!!


# In[150]:


next(fileiter)


# In[153]:


fileinfo = Out[150]
path = os.path.join(fileinfo[0],fileinfo[-1][0])
path


# In[210]:


citpaths[0]


# In[209]:


soup = file2soup()


# In[162]:


header = soup.find('SEC-HEADER').text


# In[170]:


def proc(header):

    for line in [l.strip() for l in header.split('\n') if l.strip()]:
        if (m:=re.search(r'^FILED.*DATE:\s+(\d+)$',line)):print(m[1])


# In[6]:


def find_extension_in_folder(fldr,test = lambda s: os.path.splitext(s)[1] == '.txt', i  = None):
    fns = [fn for fn in os.listdir(fldr) if test(fn)]
    return fns if i == None else fns[i] if fns else None


# In[172]:


for filinfo in fileiter:
    path = os.path.join(fileinfo[0],fileinfo[-1][0])
    with open(path) as f:
        soup = BSP( f.read(),'xml')
    header = soup.find('SEC-HEADER').text
    proc(header)



# In[5]:


basename  = dl.download_folder
for name, cik in zip(names,ciks):
    print(name,cik,dl.download_folder)
    # dl.download_folder = os.path.join(basename,name)
    # dl.get("NPORT-P", cik,limit=5) 
    dl.get("13F-HR", cik,limit=5) 
# dl.download_folder = basename


# In[6]:


a = [1,2,3]
a.clear()
a


# In[7]:


find_extension_in_folder(r'Downloads\sec-edgar-filings\0001423053\13F-HR\0000950123-25-002739',i=0)


# In[ ]:


def get_dates_and_paths(pfld):



# In[7]:


from functools import partial

@dataclass
class Finder(object):

    @staticmethod
    def _find(txt,rex): return re.search(rex,txt,re.MULTILINE)

    text:str

    def make_finders(self,*rexs):
        self.findtext = partial(self.__class__._find,self.text)
        return [partial(self.findtext,rex) for rex in rexs]

@dataclass
class ReportFinder(Finder):

    @staticmethod
    def _soupfinder(soup,string): return soup.find(string)

    def __post_init__(self):
        self.soup = BSP(self.text,'xml')

    def make_soupfinders(self,*strings):
        return [partial(self.__class__._soupfinder, self.soup,string) for string in strings]



# In[8]:


# with open(r'Downloads\recent\sec-edgar-filings\0000036405\NPORT-P\0001752724-25-126249\full-submission.txt') as f: text = f.read()
fn = r'0001752724-25-126209.txt'
with open(os.path.join(r'Downloads',fn)) as f: text = f.read()

finder = ReportFinder(text)
# finder.soup


# In[9]:


# finders = finder.make_finders( r'(?i)^\w+\W?submission.txt$', r'(?i)^[\w\s]+CONFORMED NAME:\s*([ \w,\.]+)\s*$',r'(?i)^[\w\s]+AS OF DATE:[\s\w]+(\d{8})\s*$',r'(?i)^[\w\s]+INDEX KEY:[\s\w]+(\d{10})\s*$')
args = r"(?i)AS OF DATE:\s*(\d+)\s*$;(?i)^[\w\s]+CONFORMED NAME:\s*([ \w,\.]+)\s*".split(';')
print(args)
finders = finder.make_finders(*args)


# In[10]:


finder.text[:100]
rex = r"(?i)AS OF DATE:\s*(\d+)\s*$"
# re.search(rex,finder.text,re.MULTILINE)
# Finder._find(rex,finder.text)
finders[1]()[1]
# find = partial(Finder._find,finder.text)
# find(rex)
# finder.findtext(rex)


# In[9]:


soupfinders = finder.make_soupfinders('invstOrSecs')


# In[10]:


vangdf = pd.read_xml(StringIO(str(soupfinders[0]())))
len(vangdf)


# In[11]:


vangdf


# In[87]:


vwdf = vangdf['cusip valUSD'.split()].set_index('cusip')/vangdf.valUSD.sum()


# In[89]:


vwdf.hist(bins=100)


# In[85]:


Out[84]/1e8


# In[27]:


finders[0]()


# In[12]:


extract_date = lambda soup: pd.to_datetime(soup.find('ACCEPTANCE-DATETIME').text.split('\n')[0][:8])
extract_value = lambda soup: soup.find('tableValueTotal').text


# In[53]:


from difflib import SequenceMatcher as sqmatch
def getcik(name):
    score_array= [(k,v,sqmatch(a=name,b=v.lower()).find_longest_match().size) for k,v in cikdict.items()]
    score_array.sort(key = lambda x: x[-1])
    return score_array[-1]


# In[37]:


cikdict


# In[46]:


sqmatch(a='ren',b='renaissance').find_longest_match().size


# In[55]:


getcik('cit')


# In[69]:


Out[59].assign(w = Out[59].value/Out[59].value.sum())


# In[82]:


def get_ws(df,namekey, valkey):
    df_ = df.set_index('cusip')[[namekey,valkey]]
    return df_.assign(w = df_[valkey]/df_[valkey].sum())
citwdf = get_ws(Out[59],'nameOfIssuer','value')


# In[79]:


valwdf = get_ws(vangdf,'name','valUSD')


# In[86]:


citwdf.w.to_frame()


# In[92]:


len(citwdf.index),len(set(citwdf.index)),len(valwdf.index),len(set(valwdf.index))


# In[109]:


fig,ax = plt.subplots(1,1)
ax.set_aspect(1)
(citvangwdf := pd.concat((citwdf.w.to_frame(),valwdf.w.groupby(level=0,axis=0).sum().to_frame()),axis=1,keys='cit vang'.split()).fillna(0).droplevel(level=1,axis=1)).plot.scatter(x='cit',y='vang',ax=ax)


# In[117]:


citvangwdf.assign(diffw = citvangwdf.vang - citvangwdf.cit).sort_values('diffw').join(valwdf[['name']]).join(citwdf[['nameOfIssuer']]).head(50)


# In[59]:


d['0001423053'][0][-1][0]


# In[15]:


tple = next(iter(d.values()))


# In[15]:


path = r'Downloads\recent\sec-edgar-filings\0000036405\NPORT-P\0001752724-25-126249\full-submission.txt'
def get_soup(path):
    assert os.path.splitext(path)[1] == '.txt'
    with open(path) as f: 
        text = f.read()
        soup = BSP(text,'xml')
    return text,soup
text,soup = get_soup(path)


# In[24]:


re.search("(?i)AS OF DATE:\s*(\d+)\s*$",text,re.MULTILINE)[1].strip()


# In[27]:


rp = ReportParser(path)
rp.text[:100]


# In[14]:


table = soup.find('invstOrSecs')
# table = pd.read_xml(StringIO(str(table))) #if with_info else None
print(table)


# In[20]:


for v in d.values():v.sort(key = lambda tpl: tpl[2])


# In[21]:


d


# In[10]:


k = next(iter(d.keys()))


# In[ ]:


print(d[k][-1].date)
d['0001423053'][-1].text[0].query('nameOfIssuer == "1 800 FLOWERS COM INC"')


# In[22]:


df = d['0001423053'][-1].text[0].groupby('nameOfIssuer titleOfClass'.split())['value'].sum()


# In[25]:


df/df.sum()*100


# In[ ]:


stock_data = pd.read_csv('stock_cusip_mapping.csv')

# Get CUSIP for a specific stock symbol
symbol = 'AAPL'  # Apple Inc.
cusip = stock_data[stock_data['Symbol'] == symbol]['CUSIP'].values[0]
print(f"The CUSIP for {symbol} is {cusip}")


# In[ ]:





# In[7]:


d = defaultdict(list)

filerex = r'(?i)^\w+\W?submission.txt$'
namerex =  r'(?i)^[\w\s]+CONFORMED NAME:\s*([ \w,\.]+)\s*$'
daterex = r'(?i)^[\w\s]+AS OF DATE:[\s\w]+(\d{8})\s*$'
cikrex =  r'(?i)^[\w\s]+INDEX KEY:[\s\w]+(\d{10})\s*$'
# re.search(filerex,'full-submission.txt')
for root,fldrs,fns in os.walk(r'Downloads\recent'):
    for fn in fns: 
        if re.search(filerex,fn):
            print(root)
            with open(os.path.join(root,fn)) as f:
                text = f.read()
                name = re.search(namerex,text,re.MULTILINE)
                date = re.search(daterex,text,re.MULTILINE)
                cik = re.search(cikrex,text,re.MULTILINE)
            if all((name,date,cik)):
                d[cik[1]].append((root,fn,pd.to_datetime(date[1].strip()),name[1].strip()))


# In[7]:


d = defaultdict(list)

filerex = r'(?i)^\w+\W?submission.txt$'
namerex =  r'(?i)^[\w\s]+CONFORMED NAME:\s*([ \w,\.]+)\s*$'
daterex = r'(?i)^[\w\s]+AS OF DATE:[\s\w]+(\d{8})\s*$'
cikrex =  r'(?i)^[\w\s]+INDEX KEY:[\s\w]+(\d{10})\s*$'
# re.search(filerex,'full-submission.txt')
for root,fldrs,fns in os.walk(r'Downloads\recent'):
    for fn in fns: 
        if re.search(filerex,fn):
            print(root)
            with open(os.path.join(root,fn)) as f:
                text = f.read()
                name = re.search(namerex,text,re.MULTILINE)
                date = re.search(daterex,text,re.MULTILINE)
                cik = re.search(cikrex,text,re.MULTILINE)
            if all((name,date,cik)):
                d[cik[1]].append((root,fn,pd.to_datetime(date[1].strip()),name[1].strip()))


# In[21]:


# print((cik_keys :=d.keys()))
k = next(iter(cik_keys))


# In[26]:


print(d[k][0][:-1])


# In[28]:


for tpl in d[k]: print(tpl[:-1])


# In[55]:


for k,v in d.items():
    d[k] = (v,list(map(lambda e: schedule2df(os.path.join(*e[:2])),v)))


# # Specification

# specification: v = (list(metdata),list((table,date,totalamount)))

# In[9]:


df = pd.concat([(pd.concat([rawdfs[j][1][i][0]['nameOfIssuer value'.split()].groupby('nameOfIssuer')['value'[]].sum() for i in range(5)],axis=1,keys = [rawdfs[0][1][i][1] for i in range(5)])) for j in range(3)],
         keys = 'Renaissance Bridgewater Citadel'.split(),axis=0)


# In[65]:


@dataclass
class Report(object):
    cik: str
    path: str
    total: int
    filedate: pd.Timestamp
    df: pd.DataFrame




# In[10]:


rd = defaultdict(list)
for k,v_ in d.items():
    for a,b in zip(*v_):
        path = os.path.join(*a[:2])
        # print(path)
        # print(v[1])
        total = int(b[-1])
        filedate = b[1]
        df = b[0]
        rd[k].append(Report(k,path,total,filedate,df))


# In[ ]:





# In[160]:


dfd = {}
for k,v_ in rd.items():
    df = pd.concat([r.df['nameOfIssuer value'.split()].groupby('nameOfIssuer')['value'].sum() 
                    for r in v_], axis = 1, keys = [r.filedate for r in v_]).fillna(0)
    # df = pd.concat([df.diff(axis=1),pd.DataFrame([r.total for r in v_],index=df.columns,columns='total'.split()).T])
    # df = df.diff(axis=1).dropna(how='all',axis=1)
    # df = pd.concat([df,pd.DataFrame([r.total for r in v_[1:]],index=df.columns,columns='total'.split()).T])
    dfd[k]=df


# In[161]:


ren = '0001037389'
dfd[ren].astype(int)


# In[218]:


diffd[ren].divide(diffd[ren].loc['total'].values,axis=1)


# In[230]:


diffd = {}
for k,v in dfd.items():
    delta = v.diff(axis=1).dropna(how='all',axis=1)
    totals = pd.Series([r.total for r in rd[k]],index = v.columns,name='total').to_frame().T
    diffd[k] = (100*delta.divide(totals[delta.columns].values,axis=1)).rename(columns = lambda x: (x.year,x.month))\
        .assign(name = cikdict[k],cik = k).round(5)
# totals,delta[ren].columns


# In[217]:


diffd = {}
for k,v in dfd.items():
    delta = v.diff(axis=1).dropna(how='all',axis=1)
    totals = pd.Series([r.total for r in rd[k]],index = v.columns,name='total').to_frame().T
    diffd[k] = pd.concat([delta,totals[delta.columns]]).rename(columns = lambda x: (x.year,x.month))\
       # .assign(name = cikdict[k],cik = k).round(0)
# totals,delta[ren].columns


# In[237]:


df = pd.concat(diffd.values()).sort_values((2025,2),ascending=False)
df.head(50).iloc[:,:-1]


# In[245]:


abs(df.iloc[:,:4]).max(axis=1)


# In[251]:


df = df.assign(max=abs(df.iloc[:,:4]).max(axis=1)).sort_values('max',ascending=False)
df.drop('cik'.split(),axis=1).head(25).sort_index()


# In[192]:


diffd[ren].sort_values(diffd[ren].columns[-1],ascending=False).drop('total').head(25)\
#.style.format('${:,.0f}')


# In[180]:


rcikdict = {v:k for k,v in cikdict.items()}
def nearkey(d,k):
    rex = '(?i)' + k
    keys = [k_ for k_ in d.keys() if re.search(rex,k_)]
    return keys[0] if keys else None
nearkey(rcikdict,'cit')


# In[185]:


key = rcikdict.get((name:=nearkey(rcikdict,'cit')))
print(name)
diffd[key].sort_values(diffd[key].columns[-1],ascending=False).drop('total').head(25)\
.style.format('${:,.0f}')


# In[186]:


key = rcikdict.get((name:=nearkey(rcikdict,'bridg')))
print(name)
diffd[key].sort_values(diffd[key].columns[-1],ascending=False).drop('total').head(25)\
.style.format('${:,.0f}')


# In[165]:


delta[ren] = dfd[ren].diff(axis=1).dropna(how='all',axis=1)
totals = pd.Series([r.total for r in rd[ren]],index = dfd[ren].columns,name='total').to_frame().T
totals[delta[ren].columns]
# totals,delta[ren].columns


# In[167]:


pd.concat([delta[ren],totals[delta[ren].columns]]).round(0)#.astype(int)


# In[ ]:





# In[ ]:





# In[45]:


k0,v0 = next(iter(d.items()))
delta = lambda v: pd.concat([t[0]['nameOfIssuer value'.split()].groupby('nameOfIssuer')['value'].sum()  for t in v[1]],axis=1,
         keys = [t[-2] for t in v[0]]).fillna(0).diff(axis=1).dropna(how='all',axis=1)

dfd = {(k,v[0][0][-1]):delta(v) for k,v in d.items()}


# In[48]:


dfd[('0001037389', 'RENAISSANCE TECHNOLOGIES LLC')]


# In[41]:


k0,v0 = next(iter(d.items()))
pd.concat([t[0]['nameOfIssuer value'.split()].groupby('nameOfIssuer')['value'].sum()  for t in v0[1]],axis=1,
         keys = [t[-2] for t in v0[0]]).fillna(0).diff(axis=1).dropna(how='all',axis=1)


# In[ ]:

# In[8]:

basepath= r'Downloads\sec-edgar-filings\0001423053\13F-HR'
def proc_path(path=basepath,depth = 0,maxdepth = 3, cum = list()):
    # alternative strategy for cum, which is created when function is first parsed.
    # cum = cum_.copy() # needed or repeated calls cn result in duplidate entries.  Why is unclear.
    # del cum # explicit garbage collection
    cum.clear()
    if (found := find_extension_in_folder(path,i = 0)):
        cum.append((found,path))
    # get the folders
    fldrs = []
    for nm in os.listdir(path):
        if os.path.isdir(pathnm := os.path.join(path,nm)):
            fldrs.append(pathnm)
    if fldrs and depth < maxdepth:
        for fldr in fldrs:
            if (found := proc_path(fldr,depth+1,maxdepth,cum = list())):
                cum += found # return here
    return cum.copy() # avoid future clear() calls  



# In[3]:


tpl = ('full-submission.txt',
  'Downloads\\sec-edgar-filings\\0001423053\\13F-HR\\0000950123-24-002516')

with open(os.path.join(*list(reversed(tpl)))) as f:
    soup = BSP(f.read(),'xml')


# In[4]:


soup.find('informationTable')


# In[9]:


ctdls = proc_path()
ctdls


# In[13]:


allhfs = proc_path( r'Downloads\sec-edgar-filings',maxdepth= 7)


# In[24]:


path = allhfs[0][1]
re.search(r'\W\d{10}\W',path)[0][1:-1]
allhfs


# In[ ]:


filingdict = {}


# In[16]:


namedfilings = [(cikdict[re.search(r'\W\d{10}\W',path)[0][1:-1]],fn,path) for fn, path in allhfs]
filingdict = {}
for nm, fn,path in namedfilings:
    if nm in filingdict: filingdict[nm].append((fn,path))
    else: filingdict[nm]=[(fn,path)]
filingdict


# In[60]:


d


# In[ ]:


processed = [schedule2df(os.path.join(*reversed(e))) for e in filingdict['Renaissance']]


# In[37]:


processed = [(e[0],extract_date(e[1])) for e in processed] # now automatic with schedule2df updated to include extract_date(soup)


# In[104]:


len(filingdict['Renaissance'])


# In[15]:


# rawdfs = [(k,pd.concat(schedule2df(os.path.join(*reversed(e))))) for k,v in tqdm(filingdict.items())]


# In[19]:


rawdfs = [(k,[schedule2df(os.path.join(*reversed(e))) for e in v],len(v)) for k,v in tqdm(filingdict.items())]


# In[32]:


rawdfs_ = [(k,[schedule2df(os.path.join(*reversed(e)),with_info=False) for e in v],len(v)) for k,v in tqdm(filingdict.items())]


# In[33]:


rawdfs_[0]


# In[20]:


df = pd.concat([(pd.concat([rawdfs[j][1][i][0]['nameOfIssuer value'.split()].groupby('nameOfIssuer')['value'].sum() for i in range(5)],axis=1,keys = [rawdfs[0][1][i][1] for i in range(5)])) for j in range(3)],
         keys = 'Renaissance Bridgewater Citadel'.split(),axis=0)


# In[21]:


df


# In[195]:


(df.sort_values(df.columns[-1],ascending=False)/1e6).round(0).sort_index(level=0).head(50)


# In[23]:


(df.loc['Renaissance'].sort_values(df.columns[-1],ascending=False)/1e6).head(50).round(1)


# In[179]:


for nm,df_ in df.groupby(level=0):
    print((df_.dropna().diff(axis=1).dropna(how='all',axis=1)/1e6).head(50).round(0))
    # break


# In[ ]:





# In[22]:


for nm,df_ in df.T.groupby(level=0):
    print((df_.T.dropna().diff(axis=1).dropna(how='all',axis=1)/1e6).sort_values(df_.T.columns[-1],ascending=False).tail(50).round(0))
    # break


# In[169]:


for nm,df_ in df.T.groupby(level=0):
    print((df_.T.dropna().diff(axis=1).dropna(how='all',axis=1)/1e6).sort_values(df_.T.columns[-1],ascending=False).tail(50).round(0))
    # break


# In[110]:





# In[100]:


d['Citadel'][0]


# In[83]:


def aggregate_dfs(rawdfs):
    d = {}
    for k,df_date in tqdm(rawdfs):
        df,date = df_date
        print(k,df.columns[:5])
        # df.columns = pd.MultiIndex.from_product(([date],df.columns))
        # if k in d: d[k].append(df)
        # else: d[k] = [df]
    return d
d = aggregate_dfs(rawdfs)


# In[32]:


def aggregate_dfs(rawdfs):
    d = {}
    for k,df_date in tqdm(rawdfs):
        df,date = df_date
        df.columns = pd.MultiIndex.from_product(([date],df.columns))
        if k in d: d[k].append(df)
        else: d[k] = [df]
        return {k: pd.concat(v,axis=1) for k,v in d.items()}

d = aggregate_dfs(rawdfs)


# In[44]:


d1 = {k:pd.concat(v,axis=1) for k,v in d.items()}


# In[73]:


ks,vs = zip(*d1.items())
bigdf = pd.concat(vs,axis=1,keys = ks)


# In[74]:


bigdf = bigdf.swaplevel(-1,0,axis=1)['nameOfIssuer value'.split()].swaplevel(-1,0,axis=1)#.columns


# In[75]:


bigdf.columns.names = 'fund date field'.split()


# In[79]:


bigdf.T.sort_index(level='fund date field'.split()).T


# In[48]:


for df in dfs: assert 'nameOfIssuer' in df.columns


# In[51]:


df = pd.DataFrame(dict(zip(keys,[df_.groupby('nameOfIssuer').value.sum().sort_values(ascending=False) for df_ in dfs])))


# In[63]:


(df.fillna(0).sort_values(df.columns[-1],ascending=False)/1e9).round(3).head(50)


# In[49]:


df = [pd.concat(df_.groupby('nameOfIssuer').value.sum().sort_values(ascending=False).,keys = keys,axis=1) for df_ in dfs]


# In[43]:


df


# In[83]:


tuple(reversed(ctdls[0]))


# In[90]:


(out:=schedule2df(os.path.join(*reversed(ctdls[0]))))[0]


# In[100]:


out[0].groupby('nameOfIssuer').value.sum().sort_values(ascending=False)


# In[112]:


# pd.to_datetime(out[1].find('ACCEPTANCE-DATETIME').text.split('\n')[0][:8])
[pd.to_datetime(e[1].find('ACCEPTANCE-DATETIME').text.split('\n')[0][:8]) for e in crldsprocessed]


# In[116]:


ctldsprocessed = [schedule2df(os.path.join(*reversed(e))) for e in ctdls]
# ctldsprocessed = crldsprocessed


# In[117]:


[e[1].find('ACCEPTANCE-DATETIME').text.split('\n')[0][:8] for e in ctldsprocessed]


# In[120]:


# ctldsprocessed = [schedule2df(os.path.join(*reversed(ctdls[0]))) for e in ctdls]
dfdict = {pd.to_datetime(e[1].find('ACCEPTANCE-DATETIME').text.split('\n')[0][:8]):e[0].groupby('nameOfIssuer').value.sum().sort_values(ascending=False) for e in ctldsprocessed}


# In[169]:


((ctldf := pd.DataFrame(dfdict).diff(axis=1).iloc[:,1:])/ctldf.sum()).sort_values(pd.to_datetime('2025-02-14'),ascending=False).head(50)


# In[178]:


changedf =(ctldf/ctldf.abs().sum()).sort_values(pd.to_datetime('2025-02-14'),ascending=False)


# In[185]:


changedf.dropna(axis=0,how='all',inplace=True)
changedf *=100


# In[186]:


changedf.round(2)


# In[187]:


changedflast = changedf[changedf[changedf.columns[-1]].notna()]


# In[188]:


changedf.T.plot(legend=False)


# In[189]:


changedf = changedf.sort_values(pd.to_datetime('2025-02-14'),ascending=False)


# In[190]:


changedf.iloc[:10].T.plot(figsize=(10,10),grid=True)


# In[191]:


changedflast.iloc[:10].T.plot(figsize=(10,10),grid=True)


# In[192]:


changedflast.iloc[-10:].T.plot(figsize=(10,10),grid=True)


# In[193]:


changedflast

