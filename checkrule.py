import scipy
from utils import split_words, roformer_encoder, get_csvdf, get_rulefolder, get_embedding,df2aggrid
import ast
from streamlit_echarts import st_echarts
import streamlit as st
import pandas as pd

rulefolder = 'rules'
secpath='rules/sec1.csv'
plcpath='rules/lawdfall0507.csv'
metapath='rules/lawmeta0517.csv'
dtlpath='rules/lawdtl0517.csv'

def get_samplerule(key_list, industry_choice):
    rulefolder = get_rulefolder(industry_choice)
    plcdf = get_csvdf(rulefolder)
    selectdf = plcdf[plcdf['监管要求'].isin(key_list)]
    tb_sample = selectdf[['监管要求', '结构', '条款']]
    return tb_sample.reset_index(drop=True)


def searchrule(text, column_text, make_choice, industry_choice, top):
    queries = [text]
    query_embeddings = roformer_encoder(queries)

    searchdf = get_samplerule(make_choice, industry_choice)
    # search rule
    ruledf, _ = searchByItem(searchdf, make_choice, column_text, '')
    rulefolder = get_rulefolder(industry_choice)
    emblist = ruledf['监管要求'].drop_duplicates().tolist()
    subsearchdf = get_samplerule(emblist, industry_choice)
    # fix index
    fixruledf, _ = searchByItem(subsearchdf, emblist, column_text, '')
    # get index of rule
    rule_index = fixruledf.index.tolist()

    sentence_embeddings = get_embedding(rulefolder, emblist)
    # get sub embedding
    sub_embedding = sentence_embeddings[rule_index]

    avglist = []
    idxlist = []
    number_top_matches = top
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding],
                                                 sub_embedding, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        for idx, distance in results[0:number_top_matches]:
            idxlist.append(idx)
            avglist.append(1 - distance)

    return fixruledf.iloc[idxlist]


def searchByName(search_text, industry_choice):
    rulefolder = get_rulefolder(industry_choice)
    plcdf = get_csvdf(rulefolder)
    plc_list = plcdf['监管要求'].drop_duplicates().tolist()

    choicels = []
    for plc in plc_list:
        if search_text in plc:
            choicels.append(plc)

    plcsam = get_samplerule(choicels, industry_choice)

    return plcsam, choicels


def searchByItem(searchresult, make_choice, column_text, item_text):

    # split words item_text
    item_text_list = split_words(item_text)
    column_text = fix_section_text(column_text)
    plcsam = searchresult[(searchresult['监管要求'].isin(make_choice))
                          & (searchresult['结构'].str.contains(column_text)) &
                          (searchresult['条款'].str.contains(item_text_list))]
    total = len(plcsam)
    return plcsam, total


# fix section text with +
def fix_section_text(section_text):
    if '+' in section_text:
        section_text = section_text.replace('+', '\\+')
    return section_text


def df2echart(df):
    data = dict()
    data['name'] = '法规分类'
    df['children']=df['children'].str.replace('id','value')
    # fillna(0)是为了防止出现nan
    df['children'] = df['children'].fillna('[]')
    # literal_eval 将字符串转换为字典 ignore 忽略掉异常
    df['children'] = df['children'].apply(ast.literal_eval)
    data['children'] = df.iloc[:3]['children'].tolist()
    # st.write(data)
    option = {
        "tooltip": {
            "trigger": "item",
            "triggerOn": "mousemove"
        },
        "series": [{
            "type": "tree",
            "data": [data],
            # "top": "1%",
            # "left": "7%",
            # "bottom": "1%",
            # "right": "20%",
            # "symbolSize": 7,
            "label": {
                "position": "left",
                "verticalAlign": "middle",
                "align": "right",
                # "fontSize": 9,
            },
            "leaves": {
                "label": {
                    "position": "right",
                    "verticalAlign": "middle",
                    # "align": "left",
                }
            },
            # "emphasis": {
            #     "focus": "descendant"
            # },
            # "expandAndCollapse": True,
            # "animationDuration": 550,
            # "animationDurationUpdate": 750,
        }],
    }
    events = {
    "click": "function(params) { console.log(params.name); return [params.name,params.value]  }",
    # "dblclick":"function(params) { return [params.type, params.name, params.value] }"
    }

    value =st_echarts(option, height="500px",events=events)
    return value


def get_children(df,ids):
    child=df[df['pId']==ids]
    idls=child['id'].tolist()
    return idls


def get_allchildren(df,ids):
    result=[]
    brother=get_children(df,ids)
    for bro in brother:
        little=get_children(df,bro)
        if little ==[]:
            result+=[bro]
        else:
            result+=little
    if result==[]:
        result=[ids]
    return result


# get rule list by id
def get_rulelist(idls):
    plcdf=pd.read_csv(plcpath)
    plclsdf=plcdf[plcdf['id'].isin(idls)]
    # reset index
    plclsdf=plclsdf.reset_index(drop=True)
    cols=['secFutrsLawName', 'fileno','lawPubOrgName','secFutrsLawVersion','secFutrsLawId']
    plclsdf=plclsdf[cols]
    # change column name
    plclsdf.columns=['文件名称','文号','发文单位','发文日期','id']
    return plclsdf


def get_ruletree():
    secdf = pd.read_csv(secpath)
    selected=df2echart(secdf)
    if selected is not None:
        [name, ids] = selected
        idls = get_allchildren(secdf, ids)
        plclsdf = get_rulelist(idls)
        # get total
        total = len(plclsdf)
        # display name,ids and total
        st.info('{} id: {} 总数: {}'.format(name, ids, total))
        # st.table(plclsdf)
        # display lawdetail
        display_lawdetail(plclsdf)


def get_lawdtlbyid(ids):
    metadf = pd.read_csv(metapath)
    metadf = metadf[metadf['secFutrsLawId'].isin(ids)]
    metacols=['secFutrsLawName', 'secFutrsLawNameAnno', 'wtAnttnSecFutrsLawName',
       'secFutrsLawVersion', 'fileno', 'body', 'bodyAgoCntnt']
    metadf=metadf[metacols]
    # fillna to empty
    metadf=metadf.fillna('')
    metadf.columns=['文件名称','文件名称注解','法律条文名称','法律条文版本','文号','正文','正文注解']
    metadf = metadf.reset_index(drop=True)
    dtldf=pd.read_csv(dtlpath)
    dtldf=dtldf[dtldf['id'].isin(ids)]
    dtlcol=['title', 'cntnt_x', 'cntnt_y']
    dtldf=dtldf[dtlcol]
    # fillna all columns with ''
    dtldf = dtldf.fillna('')
    # change column name
    dtldf.columns = ['标题', '内容', '法规条款']
    dtldf=dtldf.reset_index(drop=True)
    return metadf,dtldf

# display event detail
def display_lawdetail(search_df):
 
    data=df2aggrid(search_df)
    # display data
    selected_rows = data["selected_rows"]
    if selected_rows==[]:
        st.error('请先选择查看详情')
        st.stop()

    # display selected rows
    st.markdown('选择法规:')
    # convert selected rows to dataframe
    selected_df = pd.DataFrame(selected_rows)
    st.table(selected_df)
    # get id
    idls = selected_df['id'].tolist()
    # st.write(idls)
    metadf,dtldf=get_lawdtlbyid(idls)
    # display meta data
    st.markdown('法规元数据:')
    st.table(metadf)
    # display detail data
    st.markdown('法规详情:')
    st.table(dtldf)