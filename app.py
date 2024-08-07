import streamlit as st

from checkrule import (  # searchrule,; get_lawdtlbyid,; searchByNamesupa,; display_lawdetail,; get_orglist,; get_rulelist_byname,; get_ruletree,
    searchByIndustrysupa,
    searchByItem,
    searchByName,
    searchByNamesupa,
)

# from plc2audit import predict
from gptfuc import (
    add_ruleindex,
    build_ruleindex,
    delete_db,
    gpt_answer,
    similarity_search,
)
from utils import (  # keybert_keywords get_most_similar,; combine_df_columns,
    get_folder_list,
    get_section_list,
)

rulefolder = "rules"


# set page layout to wide
# st.set_page_config(page_title="Check Rule", layout="wide")


def main():
    # st.subheader("监管制度搜索")
    industry_list = get_folder_list(rulefolder)

    industry_list.append("全部")

    industry_choice = st.sidebar.selectbox("选择行业:", industry_list)

    # if industry_choice == "全部":

    #     # choose search type using radio
    #     search_type = st.sidebar.radio("搜索类型", ["法规结构", "法规搜索"])
    #     if search_type == "法规结构":
    #         # display rule tree
    #         st.markdown("### 法规体系")
    #         get_ruletree()
    #     elif search_type == "法规搜索":
    #         st.markdown("### 法规搜索")
    #         # get org list
    #         orgdf = get_orglist()
    #         # get org list
    #         org_list = orgdf["name"].tolist()
    #         # use column container
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             # input file name
    #             law_name = st.text_input("文件名称")
    #             # input start date, default is 2020-01-01

    #             start_date = st.date_input("开始日期", value=pd.Timestamp("2020-01-01"))
    #             # input org name
    #             org_name = st.multiselect("发文单位", org_list)

    #         with col2:
    #             # input fileno
    #             fileno = st.text_input("文号")
    #             # input end date
    #             end_date = st.date_input("结束日期")
    #         # if both are empty
    #         if law_name == "" and fileno == "" and org_name == []:
    #             st.error("请输入搜索关键词")
    #             st.stop()
    #         ruledf = get_rulelist_byname(
    #             law_name, fileno, org_name, start_date, end_date
    #         )
    #         display_lawdetail(ruledf)

    # else:

    match = st.sidebar.radio(
        "搜索方式",
        ("关键字搜索", "模糊搜索", "智能问答", "智能对话", "生成模型", "模型管理"),
    )
    # initialize session value search_result
    if "search_result" not in st.session_state:
        st.session_state["search_result"] = None

    # placeholder
    placeholder = st.empty()

    if match == "关键字搜索":
        name_text = ""
        searchresult, choicels = searchByNamesupa(name_text, industry_choice)

        make_choice = st.sidebar.multiselect("选择监管制度:", choicels)

        if make_choice == []:
            make_choice = choicels
        section_list = get_section_list(searchresult, make_choice)
        column_text = st.sidebar.multiselect("选择章节:", section_list)
        if column_text == []:
            column_text = ""
        else:
            column_text = "|".join(column_text)

        item_text = st.sidebar.text_input("按条文关键字搜索")
        # radio to choose whether to use the new keywords
        # use_new_keywords = st.sidebar.radio('精确模式（+章节信息）', ('否', '是'))

        if column_text != "" or item_text != "":
            fullresultdf, total = searchByItem(
                searchresult, make_choice, column_text, item_text
            )
            # new_keywords_list = item_text.split()

            # if use_new_keywords == '是' and len(fullresultdf) > 0:
            #     # proc_list = fullresultdf['条款'].tolist()
            #     proc_list = combine_df_columns(fullresultdf, ['结构', '条款'])
            #     top = len(proc_list)
            #     subidx = get_most_similar(new_keywords_list,
            #                               proc_list,
            #                               top_n=top)
            #     # st.write('关键词匹配结果：')
            #     # st.write(subidx)
            #     # get subuploaddf based on index list
            #     resultdf = fullresultdf.iloc[subidx]
            # else:
            resultdf = fullresultdf

            if resultdf.empty:
                placeholder.text("没有搜索结果")
            else:
                # reset index
                resultdf = resultdf.reset_index(drop=True)
                placeholder.table(resultdf)
            # search is done
            # st.sidebar.success('搜索完成')
            st.sidebar.success("共搜索到" + str(total) + "条结果")
            st.sidebar.download_button(
                label="下载搜索结果",
                data=resultdf.to_csv(),
                file_name="监管制度搜索结果.csv",
                mime="text/csv",
            )

            # st.sidebar.subheader("聚类分组")
            # s = st.sidebar.slider('分组阈值', max_value=20, value=10)
            # thresholdsort = s / 10
            # st.sidebar.write('分组阈值', thresholdsort)
            # # cluster the result
            # cluster = st.sidebar.button('聚类分组')
            # if cluster:
            #     # empty the table
            #     placeholder.empty()
            #     with st.spinner('正在分组...'):
            #         dfsort, clusternum = items2cluster(
            #             plcsam, thresholdsort)
            #         placeholder.table(dfsort)
            #         # search is done
            #         st.sidebar.success('分组数量: ' + str(clusternum))
            #         st.sidebar.download_button(label='下载分组结果',
            #                                    data=dfsort.to_csv(),
            #                                    file_name='分组结果.csv',
            #                                    mime='text/csv')
            #     resultdf=dfsort
            # else:
            # resultdf = plcsam
        else:
            st.sidebar.warning("请输入搜索条件")
            resultdf = st.session_state["search_result"]

    elif match == "模糊搜索":
        choicels = searchByIndustrysupa(industry_choice)
        make_choice = st.sidebar.multiselect("选择监管制度:", choicels)

        search_text = st.sidebar.text_area("输入搜索条件")
        # radio to choose whether to use the new keywords
        # use_new_keywords = st.sidebar.radio('精确模式', ('否', '是'))

        # if use_new_keywords == '是':
        #     # get keywords list
        #     keylist = keybert_keywords(search_text)
        #     # convert to string
        #     keyliststr = ' '.join(keylist)

        #     # display keywords_list
        #     new_keywords_str = st.sidebar.text_input('关键词列表：', keyliststr)
        #     # convert to list
        #     new_keywords_list = new_keywords_str.split()

        top = st.sidebar.slider("匹配数量选择", min_value=1, max_value=20, value=3)

        search = st.sidebar.button("搜索条款")

        if search:
            with st.spinner("正在搜索..."):
                # fullresultdf = searchrule(search_text, column_text,
                #                           make_choice, industry_choice,
                #                           top * 5)
                resultdf = similarity_search(
                    search_text, top, industry_choice, make_choice
                )
                # if use_new_keywords == '是':
                #     # proc_list = fullresultdf['条款'].tolist()
                #     proc_list = combine_df_columns(fullresultdf,
                #                                    ['结构', '条款'])

                #     subidx = get_most_similar(new_keywords_list,
                #                               proc_list,
                #                               top_n=top)
                #     # st.write('关键词匹配结果：')
                #     # st.write(subidx)
                #     # get subuploaddf based on index list
                #     resultdf = fullresultdf.iloc[subidx]
                # else:
                # resultdf = fullresultdf[:top]

                # reset index
                # resultdf.reset_index(drop=True, inplace=True)
                placeholder.table(resultdf)
                # st.write(resultdf)
                # search is done
                # st.sidebar.success('搜索完成')
                st.sidebar.success("共搜索到" + str(resultdf.shape[0]) + "条结果")
                st.sidebar.download_button(
                    label="下载搜索结果",
                    data=resultdf.to_csv(),
                    file_name="监管条文搜索结果.csv",
                    mime="text/csv",
                )
        else:
            st.sidebar.warning("请输入搜索条件")
            resultdf = st.session_state["search_result"]

    # st.session_state['search_result'] = resultdf

    # if resultdf is not None and resultdf.shape[0] > 0:
    #     # get proc_list and length
    #     proc_list = resultdf['条款'].tolist()
    #     proc_len = len(proc_list)
    # else:
    #     return

    elif match == "生成模型":
        st.subheader("生成模型")

        name_text = ""
        searchresult, choicels = searchByName(name_text, industry_choice)

        make_choice = st.sidebar.multiselect("选择监管制度:", choicels)

        if make_choice == []:
            make_choice = choicels
        section_list = get_section_list(searchresult, make_choice)
        column_text = st.sidebar.multiselect("选择章节:", section_list)
        if column_text == []:
            column_text = ""
        else:
            column_text = "|".join(column_text)

        fullresultdf, total = searchByItem(searchresult, make_choice, column_text, "")
        # reset index
        fullresultdf = fullresultdf.reset_index(drop=True)
        st.write(fullresultdf)
        # get total number of results
        total = fullresultdf.shape[0]
        st.markdown("共搜索到" + str(total) + "条结果")
        # metadata = fullresultdf[['监管要求','结构']].to_dict(orient="records")
        # st.write(metadata)
        # button to build model
        build_model = st.button("新建模型")
        if build_model:
            with st.spinner("正在生成模型..."):
                build_ruleindex(fullresultdf, industry_choice)
                st.success("模型生成完成")

        add_model = st.button("添加模型")
        if add_model:
            with st.spinner("正在添加模型..."):
                add_ruleindex(fullresultdf, industry_choice)
                st.success("模型添加完成")

    elif match == "智能问答":
        st.subheader("智能问答")

        choicels = searchByIndustrysupa(industry_choice)
        make_choice = st.sidebar.multiselect("选择监管制度:", choicels)

        # input question
        question = st.text_area("输入问题")

        top = st.slider("匹配数量选择", min_value=1, max_value=10, value=4)
        # choose model
        model_name = st.selectbox(
            "选择模型",
            [
                "mistral",
                "qwen:7b",
                "gpt-35-turbo",
                "gpt-35-turbo-16k",
                "gpt-4",
                "gpt-4-32k",
                "gpt-4-turbo",
                "qwen-turbo",
                "qwen-plus",
                "qwen-max",
                "chatglm-6b-v2",
                "chatglm3-6b",
                "baichuan2-13b-chat-v1",
                "ERNIE-Bot-4",
                "ERNIE-Bot-turbo",
                "ChatGLM2-6B-32K",
                "Yi-34B-Chat",
                "Mixtral-8x7B-Instruct",
                "gemini-pro",
                "Baichuan2-Turbo",
                "Baichuan2-Turbo-192K",
                "Baichuan2-53B",
            ],
        )
        rag_type = st.sidebar.selectbox(
            "选择RAG类型", ["similarity", "mmr", "multiquery", "rerank"]
        )
        # button to search
        search = st.button("搜索")
        if search:
            with st.spinner("正在搜索..."):
                # search answer
                answer, source = gpt_answer(
                    question,
                    rag_type,
                    industry_choice,
                    top,
                    model_name,
                    make_choice,
                )
                st.markdown("### 答案：")
                st.write(answer)
                st.markdown("### 来源：")
                st.table(source)

    elif match == "智能对话":
        st.subheader("智能对话")

        choicels = searchByIndustrysupa(industry_choice)
        make_choice = st.sidebar.multiselect("选择监管制度:", choicels)

        top = st.sidebar.slider("匹配数量选择", min_value=1, max_value=10, value=4)
        # choose model
        model_name = st.sidebar.selectbox(
            "选择模型",
            [
                "mistral",
                "qwen:7b",
                "gemma:7b",
                "gpt-35-turbo",
                # "gpt-35-turbo-16k",
                # "gpt-4",
                # "gpt-4-32k",
                "gpt-4-turbo",
                "qwen-turbo",
                "qwen-plus",
                "qwen-max",
                # "chatglm-6b-v2",
                # "chatglm3-6b",
                # "baichuan2-13b-chat-v1",
                "ERNIE-Bot-4",
                "ERNIE-Bot-turbo",
                # "ChatGLM2-6B-32K",
                "Yi-34B-Chat",
                "Mixtral-8x7B-Instruct",
                "gemini-pro",
                "Baichuan2-Turbo",
                "Baichuan2-Turbo-192K",
                "Baichuan2-53B",
                "moonshot-v1-8k",
                "moonshot-v1-32k",
                "moonshot-v1-128k",
                "glm-3-turbo",
                "glm-4",
                "iflytek-spark3",
                "iflytek-spark3.5",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-2.1",
                "claude-instant-1.2",
                "mistral-small-latest",
                "mistral-medium-latest",
                "mistral-large-latest",
            ],
        )
        # choose retrieval type
        retriever_type = st.sidebar.selectbox(
            "选择检索类型", ["similarity", "mmr", "multiquery", "rerank"]
        )
        # choose fusion type yes or no,default is no
        fusion_type = st.sidebar.checkbox("是否融合检索结果", value=False)

        # choose rag type
        rag_type = st.sidebar.selectbox(
            "选择RAG类型", ["basic", "withsource", "stepback", "hyde", "decomposition"]
        )

        # choose agent or not
        agent_type = st.sidebar.checkbox("是否使用代理", value=False)

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # If user inputs a new prompt, generate and draw a new response
        if prompt := st.chat_input("输入问题"):
            # st.chat_message("user").write(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            response = gpt_answer(
                prompt,
                industry_choice,
                top,
                model_name,
                make_choice,
                retriever_type,
                fusion_type,
                rag_type,
                agent_type,
            )
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                # st.markdown(response)
                # outstr = response
                outstr = st.write_stream(response)
                # with st.expander("来源："):
                #     st.table(source)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": outstr})

    elif match == "模型管理":
        st.subheader("模型管理")

        choicels = searchByIndustrysupa(industry_choice)
        make_choice = st.sidebar.multiselect("选择监管制度:", choicels)

        # delete model button
        delete_model = st.sidebar.button("删除模型")
        if delete_model:
            with st.spinner("正在删除模型..."):
                try:
                    delete_db(industry_choice, make_choice)
                    st.success("模型删除成功")
                except Exception as e:
                    st.error(e)
                    st.error("模型删除失败")

        # st.sidebar.subheader("搜索范围")
        # st.sidebar.write(make_choice)
        # save the search result
        # st.write('保存搜索结果')
        # st.write(resultdf)

        # st.write('搜索结果数量：' + str(proc_len))
        # generate the audit result
        # st.sidebar.subheader("自动生成审计程序")
        # # use expender
        # with st.sidebar.expander("参数设置",expanded=False):
        #     gen_num = st.slider('文本数量', min_value=1, max_value=10, value=1)

        #     # choose max length of auditproc
        #     max_length = st.slider('文本长度',
        #                            min_value=25,
        #                            max_value=200,
        #                            value=70)

        #     # choose start and end index
        #     start_idx = st.number_input('选择开始索引',
        #                                 min_value=0,
        #                                 max_value=proc_len - 1,
        #                                 )
        #     # st.write('开始索引：' + str(start_idx))
        #     # change idx to int
        #     start_idx = int(start_idx)
        #     end_idx = st.number_input('选择结束索引',
        #                               min_value=start_idx,
        #                               max_value=proc_len - 1,
        #                               value=proc_len - 1)

        # st.write(start_idx)
        # st.write(end_idx)

        # generate = st.sidebar.button('生成审计程序')

        # if generate:
        #     placeholder.empty()
        #     # read search result from session
        #     resultdf = st.session_state['search_result']
        #     # st.write('搜索结果')
        #     # st.table(resultdf)
        #     # get proc_list and length
        #     proc_list = resultdf['条款'].tolist()
        #     # change idx to int
        #     start_idx = int(start_idx)
        #     end_idx = int(end_idx)
        #     # get proc_list and audit_list
        #     subproc_list = proc_list[start_idx:end_idx + 1]

        #     # split list into batch of 5
        #     batch_num = 5
        #     proc_list_batch = [
        #         subproc_list[i:i + batch_num]
        #         for i in range(0, len(subproc_list), batch_num)
        #     ]

        #     dfls = []
        #     # get proc and audit batch
        #     for j, proc_batch in enumerate(proc_list_batch):

        #         with st.spinner('处理中...'):
        #             # range of the batch
        #             start = j * batch_num + 1
        #             end = start + len(proc_batch) - 1

        #             st.subheader('审计程序: ' + f'{start}-{end}')

        #             # get audit list
        #             audit_batch = predict(proc_batch, 8, gen_num, max_length)
        #             auditls = []
        #             for i, proc in enumerate(proc_batch):
        #                 # audit range with stride x
        #                 audit_start = i * gen_num
        #                 audit_end = audit_start + gen_num
        #                 # get audit list
        #                 audit_list = audit_batch[audit_start:audit_end]
        #                 auditls.append(audit_list)
        #                 count = str(j * batch_num + i + 1)
        #                 # print proc,index and audit list
        #                 st.warning('审计要求 ' + count + ': ' + proc)
        #                 # st.write(proc)
        #                 # print audit list
        #                 st.info('审计程序 ' + count + ': ')
        #                 for audit in audit_list:
        #                     st.write(audit)
        #                 # convert to dataframe
        #             df = pd.DataFrame({'审计要求': proc_batch, '审计程序': auditls})
        #             dfls.append(df)

        #     # conversion is done
        #     st.sidebar.success('处理完成!')
        #     # if dfls not empty
        #     if dfls:
        #         alldf = pd.concat(dfls)
        #         st.sidebar.download_button(label='下载结果',
        #                                    data=alldf.to_csv(),
        #                                    file_name='plc2auditresult.csv',
        #                                    mime='text/csv')
        # clear the session
        # st.session_state['search_result'] = None


if __name__ == "__main__":
    main()
