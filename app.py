import streamlit as st

from checkrule import searchByName, searchByItem, searchrule
from utils import get_folder_list, get_section_list, items2cluster

rulefolder = 'rules'


def main():

    st.subheader("监管制度搜索")
    industry_list = get_folder_list(rulefolder)

    industry_choice = st.sidebar.selectbox('选择行业:', industry_list)

    if industry_choice != '':

        name_text = ''
        searchresult, choicels = searchByName(name_text, industry_choice)

        make_choice = st.sidebar.multiselect('选择监管制度:', choicels)

        if make_choice == []:
            make_choice = choicels
        section_list = get_section_list(searchresult, make_choice)
        column_text = st.sidebar.multiselect('选择章节:', section_list)
        if column_text == []:
            column_text = ''
        else:
            column_text = '|'.join(column_text)

        match = st.sidebar.radio('搜索方式', ('关键字搜索', '模糊搜索'))

        if match == '关键字搜索':
            item_text = st.sidebar.text_input('按条文关键字搜索')
            if column_text != '' or item_text != '':
                plcsam, total = searchByItem(searchresult, make_choice,
                                             column_text, item_text)

                # placeholder
                placeholder = st.empty()
                # st.table(plcsam)
                placeholder.table(plcsam)
                # search is done
                st.sidebar.success('搜索完成')
                st.sidebar.write('共搜索到' + str(total) + '条结果')
                st.sidebar.download_button(label='下载结果',
                                           data=plcsam.to_csv(),
                                           file_name='监管制度搜索结果.csv',
                                           mime='text/csv')

                st.sidebar.subheader("聚类分组")
                s = st.sidebar.slider('分组阈值', max_value=20, value=10)
                thresholdsort = s / 10
                st.sidebar.write('分组阈值', thresholdsort)
                # cluster the result
                cluster = st.sidebar.button('聚类分组')
                if cluster:
                    # empty the table
                    placeholder.empty()
                    with st.spinner('正在分组...'):
                        dfsort, clusternum = items2cluster(
                            plcsam, thresholdsort)
                        placeholder.table(dfsort)
                        # search is done
                        st.sidebar.success('分组数量: ' + str(clusternum))
                        st.sidebar.download_button(label='下载分组结果',
                                                   data=dfsort.to_csv(),
                                                   file_name='分组结果.csv',
                                                   mime='text/csv')

        elif match == '模糊搜索':
            search_text = st.sidebar.text_area('输入搜索条件')
            top = st.sidebar.slider('匹配数量选择',
                                    min_value=1,
                                    max_value=10,
                                    value=3)

            search = st.sidebar.button('搜索条款')

            if search:
                with st.spinner('正在搜索...'):
                    resuledf = searchrule(search_text, column_text,
                                          make_choice, industry_choice, top)
                    st.table(resuledf)
                    # search is done
                    st.sidebar.success('搜索完成')
                    st.sidebar.write('共搜索到' + str(resuledf.shape[0]) + '条结果')
                    st.sidebar.download_button(label='下载结果',
                                               data=resuledf.to_csv(),
                                               file_name='监管条文搜索结果.csv',
                                               mime='text/csv')
        st.sidebar.subheader("搜索范围")
        st.sidebar.write(make_choice)


if __name__ == '__main__':
    main()