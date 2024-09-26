import streamlit as st
import pandas as pd

st.set_page_config(page_title="데이터 분석 및 개별 이미지 확인용", layout="wide")

@st.cache_data
def load_data():
    test_data = pd.read_csv('../data/test.csv')
    train_data = pd.read_csv('../data/train.csv')
    return test_data, train_data

@st.cache_data(show_spinner=False)
def split_frame(input_df, rows):
    df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df

@st.dialog("image")
def show_image(type,path):
    if type == "test":
        st.header("test image: "+path)
        st.image("../data/test/"+path)
    elif type == "train":
        st.header("train image: "+path)
        st.image("../data/train/"+path)

def show_images(type, img_pathes, window):
    cols = window.columns(5)
    for idx,path in enumerate(img_pathes.values):
        cols[idx%5].image(f"../data/{type}/"+path)
        cols[idx%5].write(path)

def show_dataframe(dataset,window,type):
    top_menu = window.columns(3)
    with top_menu[0]:
        sort = st.radio("Sort Data", options=["Yes", "No"], horizontal=1, index=1, key=[type,window,1])
    if sort == "Yes":
        with top_menu[1]:
            sort_field = st.selectbox("Sort By", options=dataset.columns, key=[type,window,2])
        with top_menu[2]:
            sort_direction = st.radio(
                "Direction", options=["⬆️", "⬇️"], horizontal=True
            )
        dataset = dataset.sort_values(
            by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True
        )
    con1,con2 = window.columns(2)

    bottom_menu = window.columns((4, 1, 1))
    with bottom_menu[2]:
        batch_size = st.selectbox("Page Size", options=[10, 25, 50], key=[type,window,3])
    with bottom_menu[1]:
        total_pages = (
            int(len(dataset) / batch_size) if int(len(dataset) / batch_size) > 0 else 1
        )
        current_page = st.number_input(
            "Page", min_value=1, max_value=total_pages, step=1
        )
    with bottom_menu[0]:
        st.markdown(f"Page **{current_page}** of **{total_pages}** ")
    pages = split_frame(dataset, batch_size)
    con1.dataframe(data=pages[current_page - 1], use_container_width=True)
    show_images(type, pages[current_page - 1]['image_path'], con2)
    return pages[current_page - 1]

option = st.sidebar.selectbox("데이터 선택",("CSV파일"))

if option == "CSV파일":
    testd, traind = load_data()
    if "path" not in st.session_state:
        st.session_state.path = [0,0]

    targetdata = st.sidebar.checkbox("트레인 데이터 타겟 설정")
    c1, c2 = st.columns(2)
    st.header("테스트 데이터")
    page = show_dataframe(testd,st,'test')
    #show_images('test', page['image_path'], c2)

    if not targetdata:
        st.header("트레인 데이터")
        show_dataframe(traind,st,'train')
        traintargetcount = traind["target"].value_counts().sort_index()
        st.header("트레인 데이터 타겟 값 분포")
        c1, c2 = st.columns([1,7])
        c1.dataframe(traintargetcount, height=350)
        c2.line_chart(traintargetcount, height=350)

        traincountcount = traintargetcount.value_counts().sort_index().rename('coc')
        c1.dataframe(traincountcount, height=350)
        c2.bar_chart(traincountcount, height=350)
    else:
        target = st.sidebar.number_input("트레인 데이터 타겟 설정",min_value=traind['target'].min(),max_value=traind['target'].max(), step=1)

        traind = traind.loc[traind.target == target].reset_index()
        st.header("트레인 데이터")
        show_dataframe(traind,st,'train')
