import streamlit as st
import pandas as pd

st.set_page_config(page_title="데이터 분석 및 개별 이미지 확인용", layout="wide")

@st.cache_data
def load_data():
    test_data = pd.read_csv('../data/test.csv')
    train_data = pd.read_csv('../data/train.csv')
    return test_data, train_data

@st.dialog("image")
def show_image(type,path):
    if type == "test":
        st.header("test image: "+path)
        st.image("../data/test/"+path)
    elif type == "train":
        st.header("train image: "+path)
        st.image("../data/train/"+path)

option = st.sidebar.selectbox("데이터 선택",("CSV파일"))

if option == "CSV파일":
    testd, traind = load_data()
    if "path" not in st.session_state:
        st.session_state.path = [0,0]

    targetdata = st.sidebar.checkbox("트레인 데이터 타겟 설정")
    c1, c2 = st.columns(2)
    c1.header("테스트 데이터")
    testevent = c1.dataframe(testd,selection_mode={"single-row"},on_select='rerun')["selection"]["rows"]
    c1.write("테스트 데이터 수: "+str(testd.shape[0])+"개")

    if not targetdata:
        c2.header("트레인 데이터")
        trainevent = c2.dataframe(traind,selection_mode={"single-row"},on_select='rerun')["selection"]["rows"]
        c2.write("트레인 데이터 수: "+str(traind.shape[0])+"개")
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

        traind = traind.loc[traind.target == target]
        c2.header("트레인 데이터")
        trainevent = c2.dataframe(traind,selection_mode={"single-row"},on_select='rerun')["selection"]["rows"]
        c2.write("타겟 값이 "+str(target)+"인 트레인 데이터 수: "+str(traind.shape[0])+"개")

path = 0
if testevent and st.session_state.path[0] != testd.loc[testevent[0]]["image_path"]:
    path = st.session_state.path[0] = testd.loc[testevent[0]]["image_path"]
    type = "test"
if trainevent and st.session_state.path[1] != traind.loc[trainevent[0]]["image_path"]:
    path = st.session_state.path[1] = traind.loc[trainevent[0]]["image_path"]
    type = "train"
if path:
    show_image(type,path)