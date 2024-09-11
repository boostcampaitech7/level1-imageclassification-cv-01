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

if "test" not in st.session_state:
    st.session_state.test = 0
    st.session_state.train = 0

testd, traind = load_data()

option = st.sidebar.selectbox("데이터 선택",("CSV파일"))

if option == "CSV파일":
    targetdata = st.sidebar.checkbox("트레인 데이터 타겟 설정")
    c1, c2 = st.columns(2)
    c1.header("테스트 데이터")
    testevent = c1.dataframe(testd,selection_mode={"single-row"},on_select='rerun')
    c1.write("테스트 데이터 수: "+str(testd.size)+"개")

    if not targetdata:
        c2.header("트레인 데이터")
        trainevent = c2.dataframe(traind,selection_mode={"single-row"},on_select='rerun')
        c2.write("트레인 데이터 수: "+str(traind.size)+"개")
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
        trainevent = c2.dataframe(traind,selection_mode={"single-row"},on_select='rerun')
        c2.write("타겟 값이 "+str(target)+"인 트레인 데이터 수: "+str(traind["target"].size)+"개")


    if testevent["selection"]["rows"] :
        path = testd.iloc[testevent["selection"]["rows"][0]]['image_path']
        if path != st.session_state.test:
            show_image("test",path)
            st.session_state.test = path

    if trainevent["selection"]["rows"] :
        path = traind.iloc[trainevent["selection"]["rows"][0]]['image_path']
        if path != st.session_state.train:
            show_image("train",path)
            st.session_state.train = path
