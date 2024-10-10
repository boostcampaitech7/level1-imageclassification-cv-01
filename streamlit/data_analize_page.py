import streamlit as st
import pandas as pd

st.set_page_config(page_title="데이터 분석 및 개별 이미지 확인용", layout="wide")

# 최초에 데이터 불러와서 저장
@st.cache_data
def load_data():
    test_data = pd.read_csv('../data/test.csv')
    train_data = pd.read_csv('../data/train.csv')
    return test_data, train_data

@st.cache_data
def load_output_data():
    import os
    # ../result 폴더에 있는 모든 csv 파일 가져오는 코드
    csv_files = []
    for root, dirs, files in os.walk('../result'):
        csv_files.extend([os.path.join(root, file) for file in files if file.endswith('.csv')])

    output_data = {}
    for file in csv_files:
        df = pd.read_csv(file)
        if 'epoch' in df.columns: continue
        output_data[file] = df

    return output_data

# 데이터 페이지 단위로 스플릿
@st.cache_data(show_spinner=False)
def split_frame(input_df, rows):
    df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df

# 팝업창 띄우기
@st.dialog("image")
def show_image(type,path):
    pass

# 페이지에 있는 이미지 출력
def show_images(type, img_pathes, window):
    cols = window.columns(5)
    for idx,path in enumerate(img_pathes.values):
        cols[idx%5].image(type+path)
        cols[idx%5].write(path)

# 데이터 프레임 페이지 단위로 출력
def show_dataframe(dataset,window,type):
    # 가장 윗부분 데이터 정렬할 지 선택, 정렬 시 무엇으로 정렬할지, 오름차순, 내림차순 선택
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
    # 데이터 크기 출력
    total_data = dataset.shape
    with top_menu[0]:
        st.write("data_shape: "+str(total_data))
    con1,con2 = window.columns(2)

    # 아래 부분 페이지당 데이터 수, 페이지 선택
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

# 원본데이터 확인 가능 아웃풋도 확인하도록 할 수 있을 듯?
option = st.sidebar.selectbox("데이터 선택",("원본 데이터", "결과 데이터"))

if option == "원본 데이터":
    # 데이터 로드
    testd, traind = load_data()
    if "path" not in st.session_state:
        st.session_state.path = [0,0]
    
    # 테스트 데이터 출력
    st.header("테스트 데이터")
    page = show_dataframe(testd,st,'../data/test/')

    # 트레인 데이터 출력
    ## 타겟 별로 출력할지 설정
    targetdata = st.sidebar.checkbox("트레인 데이터 타겟 설정")
    if not targetdata:
        # 전체 출력
        st.header("트레인 데이터")
        show_dataframe(traind,st,'../data/train/')
        traintargetcount = traind["target"].value_counts().sort_index()
        # 분포 확인
        st.header("트레인 데이터 타겟 값 분포")
        c1, c2 = st.columns([1,7])
        c1.dataframe(traintargetcount, height=350)
        c2.line_chart(traintargetcount, height=350)

        traincountcount = traintargetcount.value_counts().sort_index().rename('coc')
        c1.dataframe(traincountcount, height=350)
        c2.bar_chart(traincountcount, height=350)
    else:
        # 타겟 별 출력 시 어떤 타겟 출력할 것인지 선택
        target = st.sidebar.number_input("타겟",min_value=traind['target'].min(),max_value=traind['target'].max(), step=1)

        traind = traind.loc[traind.target == target].reset_index()
        st.header("트레인 데이터")
        show_dataframe(traind,st,'../data/train/')
# output 파일 체크
elif option == "결과 데이터":
    output_data = load_output_data()

    #사이드 바에서 결과 데이터 중 csv 파일 하나를 선택
    selected_file = st.sidebar.selectbox("CSV File", output_data.keys())
    if 'class_name' in output_data[selected_file].columns:
        folder = '../data/train/'
    else: folder = '../data/test/'
    outdata = output_data[selected_file][['image_path', 'target']]

    # 데이터 출력
    st.header(selected_file)

    targetdata = st.sidebar.checkbox("아웃풋 데이터 타겟 설정")
    if not targetdata:
        # 아웃풋 데이터 기준으로 분포 확인
        show_dataframe(outdata,st,folder)
        outtargetcount = outdata["target"].value_counts().sort_index()
        st.header("아웃풋 데이터 타겟 값 분포")
        c1, c2 = st.columns([2,7])
        c1.dataframe(outtargetcount, height=350)
        c2.line_chart(outtargetcount, height=350)

        outcountcount = outtargetcount.value_counts().sort_index().rename('coc')
        c1.dataframe(outcountcount, height=350)
        c2.bar_chart(outcountcount, height=350)
    else:
        # 아웃풋 데이터 기준으로 타겟 별로 출력
        target = st.sidebar.number_input("타겟",min_value=outdata['target'].min(),max_value=outdata['target'].max(), step=1)
        outdata = outdata.loc[outdata.target == target].reset_index(drop=True)
        show_dataframe(outdata,st,folder)