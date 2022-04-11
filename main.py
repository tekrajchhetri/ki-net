# -*- coding: utf-8 -*-
# @Time    : 18.03.22 10:40
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : main.py
# @Software: PyCharm

from helper import *
from visualisation import *
from graph_learn import *
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config('KI-Net: Optimising Manufacturing Process with Bayesian Learning and Knowledge Graphs')
display_logo()

st.title('Optimising Manufacturing Process with Bayesian Learning and Knowledge Graphs')
st.text("Tek Raj Chhetri")


def start():
    with st.sidebar:
        # https://github.com/streamlit/streamlit/issues/2058
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 600px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 500px;
                margin-left: -500px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader("Upload your file in CSV or HDF format")

    if uploaded_file is not None:
        try:
            file_details = {"name": uploaded_file.name, "type": uploaded_file.type}
            if uploaded_file.name.split(".")[1] == "hdf" or uploaded_file.name.split(".")[1] == "csv":
                saved_file = savefile(uploaded_file)
                if saved_file ==1:
                    st.success("File uploaded, now reading file for display..")
                    dataframe = read_file(uploaded_file, file_details)
                    if dataframe is not None:
                        st.write(dataframe)

                        with st.sidebar:
                            st.header("Bayesian structure learning configuration")
                            st.write("---" * 34)
                            algorithm = select_algorithm()
                            if algorithm == "Notears":

                                st.subheader("Domain Knowledge")

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("Source")
                                    domainsourceSelected = multiselect(options=dataframe.columns, key="domainsourcetabu")
                                with col2:
                                    st.subheader("Destination")
                                    domaindestinationSelected = multiselect(options=dataframe.columns, key="domaindesttabu")
                                st.caption("Selected  Edges")
                                doaminKg = make_edges(domainsourceSelected, domaindestinationSelected)
                                st.code(doaminKg["message"])

                                st.subheader("Tabu Edge(s)")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("Source")
                                    sourceSelected = multiselect(options=dataframe.columns, key="sourcetabu")
                                with col2:
                                    st.subheader("Destination")
                                    destinationSelected = multiselect(options=dataframe.columns, key="desttabu")
                                st.caption("Selected Tabu Edges")
                                tabuEdges = make_edges(sourceSelected, destinationSelected)
                                st.code(tabuEdges["message"])

                                st.write(f'Selected Optimisation Algorithm: {algorithm}')
                                ccol5, ccol6 = st.columns(2)
                                with ccol5:
                                    threshold = slider(label="Selected filter threshold %", min_value=0.0, max_value=1.0, defaultselected=0.4, step=0.01, key="filter_thres")
                                    st.code(f'Threshold: {threshold * 100}%')
                                with ccol6:
                                    max_iter = slider(label="Maximum iterations",
                                                      min_value=0, max_value=10000, defaultselected=100, step=100,
                                                      key="max_iter")
                                    st.code(f"Maximum iterations = {max_iter}")

                                ccol1, ccol2 = st.columns(2)
                                ccol3, ccol4 = st.columns(2)
                                with ccol1:
                                    bias_enable = checkbox("Enable Bias",key="enable_bias")
                                with ccol2:
                                    enable_gpu = checkbox("Enable GPU",key="enable_gpu")
                                with ccol3:
                                    lasso_reg = slider(label="L1 regularisation", min_value=0.0, max_value=1.0, defaultselected=0.1, step=0.01, key="lasso_reg")
                                    st.code(f"L1 (lasso) value = {lasso_reg}")
                                with ccol4:
                                    ridge_reg = slider(label="L2 regularisation", min_value=0.0, max_value=1.0, defaultselected=0.2, step=0.01, key="ridge_reg")
                                    st.code(f"L2 (ridge) value = {ridge_reg}")
                                hidden_unit =  checkbox("Hidden layers",key="hidden_unit")

                                if hidden_unit:
                                    textinput = st.text_input(label="Input hidden layers as a list. For example, if you want 3 hidden layers with the hidden units 3, 5, and 8, you would enter 3,5,8 in the textbox below.",
                                              key="hiddenunitlist")
                                    hiddenlayer = check_hidden_layer_input(input_text=textinput)
                                else:
                                    hiddenlayer=None
                                st.code({f"hiddenlayers = {hiddenlayer}"})
                                st.write("---" * 34)
                                button_clicked = start_structure_learning(text="Start Bayesian Structure Learning Process",
                                                                          key="startlearning")
                            else:
                                pass


                        if "startlearning" in st.session_state and st.session_state["startlearning"]==True:

                            if button_clicked and algorithm == "Notears":
                                st.markdown(
                                    """<hr style="height:1px;border:none;color:#a9a9a9;background-color:#333;" /> """,
                                    unsafe_allow_html=True)
                                st.subheader("Starting Bayesian Structure Learning with following Configuration")
                                sm1, sm2, sm3,sm4,sm5,sm6 = st.columns(6)
                                sm1.metric(label="Threshold", value=f"{threshold * 100}%")
                                sm2.metric("Max. iterations", f"{max_iter}")
                                sm3.metric("Enable Bias", f"{bias_enable}")
                                sm4.metric("Enable GPU", f"{enable_gpu}")
                                sm5.metric("L1 reg", f"{lasso_reg}")
                                sm6.metric("L2 reg", f"{ridge_reg}")

                                st.text(f'Tabu Edges: {tabuEdges["message"]}')
                                st.text(f'Domain Knowledge: {doaminKg["message"]}')
                                st.text(f'Hidden layers: {hiddenlayer}')
                                if doaminKg["status"] == 0:
                                    doaminKg["message"] = None

                                if tabuEdges["status"] == 0:
                                    tabuEdges["message"] = None

                                learned_dag = init_learning_process(dataset=preprocess_data(dataframe),
                                            threshold=float(threshold),
                                            use_bias=bias_enable,
                                            use_gpu=enable_gpu,
                                            max_iter=max_iter,
                                            hidden_layer_units=hiddenlayer,
                                            lasso_beta=lasso_reg,
                                            ridge_beta=ridge_reg,
                                            tabuedge=tabuEdges["message"],
                                            domainknowledge=doaminKg["message"] )

                                if learned_dag == False:
                                    st.error(zero_error())
                                else:
                                    display_learned_graph(learned_dag)
                                    dagdf = to_convert_to_SDOW_format(learned_dag)
                                    dagdf.to_csv("b.csv")
                                    st.write(dagdf)

                            else:
                                st.code({"Select algorithm"})
            else:
                st.alert("Invalid file format, requires either CSV or HDF format")

        except Exception as e:
            error = RuntimeError('Error occured while uploading file.')
            st.exception(e)


if __name__ == '__main__':
    # remove_files()
    start()