# -*- coding: utf-8 -*-
# @Time    : 18.03.22 10:40
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : main.py
# @Software: PyCharm

from helper import *
from visualisation import *
from bayesian_structure_learning import *
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config('KI-Net: Optimising Manufacturing Process with Bayesian Learning and Knowledge Graphs')
display_logo()

st.title('Optimising Manufacturing Process with Bayesian Learning and Knowledge Graphs')
st.text("Tek Raj Chhetri")
def file_upload():
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
                        st.header("Bayesian structure learning configuration")
                        algorithm = select_algorithm()
                        if algorithm != "Select Algorithm":
                            st.write(f'Selected Optimisation Algorithm: {algorithm}')
                            threshold = threshold_param()
                            st.write(f'Selected filter threshold: {threshold*100}%')
                            button_clicked = start_structure_learning()
                            if button_clicked:
                                st.success("Starting Bayesian Structure Learning Process...")
                                if algorithm == "NotearsLinear":
                                    init_learning_process(datasets=dataframe, threshold=float(threshold))



                                elif algorithm == "NotearsMLP":
                                    pass
            else:
                st.alert("Invalid file format, requires either CSV or HDF format")

        except Exception as e:
            error = RuntimeError('Error occured while uploading file.')
            st.exception(e)


if __name__ == '__main__':
    remove_files()
    file_upload()