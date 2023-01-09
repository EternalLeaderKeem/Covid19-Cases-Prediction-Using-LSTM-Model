# Covid19 Cases Prediction Using LSTM Model
 
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

### Summary
This project is to train a model by using the cases malaysia train dataset of daily covid19 cases in Malaysia from 25th of January 2020 to 4th of December 2020 which later tested on the covid malaysia test dataset to see the accuracy of the model in making predictions. The goal of the project is to get an error of below 1% which sadly was not achieved as the lowest error is at 15%. Improvements on the model can still be done so that the target can be achieved.

### Model architecture
![image](https://user-images.githubusercontent.com/121662880/211287390-02854520-1568-4e0d-92e2-28f23827e430.png)

### Loss graph during model training
![image](https://user-images.githubusercontent.com/121662880/211287031-f21e77de-8728-43f1-bf32-a9c6a1218410.png)

### Graph of predicted and actual covid cases
![image](https://user-images.githubusercontent.com/121662880/211288899-d6d41551-e6d9-4c89-ae61-5f0fdc315bbd.png)


### Classification report and Confusion matrix
![image](https://user-images.githubusercontent.com/121662880/211288297-0528035f-d17a-4a80-9fcd-0d75b5eceb3a.png)
![image](https://user-images.githubusercontent.com/121662880/211288632-c486beca-1fc6-4eab-8307-b283a30aba37.png)

### Aknowledgement
The dataset is obtained from --> GitHub - MoH-Malaysia/covid19-public: Official data on the COVID-19 
epidemic in Malaysia. Powered by CPRC, CPRC Hospital System, 
MKAK, and MySejahtera.
