# Safe-machine-learning-for-patient-data-privacy
This project has been implemented to evaluate the implications of privacy preserving encryption algorithms on performance of traditional machine learning techniques. The system first encrypts Breast Cancer Diagnostic Dataset from UCI Data Repository using Paillier Homomorphic Encryption which is fed to the data mining models so the prediction process of sensitive data is always maintained in encrypted form.Various algorithms such as KNN, Logistic Regression, SVC, LBFGS, Naive Bayes, Decision Tree, Bagged Decision Tree,Random Forest, Extra Trees, AdaBoost, Stochastic Gradient Boosting and Stochastic Voting Ensemble are applied to verify initial hyothesis. The output of the prediction system is then provided to the medical practitioner who already has the private key for decryption which ensures data privacy. It was observed that for the , Logistic Regression outperformed all other implemented models without compromising data privacy.
