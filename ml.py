"""
#####################################################
from: https://github.com/BioSystemsUM/propythia
####################################################
"""

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from optimizer import *
from parameters import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
import pickle
import os
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier


class ML:
    def __init__(self, X_train, X_test, y_train, y_test, report_name=None):
        """
        init function. When the class is called a dataset containing the features values and a target column must
        be provided.
        :param x_train: dataset with features or encodings for training
        :param x_test: dataset with features or encodings for evaluation
        :param y_train: class labels for training
        :param y_test: class labels for testing
        :param report_name: str. If not none it will generate a report txt with the name given) with results by functions
        called within class. Default None
        :param columns_names: Names of columns. important if features importance want to be analysed. None by default.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = None
        self.report_name = report_name
        if self.report_name:
            self._report(str(self.report_name))
        self.final_units = len(np.unique(y_train))
        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), 'data'))

    def _report(self, info, dataframe=False, **kwargs):
        filename = str(self.report_name)
        with open(filename, 'a+') as file:
            if dataframe is True:
                info.to_csv(file, sep='\t', mode='a', **kwargs)
            elif isinstance(info, str):
                file.writelines(info)
            else:
                for l in info:
                    file.writelines('\n{}'.format(l))

    def train_best_model(self, model_name, model, scaler='StandardScaler()', score=make_scorer(matthews_corrcoef),
                         cv=10, optType='gridSearch', param_grid=None, n_jobs=10, random_state=42, n_iter=15,
                         refit=True, **params):
        """
        This function performs a parameter grid search or randomizedsearch on a selected classifier model and training data set.
        It returns a scikit-learn pipeline that performs standard scaling (if not None) and contains the best model found by the
        grid search according to the Matthews correlation coefficient or other given metric.
        :param model_name: {str} model to train. Choose between 'svm', 'linear_svm', 'knn', 'sgd', 'lr','rf', 'gnb', 'nn','gboosting'
        :param model: scikit learn model
        :param scaler: {scaler} scaler to use in the pipe to scale data prior to training (integrated  in pipeline)
         Choose from
            ``sklearn.preprocessing``, e.g. 'StandardScaler()', 'MinMaxScaler()', 'Normalizer()' or None.None by default.
        :param score: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
            (choose from the scikit-learn`scoring-parameters <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_).
        :param optType: type of hyperparameter optimization to do . 'gridSearch' or 'randomizedSearch'
        :param param_grid: {dict} parameter grid for the grid or randomized search
        (see`sklearn.grid_search <http://scikit-learn.org/stable/modules/model_evaluation.html>`_).
        :param cv: {int} number of folds for cross-validation.
        :param n_iter: number of iterations when using randomizedSearch optimizaton. default 15.
        :param n_jobs: {int} number of parallel jobs to use for calculation. if ``-1``, all available cores are used.
        10 by default.
        :param refit: wether to refit the model accordingly to the best model . Default True.
        :param random_state: random_state 1 by default.
        :param **params: params to integrate in scikit models
        :return: best classifier fitted to training data.
        """
        #start = time.clock()
        print("performing {}...".format(optType))

        saved_args = locals()
        if model_name:
            model = model_name.lower()
        if model is not None:
            model = model

        if model == 'svm':
            pipe_model = Pipeline([('scl', scaler),
                                   ('selector', SelectKBest(f_classif)),
                                   ('clf', SVC(random_state=random_state, probability=True, **params))])

        elif model == 'rf':
            pipe_model = Pipeline([('scl', scaler),
                                   ('selector', SelectKBest(f_classif)),
                                   ('clf', RandomForestClassifier(random_state=random_state, **params))])

        elif model == 'gboosting':
            pipe_model = Pipeline([('scl', scaler),
                                   ('selector', SelectKBest(f_classif)),
                                   ('clf', GradientBoostingClassifier(random_state=random_state, **params))])

        elif model == 'lr':
            pipe_model = Pipeline([('scl', scaler),
                                   ('selector', SelectKBest(f_classif)),
                                   ('clf', LogisticRegression(random_state=random_state, **params))])

        elif model == 'nn':
            pipe_model = Pipeline([('scl', scaler),
                                   ('selector', SelectKBest(f_classif)),
                                   ('clf', MLPClassifier(**params))])

        elif model == 'xgboost':
            pipe_model = Pipeline([('scl', scaler),
                                   ('selector', SelectKBest(f_classif)),
                                   ('clf', XGBClassifier(random_state=random_state, **params))])

        elif model is str:
            # keras classifier
            print("Model not supported, please choose between 'svm', 'knn', 'sgd', 'rf', 'gnb', 'nn', 'gboosting' ")
            return
        else:
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', model(**params))])

        # retrieve default grids
        if param_grid is None:
            param = param_shallow()
            if optType == 'gridSearch':
                param_grid = param[model.lower()]['param_grid']
            else:
                param_grid = param[model.lower()]['distribution']

        po = ParamOptimizer(optType=optType, estimator=pipe_model, paramDic=param_grid,
                            dataX=self.X_train, datay=self.y_train,
                            cv=cv, n_iter_search=n_iter, n_jobs=n_jobs, scoring=score, model_name=self.model_name,
                            refit=refit)
        gs = po.get_opt_params()

        # # summaries
        list_to_write_top_3_models = po.report_top_models(gs)

        s1, s2, s3, df = po.report_all_models(gs)  # metrics for the best model. dataframe with all
        # except:
        #     print('not print all models')
        # Set the best parameters to the best estimator
        best_classifier = gs.best_estimator_
        best_classifier_fit = best_classifier.fit(self.X_train, self.y_train)
        self.classifier = best_classifier_fit
        #final = time.clock()
        #run_time = final - start

        # write report
        if self.report_name is not None:
            self._report(['===TRAIN MODELS===\n', self.train_best_model.__name__, saved_args])
            self._report(list_to_write_top_3_models)
            self._report([s1, s2, s3, f"Finished {self.train_best_model.__name__}"])
            self._report(df, dataframe=True, float_format='%.3f')

        print('Writing model...')
        pickle.dump(best_classifier_fit, open(str(model), 'wb'))

        return best_classifier_fit

    ####################################################################################################################
    # EVALUATE
    ####################################################################################################################
    def conf_matrix_seaborn_table(self, conf_matrix=None, classifier=None, path_save='', show=True,
                                  square=True, annot=True, fmt='d', cbar=False, **params):
        plt.clf()
        if conf_matrix is None:
            y_pred = classifier.predict(self.X_test)
            mat = confusion_matrix(self.y_test, y_pred)
        else:
            mat = conf_matrix

        sns.heatmap(mat.T, square=square, annot=annot, fmt=fmt, cbar=cbar, **params)
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        if path_save is not None:
            plt.savefig(fname=path_save)
        if show is True:
            plt.show()
        plt.clf()

    def score_testset(self, classifier=None):
        """
        Returns the tests set scores for the specified scoring metrics in a ``pandas.DataFrame``. The calculated metrics
        are Matthews correlation coefficient, accuracy, precision, recall, f1 and area under the Receiver-Operator Curve
        (roc_auc). See `sklearn.metrics <http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics>`_
        for more information.
        :param classifier: {classifier instance} pre-trained classifier used for predictions. If svm2802, will use the
        one trained in the class.
        :return: ``pandas.DataFrame`` containing the cross validation scores for the specified metrics.
        """
        if classifier is None:  # priority for the classifier in function
            classifier = self.classifier
        saved_args = locals()
        cm2 = []
        scores = {}
        y_pred = classifier.predict(self.X_test)
        try:
            y_prob = classifier.predict_proba(self.X_test)
        except:
            y_prob = None
        scores['Accuracy'] = accuracy_score(self.y_test, y_pred)
        scores['MCC'] = matthews_corrcoef(self.y_test, y_pred)
        if y_prob is not None:
            scores['log_loss'] = log_loss(self.y_test, y_prob)

        if self.final_units > 2:
            # multiclass
            scores['f1 score weighted'] = f1_score(self.y_test, y_pred, average='weighted')
            scores['f1 score macro'] = f1_score(self.y_test, y_pred, average='macro')
            scores['f1 score micro'] = f1_score(self.y_test, y_pred, average='micro')
            if y_prob is not None:
                scores['roc_auc ovr'] = roc_auc_score(self.y_test, y_prob, average='weighted', multi_class='ovr')
                y_test = self.y_test
                # y_test = y_test.reshape(y_test.shape[0])  # roc auc ovo was giving error
                scores['roc_auc ovo'] = roc_auc_score(y_test, y_prob, average='weighted', multi_class='ovo')
            scores['precision'] = precision_score(self.y_test, y_pred, average='weighted')
            scores['recall'] = recall_score(self.y_test, y_pred, average='weighted')
            cm2 = multilabel_confusion_matrix(self.y_test, y_pred)

        else:
            # binary
            scores['f1 score'] = f1_score(self.y_test, y_pred)
            scores['roc_auc'] = roc_auc_score(self.y_test, y_pred)
            #precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred)
            scores['Precision'] = precision_score(self.y_test, y_pred)
            scores['Recall'] = recall_score(self.y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            scores['fdr'] = float(fp) / (tp + fp)
            scores['sn'] = float(tp) / (tp + fn)
            scores['sp'] = float(tn) / (tn + fp)

        report = classification_report(self.y_test, y_pred, output_dict=False)
        cm = confusion_matrix(self.y_test, y_pred)
        cm2 = None

        if self.report_name is not None:
            self._report(['===SCORING TEST SET ===\n', self.score_testset.__name__, saved_args, 'report\n', report,
                          '\nconfusion_matrix\n', cm, '\nmultilabel confusion matrix\n', cm2, '\nscores report\n'])
            scores_df = pd.DataFrame(scores.keys(), columns=['metrics'])
            scores_df['scores'] = scores.values()
            self._report(scores_df, dataframe=True, float_format='%.4f', index=False)

            self.conf_matrix_seaborn_table(conf_matrix=cm, path_save=str(self.report_name + 'confusion_matrix.png'),
                                           show=False)

        return scores, report, cm, cm2

    def plot_roc_curve(self, classifier=None, ylim=(0.0, 1.00), xlim=(0.0, 1.0),
                       title='Receiver operating characteristic (ROC) curve',
                       path_save='plot_roc_curve', show=True):
        """
        Function to plot a ROC curve
        On the y axis, true positive rate and false positive rate on the X axis.
        The top left corner of the plot is the 'ideal' point - a false positive rate of zero, and a true positive rate
        of one, meaning a larger area under the curve (AUC) is usually better.
        :param classifier: {classifier instance} pre-trained classifier used for predictions.
        :param ylim: y-axis limits
        :param xlim: x- axis limits
        :param title: title of plot. 'Receiver operating characteristic (ROC) curve' by default.
        :param path_save: path to save the plot. If svm2802 , lot is not saved. 'plot_roc_curve' by default.
        :param show: Whether to display the graphic. False by default.
        :return:
        Needs classifier with probability
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        """
        lw = 2
        if classifier is None:
            classifier = self.classifier

        # binary
        if self.final_units <= 2:
            #y_score = classifier.predict(self.X_test)
            y_score = classifier.predict_proba(self.X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(self.y_test, y_score)
            roc_auc = auc(fpr, tpr)
            print(roc_auc)
            plt.show()
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right", )

        if path_save is not None:
            plt.savefig(fname=path_save)
        if show is True:
            plt.show()
        plt.clf()


