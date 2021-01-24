#Impoting the GUI module
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog1(object):

#Method for setting up the UI
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(624, 633)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(40, 30, 511, 101))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(60, 130, 521, 431))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        # self.label_4 = QtWidgets.QLabel(self.widget)
        # font = QtGui.QFont()
        # font.setPointSize(10)
        # self.label_4.setFont(font)
        # self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        # self.label_4.setObjectName("label_4")
        # self.gridLayout.addWidget(self.label_4, 2, 0, 1, 1)

        self.label_2 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.lbglcmrf = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbglcmrf.setFont(font)
        self.lbglcmrf.setAlignment(QtCore.Qt.AlignCenter)
        self.lbglcmrf.setObjectName("lbglcmrf")
        self.gridLayout.addWidget(self.lbglcmrf, 1, 1, 1, 1)


        # self.glcmxt = QtWidgets.QLabel(self.widget)
        # font = QtGui.QFont()
        # font.setPointSize(10)
        # self.glcmxt.setFont(font)
        # self.glcmxt.setAlignment(QtCore.Qt.AlignCenter)
        # self.glcmxt.setObjectName("glcmxt")
        # self.gridLayout.addWidget(self.glcmxt, 2, 1, 1, 1)



        self.label_8 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 6, 0, 1, 1)
        self.glcmgb = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.glcmgb.setFont(font)
        self.glcmgb.setAlignment(QtCore.Qt.AlignCenter)
        self.glcmgb.setObjectName("glcmgb")
        self.gridLayout.addWidget(self.glcmgb, 4, 1, 1, 1)
        self.cnn = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.cnn.setFont(font)
        self.cnn.setAlignment(QtCore.Qt.AlignCenter)
        self.cnn.setObjectName("cnn")
        self.gridLayout.addWidget(self.cnn, 6, 1, 1, 1)


        # self.label_7 = QtWidgets.QLabel(self.widget)
        # font = QtGui.QFont()
        # font.setPointSize(10)
        # self.label_7.setFont(font)
        # self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        # self.label_7.setObjectName("label_7")
        # self.gridLayout.addWidget(self.label_7, 3, 0, 1, 1)


        # self.lbglcmxt = QtWidgets.QLabel(self.widget)
        # font = QtGui.QFont()
        # font.setPointSize(10)
        # self.lbglcmxt.setFont(font)
        # self.lbglcmxt.setAlignment(QtCore.Qt.AlignCenter)
        # self.lbglcmxt.setObjectName("lbglcmxt")
        # self.gridLayout.addWidget(self.lbglcmxt, 3, 1, 1, 1)



        self.label_6 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 4, 0, 1, 1)
        self.lbglcmgb = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbglcmgb.setFont(font)
        self.lbglcmgb.setAlignment(QtCore.Qt.AlignCenter)
        self.lbglcmgb.setObjectName("lbglcmgb")
        self.gridLayout.addWidget(self.lbglcmgb, 5, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 5, 0, 1, 1)
        self.glcmrf = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.glcmrf.setFont(font)
        self.glcmrf.setAlignment(QtCore.Qt.AlignCenter)
        self.glcmrf.setObjectName("glcmrf")
        self.gridLayout.addWidget(self.glcmrf, 0, 1, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Training_results"))
        self.label.setText(_translate("Dialog", "Accuracy of Different Algorithms"))


        # self.label_4.setText(_translate("Dialog", "GLCM + Extra Trees Classifier"))


        self.label_2.setText(_translate("Dialog", "GLCM + Random Forest"))
        self.label_3.setText(_translate("Dialog", "LBGLCM + Random Forest"))
        self.lbglcmrf.setText(_translate("Dialog", ""))


        # self.glcmxt.setText(_translate("Dialog", ""))


        self.label_8.setText(_translate("Dialog", "CNN"))
        self.glcmgb.setText(_translate("Dialog", ""))
        self.cnn.setText(_translate("Dialog", ""))


        # self.label_7.setText(_translate("Dialog", "LBGLCM + Extra Trees Classifier"))
        # self.lbglcmxt.setText(_translate("Dialog", ""))


        self.label_6.setText(_translate("Dialog", "GLCM + Gradient Boosting"))
        self.lbglcmgb.setText(_translate("Dialog", ""))
        self.label_5.setText(_translate("Dialog", "LBGLCM + Gradient Boosting"))
        self.glcmrf.setText(_translate("Dialog", ""))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog1()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
