# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'connect.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1018, 654)
        MainWindow.setStyleSheet("background-color: rgb(116, 179, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.checkBox_4 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_4.setGeometry(QtCore.QRect(650, 350, 171, 51))
        self.checkBox_4.setMinimumSize(QtCore.QSize(0, 40))
        self.checkBox_4.setMaximumSize(QtCore.QSize(200, 16777215))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.checkBox_4.setFont(font)
        self.checkBox_4.setStyleSheet("font: 12pt \"幼圆\";")
        self.checkBox_4.setAutoExclusive(True)
        self.checkBox_4.setObjectName("checkBox_4")
        self.comboBox_robot_style = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_robot_style.setGeometry(QtCore.QRect(180, 350, 151, 51))
        self.comboBox_robot_style.setMinimumSize(QtCore.QSize(0, 40))
        self.comboBox_robot_style.setMaximumSize(QtCore.QSize(200, 16777215))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.comboBox_robot_style.setFont(font)
        self.comboBox_robot_style.setStyleSheet("font: 12pt \"幼圆\";")
        self.comboBox_robot_style.setObjectName("comboBox_robot_style")
        self.comboBox_robot_style.addItem("")
        self.comboBox_robot_style.addItem("")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(360, 350, 261, 50))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setMinimumSize(QtCore.QSize(0, 50))
        self.pushButton.setStyleSheet("font: 16pt \"幼圆\";")
        self.pushButton.setObjectName("pushButton")
        self.label_stacon_tip = QtWidgets.QLabel(self.centralwidget)
        self.label_stacon_tip.setGeometry(QtCore.QRect(11, 785, 16, 40))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_stacon_tip.sizePolicy().hasHeightForWidth())
        self.label_stacon_tip.setSizePolicy(sizePolicy)
        self.label_stacon_tip.setMinimumSize(QtCore.QSize(0, 40))
        self.label_stacon_tip.setStyleSheet("font: 10pt \"幼圆\";\n"
"color: rgb(54, 49, 46);")
        self.label_stacon_tip.setText("")
        self.label_stacon_tip.setAlignment(QtCore.Qt.AlignCenter)
        self.label_stacon_tip.setObjectName("label_stacon_tip")
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        self.label_title.setGeometry(QtCore.QRect(11, 29, 1001, 100))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_title.sizePolicy().hasHeightForWidth())
        self.label_title.setSizePolicy(sizePolicy)
        self.label_title.setMinimumSize(QtCore.QSize(0, 100))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(28)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_title.setFont(font)
        self.label_title.setStyleSheet("font: 28pt \"幼圆\";\n"
"")
        self.label_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title.setObjectName("label_title")
        self.label_QR = QtWidgets.QLabel(self.centralwidget)
        self.label_QR.setGeometry(QtCore.QRect(740, 430, 200, 200))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_QR.sizePolicy().hasHeightForWidth())
        self.label_QR.setSizePolicy(sizePolicy)
        self.label_QR.setMinimumSize(QtCore.QSize(200, 200))
        self.label_QR.setMaximumSize(QtCore.QSize(200, 200))
        self.label_QR.setMouseTracking(False)
        self.label_QR.setStyleSheet("")
        self.label_QR.setText("")
        self.label_QR.setObjectName("label_QR")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setGeometry(QtCore.QRect(190, 260, 458, 40))
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.label_wifi = QtWidgets.QLabel(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_wifi.sizePolicy().hasHeightForWidth())
        self.label_wifi.setSizePolicy(sizePolicy)
        self.label_wifi.setStyleSheet("font: 10pt \"幼圆\";")
        self.label_wifi.setAlignment(QtCore.Qt.AlignCenter)
        self.label_wifi.setObjectName("label_wifi")
        self.lineEdit_wifissid = QtWidgets.QLineEdit(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_wifissid.sizePolicy().hasHeightForWidth())
        self.lineEdit_wifissid.setSizePolicy(sizePolicy)
        self.lineEdit_wifissid.setMaximumSize(QtCore.QSize(16777215, 40))
        self.lineEdit_wifissid.setFrame(True)
        self.lineEdit_wifissid.setObjectName("lineEdit_wifissid")
        self.label_wifipass = QtWidgets.QLabel(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_wifipass.sizePolicy().hasHeightForWidth())
        self.label_wifipass.setSizePolicy(sizePolicy)
        self.label_wifipass.setMinimumSize(QtCore.QSize(0, 40))
        self.label_wifipass.setStyleSheet("font: 10pt \"幼圆\";")
        self.label_wifipass.setAlignment(QtCore.Qt.AlignCenter)
        self.label_wifipass.setObjectName("label_wifipass")
        self.lineEdit_wifipawd = QtWidgets.QLineEdit(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_wifipawd.sizePolicy().hasHeightForWidth())
        self.lineEdit_wifipawd.setSizePolicy(sizePolicy)
        self.lineEdit_wifipawd.setMaximumSize(QtCore.QSize(16777215, 40))
        self.lineEdit_wifipawd.setFrame(True)
        self.lineEdit_wifipawd.setObjectName("lineEdit_wifipawd")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(180, 440, 641, 136))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_title_5 = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_title_5.sizePolicy().hasHeightForWidth())
        self.label_title_5.setSizePolicy(sizePolicy)
        self.label_title_5.setMinimumSize(QtCore.QSize(0, 40))
        self.label_title_5.setStyleSheet("font: 10pt \"幼圆\";")
        self.label_title_5.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_title_5.setObjectName("label_title_5")
        self.gridLayout.addWidget(self.label_title_5, 1, 0, 1, 1)
        self.label_title_6 = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_title_6.sizePolicy().hasHeightForWidth())
        self.label_title_6.setSizePolicy(sizePolicy)
        self.label_title_6.setMinimumSize(QtCore.QSize(0, 40))
        self.label_title_6.setStyleSheet("font: 10pt \"幼圆\";")
        self.label_title_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_title_6.setObjectName("label_title_6")
        self.gridLayout.addWidget(self.label_title_6, 2, 0, 1, 1)
        self.label_title_7 = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_title_7.sizePolicy().hasHeightForWidth())
        self.label_title_7.setSizePolicy(sizePolicy)
        self.label_title_7.setMinimumSize(QtCore.QSize(0, 40))
        self.label_title_7.setStyleSheet("font: 10pt \"幼圆\";\n"
"")
        self.label_title_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_title_7.setObjectName("label_title_7")
        self.gridLayout.addWidget(self.label_title_7, 0, 0, 1, 1)
        self.splitter_4 = QtWidgets.QSplitter(self.centralwidget)
        self.splitter_4.setGeometry(QtCore.QRect(160, 170, 671, 60))
        self.splitter_4.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_4.setObjectName("splitter_4")
        self.label_title_2 = QtWidgets.QLabel(self.splitter_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_title_2.sizePolicy().hasHeightForWidth())
        self.label_title_2.setSizePolicy(sizePolicy)
        self.label_title_2.setMinimumSize(QtCore.QSize(0, 60))
        self.label_title_2.setStyleSheet("font: 12pt \"幼圆\";")
        self.label_title_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title_2.setObjectName("label_title_2")
        self.splitter_3 = QtWidgets.QSplitter(self.splitter_4)
        self.splitter_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_3.setObjectName("splitter_3")
        self.radioButton_3 = QtWidgets.QRadioButton(self.splitter_3)
        self.radioButton_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.radioButton_3.setStyleSheet("font: 12pt \"幼圆\";\n"
"")
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton = QtWidgets.QRadioButton(self.splitter_3)
        self.radioButton.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.radioButton.setStyleSheet("font: 12pt \"幼圆\";\n"
"")
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.splitter_3)
        self.radioButton_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.radioButton_2.setAutoFillBackground(False)
        self.radioButton_2.setStyleSheet("font: 12pt \"幼圆\";")
        self.radioButton_2.setChecked(True)
        self.radioButton_2.setAutoRepeat(False)
        self.radioButton_2.setAutoExclusive(True)
        self.radioButton_2.setObjectName("radioButton_2")
        self.splitter.raise_()
        self.label_title_6.raise_()
        self.label_title_5.raise_()
        self.label_title_2.raise_()
        self.label_title.raise_()
        self.label_stacon_tip.raise_()
        self.pushButton.raise_()
        self.label_QR.raise_()
        self.comboBox_robot_style.raise_()
        self.checkBox_4.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.comboBox_robot_style.setCurrentIndex(1)
        self.radioButton_2.clicked['bool'].connect(self.label_wifi.show)
        self.radioButton_2.clicked['bool'].connect(self.lineEdit_wifissid.show)
        self.radioButton_2.clicked['bool'].connect(self.label_wifipass.show)
        self.radioButton_2.clicked['bool'].connect(self.lineEdit_wifipawd.show)
        self.radioButton_2.toggled['bool'].connect(self.label_wifi.hide)
        self.radioButton_2.toggled['bool'].connect(self.lineEdit_wifissid.hide)
        self.radioButton_2.toggled['bool'].connect(self.label_wifipass.hide)
        self.radioButton_2.toggled['bool'].connect(self.lineEdit_wifipawd.hide)
        self.radioButton.clicked['bool'].connect(self.label_title_5.show)
        self.radioButton.toggled['bool'].connect(self.label_title_5.hide)
        self.radioButton.clicked['bool'].connect(self.label_title_6.show)
        self.radioButton_3.clicked['bool'].connect(self.label_title_7.show)
        self.radioButton_3.toggled['bool'].connect(self.label_title_7.hide)
        self.radioButton.toggled['bool'].connect(self.label_title_6.hide)
        self.radioButton_2.toggled['bool'].connect(self.label_QR.hide)
        self.radioButton_2.toggled['bool'].connect(self.label_stacon_tip.hide)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.checkBox_4.setText(_translate("MainWindow", "跳过连接机器人"))
        self.comboBox_robot_style.setItemText(0, _translate("MainWindow", "步兵形态"))
        self.comboBox_robot_style.setItemText(1, _translate("MainWindow", "工程形态"))
        self.pushButton.setText(_translate("MainWindow", "连接机器人"))
        self.label_title.setText(_translate("MainWindow", "欢迎使用 RoboMaster EP 功能演示平台"))
        self.label_wifi.setText(_translate("MainWindow", "Wifi名称:"))
        self.lineEdit_wifissid.setText(_translate("MainWindow", "TP-LINK_12B3"))
        self.label_wifipass.setText(_translate("MainWindow", "Wifi密码:"))
        self.lineEdit_wifipawd.setText(_translate("MainWindow", "WRKCYXGS"))
        self.label_title_5.setText(_translate("MainWindow", "1.连接之前确认机器人连接方式开关设置为AP模式."))
        self.label_title_6.setText(_translate("MainWindow", "2.确定本主机已经连接至机器人热点，默认热点名称以RM开头，密码为12341234."))
        self.label_title_7.setText(_translate("MainWindow", "1.连接之前确认本机通过USB线连接到 EP 的智能中控的 Micro USB 端口."))
        self.label_title_2.setText(_translate("MainWindow", "请选择机器人连接方式："))
        self.radioButton_3.setText(_translate("MainWindow", "USB模式"))
        self.radioButton.setText(_translate("MainWindow", "热点模式"))
        self.radioButton_2.setText(_translate("MainWindow", "无线模式"))
