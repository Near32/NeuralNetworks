QT += widgets core

greaterThan(QT_MAJOR_VERSION, 4):QT+=widgets printsupport

TARGET = GraphPlotterProject

SOURCES += \
    main.cpp \
    qcustomplot.cpp \
    TabFenetre.cpp \
    DATA.cpp \
    TabFenetreCurve.cpp

HEADERS += \
    MaFenetre.h \
    qcustomplot.h \
    TabFenetre.h \
    DATA.h \
    main.h \
    TabFenetreCurve.h

CONFIG += c++11
