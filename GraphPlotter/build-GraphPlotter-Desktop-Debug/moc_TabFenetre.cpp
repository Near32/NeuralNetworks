/****************************************************************************
** Meta object code from reading C++ file 'TabFenetre.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.2.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../GraphPlotter/TabFenetre.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'TabFenetre.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.2.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_QSliderIdx_t {
    QByteArrayData data[5];
    char stringdata[51];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    offsetof(qt_meta_stringdata_QSliderIdx_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData) \
    )
static const qt_meta_stringdata_QSliderIdx_t qt_meta_stringdata_QSliderIdx = {
    {
QT_MOC_LITERAL(0, 0, 10),
QT_MOC_LITERAL(1, 11, 15),
QT_MOC_LITERAL(2, 27, 0),
QT_MOC_LITERAL(3, 28, 15),
QT_MOC_LITERAL(4, 44, 5)
    },
    "QSliderIdx\0valueIdxChanged\0\0valueChangedIdx\0"
    "value\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_QSliderIdx[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    2,   24,    2, 0x06,

 // slots: name, argc, parameters, tag, flags
       3,    1,   29,    2, 0x0a,

 // signals: parameters
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    2,    2,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    4,

       0        // eod
};

void QSliderIdx::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        QSliderIdx *_t = static_cast<QSliderIdx *>(_o);
        switch (_id) {
        case 0: _t->valueIdxChanged((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 1: _t->valueChangedIdx((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (QSliderIdx::*_t)(int , int );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&QSliderIdx::valueIdxChanged)) {
                *result = 0;
            }
        }
    }
}

const QMetaObject QSliderIdx::staticMetaObject = {
    { &QSlider::staticMetaObject, qt_meta_stringdata_QSliderIdx.data,
      qt_meta_data_QSliderIdx,  qt_static_metacall, 0, 0}
};


const QMetaObject *QSliderIdx::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *QSliderIdx::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_QSliderIdx.stringdata))
        return static_cast<void*>(const_cast< QSliderIdx*>(this));
    return QSlider::qt_metacast(_clname);
}

int QSliderIdx::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QSlider::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 2)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 2;
    }
    return _id;
}

// SIGNAL 0
void QSliderIdx::valueIdxChanged(int _t1, int _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
struct qt_meta_stringdata_TabFenetre_t {
    QByteArrayData data[8];
    char stringdata[103];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    offsetof(qt_meta_stringdata_TabFenetre_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData) \
    )
static const qt_meta_stringdata_TabFenetre_t qt_meta_stringdata_TabFenetre = {
    {
QT_MOC_LITERAL(0, 0, 10),
QT_MOC_LITERAL(1, 11, 17),
QT_MOC_LITERAL(2, 29, 0),
QT_MOC_LITERAL(3, 30, 12),
QT_MOC_LITERAL(4, 43, 10),
QT_MOC_LITERAL(5, 54, 39),
QT_MOC_LITERAL(6, 94, 3),
QT_MOC_LITERAL(7, 98, 3)
    },
    "TabFenetre\0requestDrawGraphs\0\0"
    "changedValue\0drawGraphs\0"
    "receivedValuePourcentageEmitIdxAndValue\0"
    "idx\0val\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_TabFenetre[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   34,    2, 0x06,
       3,    2,   35,    2, 0x06,

 // slots: name, argc, parameters, tag, flags
       4,    0,   40,    2, 0x0a,
       5,    2,   41,    2, 0x0a,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    2,    2,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    6,    7,

       0        // eod
};

void TabFenetre::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        TabFenetre *_t = static_cast<TabFenetre *>(_o);
        switch (_id) {
        case 0: _t->requestDrawGraphs(); break;
        case 1: _t->changedValue((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 2: _t->drawGraphs(); break;
        case 3: _t->receivedValuePourcentageEmitIdxAndValue((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (TabFenetre::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&TabFenetre::requestDrawGraphs)) {
                *result = 0;
            }
        }
        {
            typedef void (TabFenetre::*_t)(int , int );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&TabFenetre::changedValue)) {
                *result = 1;
            }
        }
    }
}

const QMetaObject TabFenetre::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_TabFenetre.data,
      qt_meta_data_TabFenetre,  qt_static_metacall, 0, 0}
};


const QMetaObject *TabFenetre::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *TabFenetre::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_TabFenetre.stringdata))
        return static_cast<void*>(const_cast< TabFenetre*>(this));
    return QWidget::qt_metacast(_clname);
}

int TabFenetre::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 4)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 4;
    }
    return _id;
}

// SIGNAL 0
void TabFenetre::requestDrawGraphs()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}

// SIGNAL 1
void TabFenetre::changedValue(int _t1, int _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_END_MOC_NAMESPACE