#ifndef MAFENETRE_H
#define MAFENETRE_H

#include <QApplication>
#include <QtWidgets>
#include <QPushButton>
#include "qcustomplot.h"

class MaFenetre : public QWidget
{
    public :

    MaFenetre() : QWidget()
    {
        setFixedSize(800, 600);
/*
        // Construction du bouton
        m_bouton = new QPushButton("Pimp mon bouton !", this);
        m_bouton->setFont(QFont("Comic Sans MS", 14));
        m_bouton->setCursor(Qt::PointingHandCursor);

        m_bouton->move(0,20);
*/
        //---------------

        // generate some data:
        QVector<double> x(101), y(101),z(101); // initialize with entries 0..100
        float rmax = 0.0f;
        for (int i=0; i<101; ++i)
        {
          x[i] = i/50.0 - 1; // x goes from -1 to 1
          y[i] = x[i]*x[i]; // let's plot a quadratic function
          z[i] = x[i];
          float r = sqrt(x[i]*x[i]+y[i]*y[i]);
          if( r> rmax)
          {
              rmax = r;
          }
        }
        // create graph and assign data to it:
        m_plot = new QCustomPlot(this);
        m_plot->setGeometry(50,50,700,500);
        m_plot->addGraph();
        m_plot->graph(0)->setData(x, y);
        m_plot->graph(0)->setPen(QPen(Qt::blue));
        m_plot->addGraph();
        m_plot->graph(1)->setData(x, z);
        m_plot->graph(1)->setPen(QPen(Qt::red));
        // give the axes some labels:
        m_plot->xAxis->setLabel("x");
        m_plot->yAxis->setLabel("y");
        // set axes ranges, so we see all data:
        m_plot->xAxis->setRange(-rmax, rmax);
        m_plot->yAxis->setRange(-rmax, rmax);
        m_plot->replot();

    }

    ~MaFenetre()
    {
        delete m_plot;
    }

    private :
    QCustomPlot* m_plot;

};

#endif // MAFENETRE_H
