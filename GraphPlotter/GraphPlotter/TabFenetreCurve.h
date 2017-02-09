#ifndef TABFENETRECURVE_H
#define TABFENETRECURVE_H

#include <QApplication>
#include <QtWidgets>
#include <QPushButton>
#include <QThread>

#include "qcustomplot.h"
#include <vector>
#include <mutex>
#include "DATA.h"
#include <iostream>
#include <thread>

#include "TabFenetre.h"

//#define debuglvl1


class TabFenetreCurve : public QWidget
{
    Q_OBJECT

    public :

    TabFenetreCurve(unsigned int nbrTab_=0) : QWidget(), nbrTab(nbrTab_)
    {
        setFixedSize(650, 450);
        continuer = true;
        hasChanged = true;

        //---------------

        m_onglets = new QTabWidget(this);
        m_onglets->setGeometry( 10, 10, 630, 430);

        int r=50;
        int g=50;
        int b=50;

        /*
        for(int ri=0;ri<5;ri++)
        {
            for(int gi=0;gi<5;gi++)
            {
                for(int bi=0;bi<5;bi++)
                {
                    QColor dummy;
                    dummy.fromRgb( (ri*r)%250,(gi*g)%250,(bi*b)%250 );
                    //dummy.fromRgb( ri,gi,bi );
                    std::cout << dummy.isValid() << std::endl;
                    colors.push_back( dummy );
                }
            }
        }
        */
        colors.push_back(Qt::black);
        colors.push_back(Qt::cyan);
        colors.push_back(Qt::darkCyan);
        colors.push_back(Qt::red);
        colors.push_back(Qt::magenta);
        colors.push_back(Qt::green);
        colors.push_back(Qt::yellow);
        colors.push_back(Qt::blue);
        colors.push_back(Qt::gray);

        nbrColors = colors.size();
        idxColor = 0;

    }

    ~TabFenetreCurve()
    {
        this->setContinuer(false);

        for(uint i=0;i<m_widgets.size();i++)
        {
            delete m_widgets[i];
        }

        /*for(int i=0;i<m_qcps.size();i++)
        {
            delete m_qcps[i];
        }
        */

        /*
        for(int i=0;i<m_sliders.size();i++)
        {
            delete m_sliders[i];
        }
        */

        delete m_onglets;
    }

    void initializeTab()
    {
        if(hasChanged)
        {
            for(uint i=0;i<m_initialized.size();i++)
            {
                if(m_initialized[i] == false)
                {

                    //initialization of the frame :
                    if( m_initializedFrame[i] == false)
                    {
                        m_initializedFrame[i] = true;
                        m_widgets.push_back(new QWidget());

                        //add the slider :
                        //m_sliders.push_back( new QSlider(Qt::Horizontal, m_widgets[m_widgets.size()-1] ) );
                        m_sliders.push_back( new QSliderIdx(m_widgets.size()-1, Qt::Horizontal, m_widgets[m_widgets.size()-1] ) );
                        m_sliders[m_sliders.size()-1]->setRange(1,101);
                        m_sliders[m_sliders.size()-1]->setValue(100);
                        m_sliders[m_sliders.size()-1]->setGeometry(10, 370, 600, 20);
                        //m_sliders[m_slider.size()-1]->setRange(0,100);
                        QObject::connect(m_sliders[m_sliders.size()-1], SIGNAL(valueIdxChanged(int,int)), this, SLOT(receivedValuePourcentageEmitIdxAndValue(int,int)) );

                        m_qcps.push_back(new QCustomPlot(m_widgets[m_widgets.size()-1]) );
                        m_qcurves.push_back( std::vector<QCPCurve*>(0));
                        m_qcps[m_qcps.size()-1]->setGeometry(25,0,550,350);
                        //end of initialization of the frame.
                        std::string title = std::string("DATA")+std::to_string(m_widgets.size()-1)+std::string(" : ")+ m_data[i].name;
                        m_onglets->addTab(m_widgets[m_widgets.size()-1], QString::fromUtf8(title.c_str()) ) ;

                    }
                    //let us initialiaze that tab with the given datas :
                    //std::vector<QVector<double> > d = STD2QTVector(m_data[i].vectors);
                    std::vector<QVector<double> > d = STD2QTVector(m_data[i]);
                    uint size = d.size();

                    if(size ==2)
                    {
                        //there is only one curve :
                        int idxQcps = i;
                        int nbrCurve = m_qcurves[idxQcps].size();

                        if( nbrCurve < 1)
                        {
                            m_qcurves[idxQcps].push_back( new QCPCurve(m_qcps[idxQcps]->xAxis, m_qcps[idxQcps]->yAxis) );
                            m_qcps[idxQcps]->addPlottable( m_qcurves[idxQcps][0] );
                        }

                        m_qcurves[idxQcps][0]->setData(d[0],d[1]);
                    }
                    else
                    {
                        //then it is not a graph but some parametric curve...
                        int nbrCurveNeeded = size/2;
                        std::cout << " NBR CURVE NEEDED : " << size << std::endl;
                        int idxQcps = i;
                        int idxCurve = m_qcurves[idxQcps].size()-1;
                        int nbrCurveCurrently = m_qcurves[idxQcps].size();

                        bool add = (nbrCurveCurrently < size/2?true:false);
                        while( add)
                        {
                            m_qcurves[idxQcps].push_back( new QCPCurve(m_qcps[idxQcps]->xAxis, m_qcps[idxQcps]->yAxis) );

                            nbrCurveCurrently = m_qcurves[idxQcps].size();
                            idxCurve = m_qcurves[idxQcps].size()-1;

                            m_qcps[idxQcps]->addPlottable( m_qcurves[idxQcps][idxCurve] );

                            m_qcurves[idxQcps][idxCurve]->setPen(QPen(colors[idxColor]) );
                            idxColor = (idxColor+10)%nbrColors;

                            add = (m_qcurves[idxQcps].size() < size/2?true:false);
                        }

                        std::cout << " NBR CURVE CURRENTLY : " << nbrCurveCurrently << std::endl;

                        for(int c=0;c<nbrCurveNeeded;c++)
                        {
                            m_qcurves[i][c]->setData(d[2*c],d[2*c+1]);

                            m_qcurves[i][c]->setPen(QPen(colors[idxColor]) );
                            idxColor = (idxColor+10)%nbrColors;
                            /*
                            if(idxColor%2)
                            {
                                m_qcps[i]->graph(c)->setPen(QPen(Qt::red) );
                                std::cout << " set color red to graph : " << c << std::endl;
                            }
                            else
                            {
                                m_qcps[i]->graph(c)->setPen(QPen(Qt::blue) );
                                std::cout << " set color blue to graph : " << c << std::endl;
                            }
                            idxColor = (idxColor+1)%nbrColors;
                            */
                        }

                    }

                    m_initialized[i] = true;
                }
            }
        }

    }

    void drawTab()
    {
        if(hasChanged)
        {
            hasChanged = false;
#ifdef debuglvl1
            std::cout << "GRAPHS UPDATED." << std::endl;
#endif
            for(uint i=0;i<m_qcps.size();i++)
            {
                float rmax = getMaxDatas(m_data[i]);

                // set axes ranges, so we see all data:
                m_qcps[i]->xAxis->setRange(-rmax, rmax);
                m_qcps[i]->yAxis->setRange(-rmax, rmax);
                m_qcps[i]->replot();
            }

        }
    }

    void addData(const DATA& d)
    {
        hasChanged = true;
        nbrTab++;
        m_data.push_back(d);

        m_initializedFrame.push_back(false);
        m_initialized.push_back(false);
    }

    //Real time update :
    void ReplaceLastData(const DATA& d)
    {
        m_data[m_data.size()-1] = d;
        m_initialized[m_initialized.size()-1] = false;
    }

    void run()
    {
        while(continuer)
        {

            initializeTab();

            drawTab();

            continuer = false;
        }

    }


    bool getContinuerStatus()   const
    {
        return continuer;
    }

    void setContinuer(bool cont)
    {
        continuer = cont;
    }

    public slots:
    void drawGraphs();
    void receivedValuePourcentageEmitIdxAndValue(int idx,int val);

    signals :
    void requestDrawGraphs();
    void changedValue(int,int);
    //index of the onglets and pourcentage.

    private :
    unsigned int nbrTab;

    QTabWidget* m_onglets;
    std::vector<QWidget*> m_widgets;
    std::vector<QSliderIdx*> m_sliders;
    std::vector<QCustomPlot*> m_qcps;

    std::vector<std::vector<QCPCurve*> > m_qcurves;
    std::vector<bool> m_initialized;
    std::vector<bool> m_initializedFrame;
    std::vector<DATA> m_data;
    bool continuer;

    std::vector<QColor> colors;
    int nbrColors;
    int idxColor;

    bool hasChanged;


};

#endif // TABFENETRECURVE_H
