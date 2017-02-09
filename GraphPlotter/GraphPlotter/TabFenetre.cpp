#include "TabFenetre.h"


void QSliderIdx::valueChangedIdx(int value)
{
    emit valueIdxChanged(idx,value);
}

void TabFenetre::drawGraphs()
{
    for(int idx=m_qcps.size();idx--;)
    {
        m_qcps[idx]->replot();
    }
}

void TabFenetre::receivedValuePourcentageEmitIdxAndValue(int idx,int val)
{
    hasChanged = true;
    m_data[idx].setMaxRange(val);
    m_initialized[idx] = false;
    initializeTab();
    drawTab();
}


std::vector<QVector<double> > STD2QTVector(const std::vector<Mat<float> >& d)
{
    std::vector<QVector<double> > r;
    int sized = d.size();
    int idxmatrix = 0;
    int idxqv = 0;

    while(idxmatrix <= sized)
    {
        int sizeT = d[idxmatrix].getLine();
        int nbrColumn = d[idxmatrix].getColumn();

        for(int i=nbrColumn;i--;)
        {
            r.push_back(QVector<double>(sizeT));
        }


        for(int l=1;l<=nbrColumn;l++)
        {
            for(int k=1;k<=sizeT;k++)
            {
                r[idxqv+l-1][k-1] = d[idxmatrix].get(k,l);
            }
        }

        idxqv += nbrColumn;

        idxmatrix++;
    }

    return r;

}


std::vector<QVector<double> > STD2QTVector(const DATA& data)
{
    std::vector<Mat<float> > d = data.vectors;
    std::vector<QVector<double> > r;
    int sized = d.size();
    int idxmatrix = 0;
    int idxqv = 0;

    while(idxmatrix < sized)
    {
        //int sizeT = d[idxmatrix].getLine();
        int sizeT = data.ranges[idxmatrix];

        int nbrColumn = d[idxmatrix].getColumn();


        for(int i=nbrColumn;i--;)
        {
            r.push_back(QVector<double>(sizeT));
        }


        for(int l=1;l<=nbrColumn;l++)
        {
            for(int k=1;k<=sizeT;k++)
            {
                r[idxqv+l-1][k-1] = d[idxmatrix].get(k,l);
            }
        }

        idxqv += nbrColumn;

        idxmatrix++;
    }

    return r;

}




