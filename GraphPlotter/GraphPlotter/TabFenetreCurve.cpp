#include "TabFenetreCurve.h"


void TabFenetreCurve::drawGraphs()
{
    for(int idx=m_qcps.size();idx--;)
    {
        m_qcps[idx]->replot();
    }
}

void TabFenetreCurve::receivedValuePourcentageEmitIdxAndValue(int idx,int val)
{
    hasChanged = true;
    m_data[idx].setMaxRange(val);
    m_initialized[idx] = false;
    initializeTab();
    drawTab();
}







