#ifndef DATA_H
#define DATA_H

#include "Mat.h"
#include <string>
#include <vector>

//#define debuglvl1

class DATA
{
    public :

    std::vector<Mat<float> > vectors;
    std::string name;
    std::vector<int> ranges;

    //----------------------------------------

    DATA(const string& name_, const std::vector<Mat<float> >& v)    : vectors(v), name(name_)
    {
        ranges.clear();
        for(uint i=0;i<vectors.size();i++)
        {
            ranges.push_back( vectors[i].getColumn() );
        }
    }

    ~DATA()
    {

    }

    void setMaxRange(const int& maxratio)
    {
        for(uint i=0;i<ranges.size();i++)
        {
            float val = vectors[i].getLine();
            ranges[i] = (int)(maxratio/100.0f*val)+1;

#ifdef debuglvl1
            std::cout << " RANGES " << i << " : " << ranges[i] << " / " << val << std::endl;
#endif

        }
    }

};

float getMaxDatas(const DATA& d);

#endif // DATA_H
