#include "DATA.h"

float getMaxDatas(const DATA& d)
{
    float max = 0.0f;

    for(int k=d.vectors.size();k--;)
    {
        //for(int i=1;i<=d.vectors[k].getLine();i++)
        for(int i=1;i<=d.ranges[k];i++)
        {
            for(int j=1;j<=d.vectors[k].getColumn();j++)
            {
                if( fabs_(d.vectors[k].get(i,j)) > max)
                {
                    max = fabs_(d.vectors[k].get(i,j));
                }
            }
        }
    }

    return max*1.1f;
}

