#ifndef MAIN_H
#define MAIN_H

#include <QApplication>
#include "MaFenetre.h"
#include "TabFenetre.h"
#include "TabFenetreCurve.h"


#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>


//void createView( QWidget* f);
std::vector<std::string> split(const std::string &s, char delim);
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);
std::vector<DATA> extract(const string& filepath, char delim);
void regularizeDATA(std::vector<std::string>& d);
void groupDATAby(std::vector<DATA>& data, const int& nbrpergroup);

class createView : public QThread
{

    public :

    createView()
    {
        f = NULL;
    }

    ~createView()
    {
        delete f;
    }

    virtual void run() override
    {
        f = new TabFenetre();
        f->show();
        f->run();
    }

    //----------------------------

    TabFenetre* f;

    private :


};

#endif // MAIN_H
