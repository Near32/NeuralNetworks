#include "main.h"

int main(int argc, char* argv[])
{
    QApplication app(argc,argv);

    /*// Création d'un widget qui servira de fenêtre
    QWidget fenetre;
    fenetre.setFixedSize(300, 150);

    // Création du bouton, ayant pour parent la "fenêtre"

    QPushButton bouton("Pimp mon bouton !", &fenetre);

    // Personnalisation du bouton
    bouton.setCursor(Qt::PointingHandCursor);

    bouton.setGeometry(60, 50, 180, 70);
    // Création d'un autre bouton ayant pour parent le premier bouton

    QPushButton autreBouton("Autre bouton", &bouton);
    autreBouton.move(30, 15);

    fenetre.show();
    */

    /*
    MaFenetre fenetre;
    fenetre.show();
    */

    //TabFenetre tf;
    TabFenetreCurve tf;
    tf.show();

    //DATA LOADING :
    /*
    std::vector<Mat<float> > v;
    int nbrT = 1000;
    for(int i=0;i<2;i++) v.push_back( Mat<float>(nbrT,2,(char)1) );
    v[0].set(20,1,1);
    v[1].set(10,1,1);

    for(int i=-nbrT/2;i<=nbrT/2;i++)
    {
        v[0].set(((float)i)/100,nbrT/2+i+1,1);
        v[1].set(((float)i)/100,nbrT/2+i+1,1);
        v[0].set( 3*cos(((float)i)/100), nbrT/2+i+1,2);
        v[1].set( exp(((float)i)/100), nbrT/2+i+1,2);
    }

    tf.addData( DATA(std::string("DATA1"), v ) );

    v.clear();

    for(int i=0;i<1;i++) v.push_back( Mat<float>(40,2,(char)1) );
    tf.addData( DATA(std::string("DATA2"), v ) );
    */

    std::string filepath("./data.txt");
    char delim = ' ';
    if(argc > 1)
    {
        delim = argv[1][0];

    }
    std::cout << " DELIMITER is : ---" << delim << "---" << std::endl;

    std::vector<DATA> datas = extract(filepath, delim);
    int nbrpergroup = 3;
    //groupDATAby(datas,nbrpergroup);

    for(uint i=0;i<datas.size();i++)
    {
        tf.addData(datas[i]);
    }

    tf.run();


    /*
    createView view;
    view.start();

    std::vector<Mat<float> > v;
    for(int i=0;i<=4;i++) v.push_back( Mat<float>(20,2,(char)1) );

    while(view.f == NULL);

    view.f->addData( DATA(std::string("DATA1"), v ) );

    v.clear();
    for(int i=0;i<=1;i++) v.push_back( Mat<float>(1.0f,20,2) );
    view.f->addData( DATA(std::string("DATA2"), v ) );


    //view.f->show();
    */

    return app.exec();
}

/*
void createView( QWidget* f)
{
    f = (QWidget*)(new TabFenetre());
}
*/


std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

std::vector<DATA> extract(const string& filepath, char delim)
{
    std::vector<Mat<float> > vectors;
    std::vector<std::string> names;
    std::vector<DATA> r;

    std::string line;
    std::ifstream infile(filepath);

    int nbrr = 0;
    while (std::getline(infile, line))
    {
        std::vector<std::string> lineSplit = split(line,':');
        regularizeDATA(lineSplit);

        if(lineSplit.size() == 2)
        {
            names.push_back(lineSplit[0]);

            std::vector<std::string> data = split(lineSplit[1],delim);
            regularizeDATA(data);

            int nbrLine = data.size();

            if(nbrLine>0)
            {
                Mat<float> column(0.0f,nbrLine,1);
                for(int i=0;i<nbrLine;i++)
                {
                    column.set( std::stof(data[i]), i+1,1);
                }
                vectors.push_back(column);
            }
            else
            {
                std::cout << "PROBLEM EXTRACTING VALUES...." << std::endl;
            }
        }
        else
        {
            std::cout << "PROBLEM EXTRACTING DATAS..." << std::endl;
        }


        nbrr++;

        if(nbrr%2 == 0)
        {
            //let us reunite those two data :
            Mat<float> dummy(vectors[1]);
            vectors.pop_back();

            if(dummy.getLine() == vectors[0].getLine())
            {
                dummy = operatorL(vectors[0],dummy);
                vectors.pop_back();
            }
            else
            {
                std::cout << "NOT THE SAME NBR OF LINE ...." << std::endl;
                dummy = operatorL( extract( vectors[0], 1,1,dummy.getLine(),1) ,dummy);
                vectors.pop_back();
            }

            std::vector<Mat<float> >dummyvec;
            dummyvec.push_back(dummy);

            r.push_back( DATA( names[0]+std::string(" =f( ")+names[1]+std::string(" )"), dummyvec) );
            names.clear();
        }

    }

    infile.close();

    return r;
}


void regularizeDATA(std::vector<std::string>& d)
{
    for(int i=0;i<d.size();i++)
    {
        if( d[i].compare( std::string("") )  == 0)
        {
            d.erase(d.begin()+i);
            i--;
        }
    }
}

void groupDATAby(std::vector<DATA>& data, const int& nbrpergroup)
{
    std::vector<DATA> r;
    DATA di(data[0].name, data[0].vectors);

    for(int i=1;i<data.size();i++)
    {
        di.name = di.name+data[i].name;
        for(int j=0;j<data[i].vectors.size();j++)
        {
            di.vectors.push_back( data[i].vectors[j]);
        }

        if( i%nbrpergroup == 1)
        {
            r.push_back(di);

            i++;
            if(i<data.size())
            {
                di.name = data[i].name;
                di.vectors = data[i].vectors;
            }
        }
    }

    data = r;
}
