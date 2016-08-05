#include "Mat.h"


template<typename T>
class Tensor : public Mat<T>
{
	public :
	
	std::vector<Mat<T>*> elements;
	std::vector<int> shape;
	unsigned int depth;
	
	Tensor() : Mat<T>()
	{
		this->elements.clear();
		this->shape.clear();
		
		shape.push_back(1);
		shape.push_back(1);
		shape.push_back(1);
		
		this->elements.push_back( new Mat<T>() );
		this->depth = 1;
	}
	
	Tensor(const int& val) : Mat<T>(val)
	{
		this->elements.clear();
		this->shape.clear();
		
		shape.push_back(1);
		shape.push_back(1);
		shape.push_back(1);
		
		this->elements.push_back( new Mat<T>(val) );
		this->depth =1;
	}
	
	Tensor(const Mat<T>& m) : Mat<T>(m)
	{
		this->elements.clear();
		this->shape.clear();
		
		shape.push_back(1);
		shape.push_back(1);
		shape.push_back(1);
		
		this->elements.push_back( new Mat<T>(m) );
		this->depth = 1;
		
	}
	
	
	Tensor(int line, int column, char mode) : Mat<T>(line,column,mode)
	{
		this->elements.clear();
		this->shape.clear();
		
		shape.push_back(1);
		shape.push_back(line);
		shape.push_back(column);
		
		this->elements.push_back( new Mat<T>(line,column,mode) );
		this->shape = 1;
		
	}
	
	Tensor( int line, int column, unsigned int depth = 1) : Mat<T>(line,column)
	{
		this->elements.clear();
		this->shape.clear();
		
		shape.push_back(depth);
		shape.push_back(line);
		shape.push_back(column);
		
		for(int i=depth;i--;)	this->elements.push_back( new Mat<T>(line,column) );
		this->depth = depth;
	}
       
    Tensor( T value, int line, int column, unsigned int depth = 1) : Mat<T>(value,line,column)
	{
		this->elements.clear();
		this->shape.clear();
		
		shape.push_back(depth);
		shape.push_back(line);
		shape.push_back(column);
		
		for(int i=depth;i--;)	this->elements.push_back( new Mat<T>(value,line,column) );
		this->depth = depth;
	}
	
	void copy(const Tensor<T>& t)
	{
		this->~Tensor();
		for(int i=t.getDepth();i--;)	this->addElement(t.getElement(i+1));
	}
	
	Tensor<T>& operator=(const Tensor<T>& t)
	{
		if(this != &t)
		{
		    if( !(t.getDepth() == this->getDepth() && t.getLine() == this->getLine() && t.getColumn() == this->getColumn() ) )
		    {
				this->copy(t);
				
			}
			else
			{
				this->~Mat();
				this->copy((Mat<T>&)t);
				
				for(int k=this->getDepth();k--;)
				{
					this->setElement( t.getElement(k+1), k+1);
				}
			}
		}
		
		return *this;
	}
	
	~Tensor()
	{
		for(int k=this->elements.size();k--;)
		{
			delete elements[k];
		}
	}
	
	int getLine()	const
	{
		if(this->elements.size() >0)
		{
			return this->elements[0]->getLine();
		}
		else
		{
			return 0;
		}
	}

	int getColumn()	const
	{
		if(this->elements.size() >0)
		{
			return this->elements[0]->getColumn();
		}
		else
		{
			return 0;
		}
	}
	
	int getDepth()	const
	{
		return this->elements.size();
	}
	


	inline T get(int line, int column, int depth=0) const 
	{
		if(depth >0 && depth <= this->elements.size() )
		{
			return this->elements[depth-1]->get(line,column);
		}
		else if(depth == 0)
		{
			return this->get(line,column);
		}
		
		throw;
	}

	/*---------------------------------------------*/

	inline void set(T value, int line, int column, int depth)
	{
		if(depth <= this->elements.size() )
		{
			this->elements[depth-1]->set(value,line,column);
		}
	}
	
	
	inline T& operator()(const int& line, const int& column, const int& depth)
	{
		if(depth <= this->elements.size() )
		{
			return this->elements[depth-1]->get(line,column);
		}
		
		return this->dummy;
	}
	
	inline Mat<T> operator[](const int& depth)
	{
		if(depth > 0 && depth <= this->elements.size() )
		{
			return *(this->elements[depth-1]);
		}
		
		return Mat<T>(0);
	}

	/*---------------------------------------------*/

	void afficher()	const
	{
		for(int k=this->elements.size();k--;)
		{
			cout << "Matrice : " << k << endl;

		    for(int i=1;i<=this->elements[k]->getLine();i++)
		    {
		        for(int j=1;j<=this->elements[k]->getColumn();j++)
		        {
		            cout << "  " << this->elements[k]->get(i,j) ;
		            //cout << "  " << ( fabs_(this->get(i,j)) >= epsilon*10e-20 ? this->get(i,j) : (T)0) ;
		            //cout << "  " << roundoffmat[i-1][j-1] ;
		        }

		        cout << endl;
		    }
		    
		    cout << std::endl;
		    cout << "///////////////////////////////////////////////////" << std::endl;
		}
	}
	
	
};


