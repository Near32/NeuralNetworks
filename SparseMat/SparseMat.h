#ifndef SPARSEMAT_H
#define SPARSEMAT_H

//#define useAsTensor
#include <iostream>
#include <cstdlib>
#include <map>
#include "../MAT/Mat.h"

template<class T>
class SparseMat
{
	private :
	
	typedef std::map<size_t, std::map<size_t, T> > mat_t;
	typedef typename mat_t::iterator row_iter;
	typedef std::map<size_t, T> col_t;
	typedef typename col_t::iterator col_iter;
	typedef typename mat_t::const_iterator row_citer;
	typedef typename col_t::const_iterator col_citer;
	
	mat_t mat;
	size_t n;
	size_t m;
	
	public :
	
	SparseMat()
	{
		mat[0][0] = (T)0;
		n = 0;
		m = 0;
	}
	
	SparseMat(size_t i)
	{
		mat[0][0] = (T)0;
		n = i;
		m = i;
	}
	
	SparseMat(size_t i, size_t j)
	{
		mat[0][0] = T(0);
		n = i;
		m = j;
	}
	
	SparseMat(const Mat<T>& mat)
	{
		n = mat.getLine();
		m = mat.getColumn();
		
		for(int i=1;i<=n;i++)
		{
			for(int j=1;j<=m;j++)
			{
				if(mat.get(i,j) != (T)0)
				{
					set( i,j, mat.get(i,j) );
				}
			}
		}
	}
	
	~SparseMat()
	{

	}
	
	inline T operator()(size_t i, size_t j)
	{
		if( !(i>0 && i<=n && j>0 && j<=m) )
		{
			throw;
		}
		
		if( mat.count(i) == 0) 
		{
			return T(0);//mat[0][0];
		}
		
		if( mat[i].count(j) == 0)
		{
			return T(0);//mat[0][0];
		}
		
		return mat[i][j];	
	}
	
	inline T operator()(size_t i, size_t j)	const
	{
		if( !(i>0 && i<=n && j>0 && j<=m) )
		{
			throw;
		}
		
		if( mat.count(i) == 0) 
		{
			return (T)0;//mat[0][0];
		}
		
		if( mat.at(i).count(j) == 0)
		{
			return (T)0;//mat[0][0];
		}
		
		return mat.at(i).at(j);	
	}
	
	/*SparseMat<T>& operator=( SparseMat<T>& sm)
	{
		n = sm.getLine();
		m = sm.getColumn();
		
		mat.clear();
		
		
		size_t i = 0;
		size_t lasti = i;
		size_t j = 0;
		size_t lastj = j;
		bool goOn = true;
		T val;
	
		while(goOn)
		{
			lasti = i;
			lastj = j;
		
			goOn = sm.parcourir(i,j,val);
		
			this->set( val, lasti, lastj);
		}
		
		return *this;
	}*/
	
	inline void set( size_t i, size_t j, T value)
	{
		if( !(i>0 && i<=n && j>0 && j<=m) )
		{
			throw;
		}
		
		mat[i][j] = value;
		
	}
	
	void afficher()
	{
		row_iter ii;
		col_iter jj;
		
		for(ii=mat.begin();ii!=mat.end();ii++)
		{
			for(jj=(*ii).second.begin();jj!=(*ii).second.end();jj++)
			{
				std::cout << (*ii).first << ' ';
				std::cout << (*jj).first << ' ';
				std::cout << (*jj).second << std::endl;
			}
			std::cout << std::endl;
		}
		
		std::cout << std::endl;
	}
	
	void print()
	{
		for(int i=1;i<=n;i++)
		{
			for(int j=1;j<=m;j++)
			{
				std::cout << this->operator()(i,j) << " ";
			}
			std::cout << std::endl;
		}
		
		std::cout << std::endl;
	}
	
	SparseMat<T> operator*( SparseMat<T>& B)
	{
		if( m != B.getLine())	throw;
		
		SparseMat<T> ret(n,B.getColumn());
		T sum;
		
		for(size_t i=1;i<=n;i++)
		{
			if( mat.count(i)>0 )
			{
				row_iter ii = mat.find(i);
				for(size_t j=1;j<=ret.getColumn();j++)
				{
			
					sum = (T)0;
					/*for(size_t k=1;k<=m;k++)
					{
						sum+=this->(i,k)*B(k,j);
					}
					*/
			
					col_iter jj;
					for(jj=(*ii).second.begin();jj!=(*ii).second.end();jj++)
					{
						sum += (*jj).second * B( jj->first, j);
					}
					
					//let's assert that it hasn't be zeroed.
					if( sum != (T)0)
					{
						ret.set(i,j, sum);
					}
			
				
				}
				
				
			}
			
		}
		
		return ret;
	}
	
	/*
	SparseMat<T> operator*( const Mat<T>& B)
	{
		if( m != B.getLine())	throw;
		
		SparseMat<T> ret(n,B.getColumn());
		T sum;
		
		for(size_t i=1;i<=n;i++)
		{
			if( mat.count(i)>0 )
			{
				row_iter ii = mat.find(i);
				for(size_t j=1;j<=ret.getColumn();j++)
				{
			
					sum = (T)0;
			
					col_iter jj;
					for(jj=(*ii).second.begin();jj!=(*ii).second.end();jj++)
					{
						sum+= (*jj).second * B.get( jj->first, j);
					}
					
					//let's assert that it hasn't be zeroed.
					if( sum != (T)0)
					{
						ret.set(i,j, sum);
					}
			
				
				}
				
				
			}
			
		}
		
		return ret;
	}
	*/
	
	Mat<T> operator*( const Mat<T>& B)
	{
		if( m != B.getLine())	throw;
		
		Mat<T> ret((T)0,n,B.getColumn());
		T sum;
		
		for(size_t i=1;i<=n;i++)
		{
			if( mat.count(i)>0 )
			{
				row_iter ii = mat.find(i);
				for(size_t j=1;j<=ret.getColumn();j++)
				{
			
					sum = (T)0;
					/*for(size_t k=1;k<=m;k++)
					{
						sum+=this->(i,k)*B(k,j);
					}
					*/
			
					col_iter jj;
					for(jj=(*ii).second.begin();jj!=(*ii).second.end();jj++)
					{
						sum += (*jj).second * B.get( jj->first, j);
					}
					
					//let's assert that it hasn't be zeroed.
					if( sum != (T)0)
					{
						ret.set(sum, i,j);
					}
			
				
				}
				
				
			}
			
		}
		
		return ret;
	}
	
	SparseMat<T> operator*(const T& value)
	{
		row_iter ii;
		col_iter jj;
		
		for(ii=mat.begin();ii!=mat.end();ii++)
		{
			for(jj=(*ii).second.begin();jj!=(*ii).second.end();jj++)
			{
				(*jj).second *= value;
			}
		}
		
		return (*this);
	}
	
	void operator*=(const T& value)
	{
		row_iter ii;
		col_iter jj;
		
		for(ii=mat.begin();ii!=mat.end();ii++)
		{
			for(jj=(*ii).second.begin();jj!=(*ii).second.end();jj++)
			{
				(*jj).second *= value;
			}
		}
	}
	
	void operator/=(const T& value)
	{
		row_iter ii;
		col_iter jj;
		
		for(ii=mat.begin();ii!=mat.end();ii++)
		{
			for(jj=(*ii).second.begin();jj!=(*ii).second.end();jj++)
			{
				(*jj).second /= value;
			}
		}
	}
	
	/*SparseMat<T> operator=(SparseMat<T> B)
	{
		n = B.getLine();
		m = B.getColumn();
		
		for(size_t i=1;i!=n;i++)
		{
			for(size_t j=1;j!=m;j++)
			{
				T val = B(i,j);
				
				if(val != (T)0)	mat[i][j] = val;
			}
		}
	}*/
	
	SparseMat<T>& operator=(const SparseMat<T>& B)
	{
		n = B.getLine();
		m = B.getColumn();
		
		for(size_t i=1;i!=n;i++)
		{
			for(size_t j=1;j!=m;j++)
			{
				T val = B(i,j);
				
				if(val != (T)0)	mat[i][j] = val;
			}
		}
	}
	
	//-----------------------------------------
	
	size_t getLine()	const
	{
		return n;
	}
	
	size_t getColumn()	const
	{
		return m;
	}
	
	void addLine(size_t an)
	{
		n+=an;
	}
	
	void addColumn(size_t am)
	{
		m+=am;
	}
	
	bool parcourir( size_t& i, size_t& j, T& val)
	{
		bool ret = true;
		
		row_iter ii;
		col_iter jj;
		
		if( i==0 && j==0)
		{
			ii = mat.begin();
			jj = (*ii).second.begin();
		}
		else
		{
			ii = mat.find(i);
			jj = mat[i].find(j);
		}
		
		//---------------------------------------
		//set the value to be used :
		
		val = (*jj).second;
		//---------------------------------------
		
		
		//---------------------------------------
		//handle the next call :
		
		if( (ii==mat.end() && jj == mat[i].end()) || (i==n && j==m))	
		{
			//there will be no next call:
			i = 0;
			j = 0;
			//resetting...
			return false;
		}
		else
		{
			//let us upgrade the indexes for the next call :
			//if( jj == (*ii).second.end())
			if(j==m)
			{
				//end of line --> change the line :
				ii++;
				//reset the column :
				jj = (*ii).second.begin();
			}
			else
			{
				//not the end of line --> change the column :
				jj++;
			}
		
		
			//set the indexes values :
			i = (*ii).first;
			j = (*jj).first;
			
			std::cout << i << " " << j << std::endl;
		}
		
		//-------------------------------------------
		//-------------------------------------------
		
		
		return ret;
	}
	
#ifndef useAsTensor	
	Mat<T> SM2mat()
	{
		Mat<T> ret((T)0, this->n, m);
		row_citer ii;
		col_citer jj;
		
		for(ii=mat.cbegin();ii!=mat.cend();ii++)
		{
			for(jj=(*ii).second.cbegin();jj!=(*ii).second.cend();jj++)
			{
				ret.set( (*jj).second, (*ii).first, (*jj).first);
			}
		}
		
		return ret;
	}
#endif
	
};

#ifndef useAsTensor
template<class T>
Mat<T> SM2Mat(SparseMat<T> sm)
{
	Mat<T> ret((T)0,sm.getLine(),sm.getColumn());
	
	size_t i = 0;
	size_t lasti = i;
	size_t j = 0;
	size_t lastj = j;
	bool goOn = true;
	T val;
	
	while(goOn)
	{
		lasti = i;
		lastj = j;
		
		goOn = sm.parcourir(i,j,val);
		
		ret.set( val, lasti, lastj);
	}
	
	return ret;
}
#endif

#endif
