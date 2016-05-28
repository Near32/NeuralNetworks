#ifndef ONSCNEWTON_H
#define ONSCNEWTON_H
#include "../MAT/Mat.h"
//#define verbose_mini
#define verbose

#include <vector>

template<typename T>
class ONSCNewton
{

	private :
	vector<Mat<T> > x_k;					//variables
	int k;
	T h;							//step for the derivations...
	//vector<Mat<T> > params;				//argument to the function.	
		
	std::vector<Mat<T> > Delta;					//Error terms.
	std::vector<Mat<T> > grad;					//Jacobian to store.
	std::vector<Mat<T> > hess;
	
		
	Mat<T> (*energy)(const Mat<T>&,void*);
	void * obj;
	
	
	public:
		
	/*Constructor
	 * @param OO object 
	 * Ff ; pointer to the function to be optimized. OUGHT TO BE IMPLEMENTED
	 * init; pointer to the function which initializes and handle the variable x_k (give dimension and so on...) OUGHT TO BE IMPLEMENTED
	 * @param iteration ; number of iteration to do ; by default there will be a stopping criteria applied.
	 **/
    ONSCNewton(  Mat<T> (*energy_)(const Mat<T>&,void*), const Mat<T>& x_k_init, int it = 1, bool approxHessian = false, void* obj_ = NULL) : energy(energy_),obj(obj_)
	{
//		int counter = 0;	//count le nombre d'essai de rattraper une mauvaise variation..
		x_k.insert( x_k.end(), x_k_init );
		int counter = 0;
		
#ifdef verbose_mini
		cout << "ONSCNewton Initialization : DONE." << endl;	
#endif
		vector<Mat<T> > cost;
		cost.insert(cost.end(), (this->energy(x_k[0],this->obj)) );
		
		T variation = 0;
		k=0;
		
        while( /*(fabs_(variation) > 10e-5 || variation == 0)   &&*/ k<it)
		{
            if(k==0)
                h = sqrt(numeric_limits<T>::epsilon())*norme2(cost[0])+numeric_limits<T>::epsilon();
            else
                h = sqrt(numeric_limits<T>::epsilon())*norme2(x_k[k])+numeric_limits<T>::epsilon();

#ifdef verbose_mini
			cout << "///////////////////////////\n ONSCNewton : Running : iteration " << k << " COST : " << endl;
			cost[k].afficher();
			cout << "///////////////////////////\n ONSCNewton : variable : " << endl;
			x_k[k].afficher();
#endif
            if(grad.size() < k+1)
				grad.insert( grad.end(), computeGrad() );
			else
				grad[k]=computeGrad();
			
            if(hess.size() < k+1)
            {
                if(approxHessian)
                    hess.insert(hess.end(), grad[k]*transpose(grad[k]) );
                else
                    hess.insert(hess.end(), computeHessian() );
            }
			else
            {
                if(approxHessian)
                    hess[k] = grad[k]*transpose(grad[k]);
                else
                    hess[k] = computeHessian();
            }

#ifdef verbose
			cout << "ONSCNewton : gradient : computed : "<< endl;

			(grad[k]).afficher();
#endif
			T normeGrad = norme2(grad[k]);
			

            while(normeGrad == (T)0 && h<=(T)10)
			{
                //minimum local...
                h = 2*h;
                cout << "gradient recomputation" << endl;
                grad[k] = computeGrad();
				normeGrad = norme2(grad[k]);		
			}            
			
#ifdef verbose
			cout << "ONSCNewton : Hessian : computed : "<< endl;
			(hess[k]).afficher();
#endif
			T normeHess = norme2(hess[k]);
			
            /*
			while(normeHess == (T)0 && h <= 10)
			{
                /minimum local...
                h = sqrt(h);
#ifdef verbose_mini
                cout << "ONSCNewton : variation h : " << h << endl;
#endif
                if(approxHessian)
                {
                    grad[k] = computeGrad();
                    hess[k] = grad[k]*transpose(grad[k]);
                }
                else
                    hess[k] = computeHessian();
#ifdef verbose_mini
				cout << "ONSC : Hessian : recomputation... :" << endl;
				hess[k].afficher();
#endif
				normeHess = norme2(hess[k]);		
			}
            */
			
            /* INVERSION WITH SVD COMPUTATION*/
            /*
            SVD<T> instanceSVD(hess[k]);
			Mat<T> S(instanceSVD.getS());
            for(int i=S.getLine();i--;)	S.set( (T)(1.0/(S.get(i+1,i+1) != (T) 0? S.get(i+1,i+1) : sqrt(numeric_limits<T>::epsilon()) )), i+1,i+1);
            Mat<T> invHess( instanceSVD.getV()*S*transpose(instanceSVD.getU() ) );
            */
            //Mat<T> invHess(invSVD(hess[k]));

            /*INVERSION WITH GAUSS-JORDAN*/
            Mat<T> invHess( invGJ(hess[k]) );

#ifdef verbose
			cout << "ONSCNewton : Inversion : computed " << endl;
			invHess.afficher();
			(hess[k]*invHess).afficher();
#endif
			

			
			if(Delta.size() < k+1)
				Delta.insert( Delta.end(), invHess*grad[k] );
			else
				Delta[k] = invHess*grad[k];
#ifdef verbose
			cout << "ONSCNewton : Delta : computed : size vector : " << Delta.size()  << endl;			
			(Delta[k]).afficher();
#endif
								
			if(x_k.size() < k+2)	
				x_k.insert( x_k.end(), x_k[k] - Delta[k] );
			else
				x_k[k+1] = x_k[k] - Delta[k];

#ifdef verbose_mini
			cout <<  "ONSCNewton : X_k+1 : updated : "<< endl;
			x_k[k+1].afficher();
#endif
			/*----------------------------------------------------------*/
			//Handling of the Convergence :
			if(cost.size() < k+2)
				cost.insert( cost.end(), this->energy(x_k[k+1],this->obj) );//- cost[k] );
			else
				cost[k+1] = this->energy(x_k[k+1],this->obj);
								
			variation = (cost[k+1]-cost[k]).get(1,1);
#ifdef verbose_mini
            cout << "///////// VARIATION : " << variation << endl;
#endif
				        
		        /*----------------------------------------------------------*/
		        k++;
                counter++;
		        
		}
		

	}
	
	
    ~ONSCNewton()
    {

    };
	
    Mat<T> getX(int rank = -1)
	{		
		if(rank > -1)
			return x_k[rank];
		else
            return x_k[k-1];
	}
	
	Mat<T> computeGrad()
	{
		int n = x_k[k].getLine();		
		Mat<T> delta((T)0, n,1);
		delta.set(h, 1,1);
		Mat<T> grad( this->energy(x_k[k]+delta,this->obj) - this->energy(x_k[k]-delta,this->obj)    );        
		
		for(int i=2;i<=n;i++)
		{
			delta.set((T)0,i-1,1);
			delta.set(h,i,1);
			grad = operatorC(grad, this->energy(x_k[k]+delta,this->obj) - this->energy(x_k[k]-delta,this->obj)    );		
		}        
		
		return ((T)(1.0/(2*h)))*grad;
	}
	
    inline Mat<T> computeHessian()
	{
        //clock_t timer = clock();
		int n = x_k[k].getLine();
		Mat<T> delta1((T)0, n,1),delta2((T)0,n,1);
		delta1.set(h, 1,1);
		delta2.set(h, 1,1);		
		Mat<T> hessian((T)0,n,1);
		
		
		for(int j=1; j<=n;j++)
		{
			Mat<T> hess1( this->energy(x_k[k]+delta2+delta1,this->obj) - this->energy(x_k[k]+delta2-delta1,this->obj)  );
			Mat<T> hess2( this->energy(x_k[k]-delta2+delta1,this->obj) - this->energy(x_k[k]-delta2-delta1,this->obj) );			
			for(int i=2;i<=n;i++)
			{
				delta1.set((T)0,i-1,1);
				delta1.set(h,i,1);
				
				hess1 = operatorC(hess1,  this->energy(x_k[k]+delta2+delta1,this->obj) - this->energy(x_k[k]+delta2-delta1,this->obj));		
				hess2 = operatorC(hess2,  this->energy(x_k[k]-delta2+delta1,this->obj) - this->energy(x_k[k]-delta2-delta1,this->obj));		
			}
			hessian = operatorL(hessian, hess1-hess2);
			
			delta2.set((T)0,j,1);
			delta2.set(h,j+1,1);
			
			delta1.set((T)0,n,1);
			delta1.set(h,1,1);
		}
		
		hessian = extract(hessian, 1,2, hessian.getLine(),hessian.getColumn());

        //cout << ((T)(clock()-timer)/CLOCKS_PER_SEC) << " secondes prise pour la HESSIANNE." << endl;
		
		return ((T)(1.0/(4*h*h)))*hessian;
	
	}

};


#endif
