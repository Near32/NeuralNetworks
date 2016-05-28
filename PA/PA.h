#ifndef PA_H
#define PA_H

/*Policy Approximator Interface */
#include "../MAT/Mat.h"
#include "../ONSCNewton/ONSCNewton.h"
#include "../NN.h"


template<typename T>
class PA
{
	public :
	
	PA(const float& lr_, const float& eps_) : lr(lr_), eps(eps_)
	{
	
	
	}
	
	~PA()
	{
	
	}
	
	virtual void initialize()=0;
	
	virtual void update(const Mat<T>& S, const Mat<T>& A, const Mat<T>& S1, const Mat<T>& R ) = 0;
	
	virtual Mat<T> estimateAction(const Mat<T>& S) =0;
	
	virtual Mat<T> eps_greedy(const Mat<T>& S) = 0;
	
	protected :
	
	float lr;		//learning rate.
	float eps;		//epsilon greedy.
	
};



template<typename T>
class QPANN : public PA<T>
{
	public :
	
	QPANN(const float& lr_, const float& eps_, const float& gamma_, const int& dimActionSpace_, const Topology& topo) : PA<T>(lr_,eps_), S1(Mat<T>((T)0,1,1)), A1(Mat<T>((T)0,1,1)), dimActionSpace(dimActionSpace_), gamma(gamma_)
	{
		this->net = new NN<T>(topo);
		srand(time(NULL));
		
		this->dimStateSpace = topo.getNeuronNumOnLayer(0);
	}
	
	~QPANN()
	{
		delete this->net;
	}
	
	
	virtual void initialize()	override
	{
		//let us handle the initialization of A1,S1 :
		this->A1 = Mat<T>((T)0,this->dimActionSpace,1);
		this->S1 = Mat<T>((T)0,this->dimStateSpace,1);
	}
	
	virtual Mat<T> estimateAction(const Mat<T>& S)	override
	{
		this->S1 = S;
		this->A1 = this->net->feedForward( this->S1 );
		
		return this->A1;
	}
	
	
	virtual void update(const Mat<T>& S, const Mat<T>& A, const Mat<T>& S1, const Mat<T>& R )	override
	{
		/*--------------------------------------------
		architecture is : Q(s,a) = NN( (s,a) )
		----------------------------------------------*/
		this->A1 = A;
		this->estimate(S,A);
		//Mat<float> Qtarget( ((T)(1.0f-this->lr))*this->Qsa + ((T)this->lr)*(R+gamma * this->estimate(S1, this->greedy(S1) ) ) );
		//Mat<float> Qtarget( (R+gamma * this->estimate(S1, this->greedy(S1) ) - this->Qsa) );
		Mat<float> Qtarget( (R+gamma * this->estimate(S1, this->greedy(S1) ) ) );

		//std::cout << "QTARGET = " << Qtarget.get(1,1) << std::endl;
		
		this->net->learning = true;
		this->net->feedForward( operatorC(S,A) );
		this->net->backProp( Qtarget );
		this->net->learning = false;
		
		
		this->greedyAlreadyComputed = true;
	}
	
	void update(const Mat<T>& input, const Mat<T>& target )
	{
		/*--------------------------------------------
		architecture is : Q(s,a) = NN( (s,a) )
		----------------------------------------------*/	
		//TODO : learning ? true false ?	
		this->net->feedForward( input );
		this->net->backProp( target );
		
		this->greedyAlreadyComputed = false;
	}
	
	
	virtual Mat<T> eps_greedy(const Mat<T>& S1)	override
	{
		float r = ((float)(rand()%100))/100.0f;
		
		if( r > this->eps)
		{
			//greedy :
			return this->greedy(S1);
		}
		else
		{
			//random :
			Mat<float> r(this->dimActionSpace,1);
			float max = 1e1f;
			for(int i=1;i<=this->dimActionSpace;i++)
			{
				r.set(  ( ((float)(rand()%10000)/1e5f) - ((float)(rand()%10000)/1e5f) )*max,i,1);
			}
			
			return r+this->greedy(S1)+Mat<T>((T)numeric_limits<T>::epsilon(),r.getLine(),r.getColumn());;
		}
	}

	
	virtual Mat<T> greedy(const Mat<T>& S1)	override
	{
		this->S1 = S1;
		int nbrit = 5;
		float alpha = 1e-1f;
		//TODO : evaluate the number of it needed.
		
		
		Mat<float> A(this->A1);
		Mat<float> grad( 1, dimActionSpace);
		int nbrDim = dimStateSpace+dimActionSpace;
		Mat<float> IdAction( 0.0f,nbrDim,dimActionSpace);
		for(int i=1;i<=dimActionSpace;i++)
		{
			IdAction.set( 1.0f, dimStateSpace + i, i);
		}
		
		while(nbrit--)
		{
			this->net->feedForward(operatorC(S1,A));
			grad = (this->net->getGradientWRTinput() * IdAction);
			
			A += alpha*transpose(grad);
		}
		
		this->A1 = A+numeric_limits<T>::epsilon()*Mat<T>((T)1,A.getLine(),A.getColumn());
		
		return this->A1;
		
	}
	

	
	NN<T>* getNetPointer()	const
	{
		return this->net;
	}
	
	Mat<T> getS1()	const
	{
		return S1;
	}
	
	private :
	
	NN<T>*	net;	
	float gamma;
	
	Mat<T> S1;			/*current state used in the estimation of the action*/
	Mat<T> A1;			/*at the beginning : current action estimated.*/
						  
	int dimStateSpace;
	int dimActionSpace;
	
};

template<typename T>
Mat<T> wrapperGREEDY( const Mat<T>& A, void* obj)
{
	return ((QFANN<T>*)obj)->getNetPointer()->feedForward( operatorC( ((QFANN<T>*)obj)->getS1(),A) );
}
 
#endif
