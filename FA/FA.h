#ifndef FA_H
#define FA_H

/*Function Approximator Interface */
#include "../MAT/Mat.h"
#include "../ONSCNewton/ONSCNewton.h"
#include "../NN.h"

template<typename T>
Mat<T> wrapperGREEDY( const Mat<T>& A, void* obj);

template<typename T>
class FA
{
	public :
	
	FA(const float& lr_, const float& eps_) : lr(lr_), eps(eps_)
	{
	
	
	}
	
	~FA()
	{
	
	}
	
	virtual void initialize()=0;
	
	virtual void update(const Mat<T>& S, const Mat<T>& A, const Mat<T>& S1, const Mat<T>& R ) = 0;
	
	virtual Mat<T> estimate(const Mat<T>& S, const Mat<T>& A) =0;
	
	virtual Mat<T> eps_greedy(const Mat<T>& S1) = 0;
	virtual Mat<T> greedy(const Mat<T>& S1) = 0;
	
	virtual Mat<T> getQvalue( const Mat<T>& S, const Mat<T>& A) = 0;
	
	protected :
	
	float lr;		//learning rate.
	float eps;		//epsilon greedy.
	
};



template<typename T>
class QFANN : public FA<T>
{
	public :
	
	QFANN(const float& lr_, const float& eps_, const float& gamma_, const int& dimActionSpace_, const Topology& topo) : FA<T>(lr_,eps_), Qsa(Mat<T>((T)0,1,1)), S1(Mat<T>((T)0,1,1)), A1(Mat<T>((T)0,1,1)), dimActionSpace(dimActionSpace_), gamma(gamma_), greedyAlreadyComputed(false)
	{
		this->net = new NN<T>(topo);
		srand(time(NULL));
		
		this->dimStateSpace = topo.getNeuronNumOnLayer(0)-this->dimActionSpace;
	}
	
	~QFANN()
	{
		this->net->save(std::string("CARTPOLENN") );
		delete this->net;
	}
	
	
	virtual void initialize()	override
	{
		//let us handle the initialization of A1 :
		this->A1 = Mat<T>((T)0,this->dimActionSpace,1);
	}
	
	virtual Mat<T> estimate(const Mat<T>& S, const Mat<T>& A)	override
	{
		this->Qsa = this->net->feedForward( operatorC(S,A) ) ;
		
		return this->Qsa;
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
		
		//this->net->learning = true;
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
		this->net->feedForward( input );
		this->net->backProp( target );
		
		this->greedyAlreadyComputed = false;
	}
	
	
	virtual Mat<T> eps_greedy(const Mat<T>& S1)	override
	{
		float r = ((float)(rand()%100))/100.0f;
		
		if( r>this->eps)
		{
			//greedy :
			//initialization is handle by the initialization method.
			//then the computation is done by the greedy method which is called during the update.
			if( ! this->greedyAlreadyComputed)
			{
				this->greedy(S1);
				this->S1 = S1;
			}
			
			return this->A1;
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
			
			return r+this->greedy(S1)+numeric_limits<T>::epsilon()*Mat<T>((T)1,r.getLine(),r.getColumn());;
		}
	}
	
	/*virtual Mat<T> greedy(const Mat<T>& S1)	override
	{
		this->S1 = S1;
		int nbrit = 3;
		//TODO : evaluate the number of it needed.
		bool approxHessian = false;
		//TODO : evaluate the need of the approximation.
		
		ONSCNewton<T> instance(  &(wrapperGREEDY), this->A1, nbrit, approxHessian, this);
		this->A1 = instance.getX();
		
		return this->A1;
		
	}
	*/
	
	virtual Mat<T> greedy(const Mat<T>& S1)	override
	{
		this->S1 = S1;
		int nbrit = 10;
		float alpha = 1e-4f;
		float momentum = 0.9f;
		//TODO : evaluate the number of it needed.
		
		
		Mat<float> A(this->A1);
		Mat<float> grad(0.0f, 1, dimActionSpace);
		int nbrDim = dimStateSpace+dimActionSpace;
		Mat<float> IdAction( 0.0f,nbrDim,dimActionSpace);
		for(int i=1;i<=dimActionSpace;i++)
		{
			IdAction.set( 1.0f, dimStateSpace + i, i);
		}
		
		while(nbrit--)
		{
			this->net->feedForward(operatorC(S1,A));
			grad = (1.0f-momentum)*(this->net->getGradientWRTinput() * IdAction) + momentum*grad;
			
			A += alpha*transpose(grad);
		}
		
		this->A1 = A+numeric_limits<T>::epsilon()*Mat<T>((T)1,A.getLine(),A.getColumn());
		
		return this->A1;
		
	}
	
	virtual Mat<T> getQvalue( const Mat<T>& S, const Mat<T>& A) override
	{
		return this->net->feedForward(operatorC(S,A));
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
	Mat<T> Qsa;			//current state action value.
	Mat<T> S1;			//current next state.
	Mat<T> A1;			/*at the beginning : current action used in the estimation of the current state action value.
						  at the end : argmax Q(S1,A1)*/
						  
	int dimStateSpace;
	int dimActionSpace;
	
	bool greedyAlreadyComputed;
	
};

template<typename T>
Mat<T> wrapperGREEDY( const Mat<T>& A, void* obj)
{
	return ((QFANN<T>*)obj)->getNetPointer()->feedForward( operatorC( ((QFANN<T>*)obj)->getS1(),A) );
}
 
#endif
