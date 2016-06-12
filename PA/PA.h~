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
	std::mutex mutexPA;
	
	PA(const float& lr_, const float& eps_) : lr(lr_), eps(eps_)
	{
	
	
	}
	
	~PA()
	{
	
	}
	
	virtual void initialize()=0;
	
	virtual void update(const Mat<T>& S, const Mat<T>& A, const Mat<T>& S1, const Mat<T>& R ) = 0;
	virtual void updateBATCH(const Mat<T>& S, const Mat<T>& A, const Mat<T>& S1, const Mat<T>& R, int batchSize = 10 ) = 0;
	
	virtual	void update(const Mat<T>& input, const Mat<T>& target) = 0;
	virtual	void updateBATCH(const Mat<T>& input, const Mat<T>& target, int batchSize =10 ) = 0;
	
	virtual	void updateDelta(const Mat<T>& input, const Mat<T>& delta) = 0;
	virtual	void updateDeltaBATCH(const Mat<T>& input, const Mat<T>& delta, int batchSize =10 ) = 0;
	
	virtual Mat<T> estimateAction(const Mat<T>& S) =0;
	
	virtual Mat<T> eps_greedy(const Mat<T>& S) = 0;
	
	virtual NN<T>* getNetPointer()	const
	{
		return (NN<T>*)NULL;
	}
	
	float getLR()	const
	{
		return lr;
	}
	
	float getEPS()	const
	{
		return eps;
	}
	
	virtual bool updateToward(PA<T>* pa_, const T& momentum) = 0;
	
	
	virtual void setInputNormalization(const Mat<T>& mean, const Mat<T>& std)=0;
	
	virtual inline Mat<float> normalize( const Mat<float>& in)=0;
	
	protected :
	
	float lr;		//learning rate.
	float eps;		//epsilon greedy.
	
};



template<typename T>
class QPANN : public PA<T>
{
	public :
	
	int dimStateSpace;
	int dimActionSpace;
	std::string filepathRSNN;
	
	bool normalization;
	
	//-----------------------------
	//-----------------------------
	//-----------------------------
	
	//load from a topology :
	QPANN(const float& lr_, const float& eps_, const float& gamma_, const int& dimActionSpace_, const Topology& topo, const std::string& filepathRSNN_ = std::string("./NN.txt")) : PA<T>(lr_,eps_), S1(Mat<T>((T)0,1,1)), A1(Mat<T>((T)0,1,1)), dimActionSpace(dimActionSpace_), gamma(gamma_), filepathRSNN(filepathRSNN_), normalization(false)
	{
		this->noise = new NormalRand(0.0f,1.0f,1243);
		this->net = new NN<T>(topo, this->lr, filepathRSNN);
		srand(time(NULL));
		
		this->dimStateSpace = topo.getNeuronNumOnLayer(0);
		
		this->maxOutput = 100.0f;
	}
	
	//load from a file :
	QPANN(const float& lr_, const float& eps_, const float& gamma_, const int& dimActionSpace_, const std::string& filepath, const std::string& filepathRSNN_ = std::string("./NN.txt")) : PA<T>(lr_,eps_), S1(Mat<T>((T)0,1,1)), A1(Mat<T>((T)0,1,1)), dimActionSpace(dimActionSpace_), gamma(gamma_), filepathRSNN(filepathRSNN_), normalization(false)
	{
		this->noise = new NormalRand(0.0f,1.0f,12343);
		this->net = new NN<T>(filepath, this->lr, filepathRSNN);
		srand(time(NULL));
		
		this->dimStateSpace = this->net->topology.getNeuronNumOnLayer(0);
		
		this->maxOutput = 100.0f;
	}
	
	~QPANN()
	{
		delete this->net;
		delete noise;
	}
	
	
	virtual void initialize()	override
	{
		//let us handle the initialization of A1,S1 :
		this->S1 = Mat<T>((T)0,this->dimStateSpace,1);
		this->A1 = clipping(this->net->feedForward( (normalization? normalize( this->S1) : this->S1 ) ));
		
	}
	
	virtual Mat<T> estimateAction(const Mat<T>& S)	override
	{
		this->mutexPA.lock();
		this->S1 = S;
		this->A1 = clipping(this->net->feedForward( (normalization? normalize( this->S1) : this->S1 ) ));
		
		this->mutexPA.unlock();
		return this->A1;
	}
	
	
	virtual void update(const Mat<T>& S, const Mat<T>& A, const Mat<T>& S1, const Mat<T>& R )	override
	{
		/*--------------------------------------------
		architecture is : A(s) = NN( (s) )
		----------------------------------------------*/
		/*
		this->A1 = A;
		this->estimate(S,A);
		Mat<float> Qtarget( (R+gamma * this->estimate(S1, this->greedy(S1) ) ) );
		
		//TODO : decide if we need dropout or not...
		//this->net->learning = true;
		this->net->feedForward( operatorC(S,A) );
		this->net->backProp( Qtarget );
		//this->net->learning = false;
		*/
	}
	
	
	virtual void updateBATCH(const Mat<T>& S, const Mat<T>& A, const Mat<T>& S1, const Mat<T>& R, int batchSize = 10 )	override
	{
		/*--------------------------------------------
		architecture is : Q(s,a) = NN( (s,a) )
		----------------------------------------------*/
		/*
		this->A1 = A;
		this->estimate(S,A);
		//Mat<float> Qtarget( ((T)(1.0f-this->lr))*this->Qsa + ((T)this->lr)*(R+gamma * this->estimate(S1, this->greedy(S1) ) ) );
		//Mat<float> Qtarget( (R+gamma * this->estimate(S1, this->greedy(S1) ) - this->Qsa) );
		Mat<float> Qtarget( (R+gamma * this->estimate(S1, this->greedy(S1) ) ) );

		//std::cout << "QTARGET = " << Qtarget.get(1,1) << std::endl;
		
		//this->net->learning = true;
		this->net->feedForward( operatorC(S,A) );
		this->net->backPropBATCH( Qtarget, batchSize );
		this->net->learning = false;
		
		*/
	}
	
	virtual void update(const Mat<T>& input, const Mat<T>& target )	override
	{
		/*--------------------------------------------
		architecture is : A(s) = NN( (s) )
		----------------------------------------------*/	
		//TODO : learning ? true false ?	
		this->net->learning = false;
		this->net->feedForward( (normalization? normalize(input) : input ) );
		this->net->backProp( target );
		
	}
	
	virtual void updateBATCH(const Mat<T>& input, const Mat<T>& target, int batchSize )	override
	{
		/*--------------------------------------------
		architecture is : Q(s,a) = NN( (s,a) )
		----------------------------------------------*/	
		this->net->learning = false;	
		this->net->feedForward( (normalization? normalize(input) : input ) );
		this->net->backPropBATCH( target, batchSize );
		
	}
	
	virtual void updateDelta(const Mat<T>& input, const Mat<T>& delta )	override
	{
		/*--------------------------------------------
		architecture is : A(s) = NN( (s) )
		----------------------------------------------*/	
		//TODO : learning ? true false ?	
		this->net->learning = false;
		this->net->feedForward( (normalization? normalize(input) : input ) );
		//this->net->backPropDelta( (-1.0f)*delta );
		this->net->backPropDelta( delta );
	}
	
	virtual void updateDeltaBATCH(const Mat<T>& input, const Mat<T>& delta, int batchSize )	override
	{
		/*--------------------------------------------
		architecture is : A(s) = NN( (s )
		----------------------------------------------*/		
		this->net->learning = false;
		this->net->feedForward( (normalization? normalize(input) : input ) );
		//this->net->backPropDeltaBATCH( (-1.0f)*delta, batchSize );
		this->net->backPropDeltaBATCH( delta, batchSize );
		
	}
	
	/*
	virtual Mat<T> eps_greedy(const Mat<T>& S1)	override
	{
		
		float r = ((float)(rand()%100))/100.0f;
		
		if( r > this->eps)
		{
			//greedy :
			return this->estimateAction(S1);
		}
		else
		{
			//random :
			Mat<float> ret(this->dimActionSpace,1);
			float amount = 1e-1f;
			
			for(int i=1;i<=this->dimActionSpace;i++)
			{
				//ret.set(  ( ((float)(rand()%(int)1000) ) - ((float)(rand()%(int)1000) ) )*maxOutput/2e3f,i,1);
				ret.set( noise->dev()*maxOutput/2*amount, i,1);
			}
			
			return ret+this->estimateAction(S1);
		}
		
		return Mat<float>(1,1);
	}
	*/
	
	virtual Mat<T> eps_greedy(const Mat<T>& S1)	override
	{
		
		float r = ((float)(rand()%100))/100.0f;
		
		if( r > this->eps)
		{
			//greedy :
			return this->estimateAction( S1);
		}
		else
		{
			//random :
			Mat<float> ret(this->dimActionSpace,1);
			float amount = 1e-2f;	//1% of exploration...
			
			for(int i=1;i<=this->dimActionSpace;i++)
			{
				//ret.set(  ( ((float)(rand()%(int)1000) ) - ((float)(rand()%(int)1000) ) )*maxOutput/2e3f,i,1);
				ret.set( noise->dev()*maxOutput*amount, i,1);
			}
			
			return ret+this->estimateAction(S1);
		}
		
		return Mat<float>(1,1);
	}

	
	NN<T>* getNetPointer()	const
	{
		return this->net;
	}
	
	Mat<T> getS1()	const
	{
		return S1;
	}
	
	
	void save(const std::string& filepath)	const
	{
		net->save(filepath);
	}
	
	void load(const std::string& filepath)	const
	{
		net->load(filepath);
	}
	
	Mat<T> clipping(const Mat<T>& m)	const
	{
		Mat<T> r(m);
		for(int i=1;i<=r.getLine();i++)
		{
			for(int j=1;j<=r.getColumn();j++)
			{
				T val = r.get(i,j);
				if(isnan(val))
				{
					val = 0.0f;
				}
				
				if( fabs_(val) > maxOutput)
				{
					r.set( (val/fabs_(val))*maxOutput, i,j);
				}
			}
		}
		
		return r;
	}
	
	virtual bool updateToward(PA<T>* pa_, const T& momentum) override
	{
		this->mutexPA.lock();
		//pa_ must be a QPANN
		bool ret = net->updateToward( ((QPANN<T>*)pa_)->getNetPointer(), momentum);
		
		this->mutexPA.unlock();
		return ret;
	}
	
	virtual void setInputNormalization(const Mat<T>& mean, const Mat<T>& std) override
	{
		this->inputMean = mean;
		this->invinputStd = inverseM(std);
		this->normalization = true;
	}
	
	virtual inline Mat<float> normalize( const Mat<float>& in)	override
	{
		return (in-this->inputMean)%this->invinputStd;
	}
	
	private :
	
	NN<T>*	net;	
	float gamma;
	
	Mat<T> S1;			/*current state used in the estimation of the action*/
	Mat<T> A1;			/*at the beginning : current action estimated.*/
						  
	T maxOutput;
	NormalRand* noise;
	
	Mat<float> inputMean;
	Mat<float> invinputStd;
};
 
#endif
