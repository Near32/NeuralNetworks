#ifndef NN_H
#define NN_H

#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
//#include <memory>
#include <vector>
#include <fstream>
#include <mutex>
#include <exception>
#include <typeinfo>

//#include "MAT/Mat.h"
#include "MAT/Mat2.h"
#include "RAND/rand.h"
#include "RunningStats/RunningStats.h"

#define DECAY 7e-1f
#define LEARNINGRATE 1e-2f

//#define sgd_use

#define outer_momentum

//#define gradient_check

//#define debuglvl1
//#define debuglvl2

/*TODO :
// topology :
	- save and load of the new variables for conv;
// connection :
	- save and load of the new variables for conv :
// layer :
	- save and load ???
*/


typedef enum LAYERTYPE{
	LTINPUT,
	LTNORMAL,
	LTOUTPUT,
	LTCONV
}	LAYERTYPE;


typedef enum NEURONTYPE{
	NTNONE,	//used for input layer...
	NTSIGMOID,
	NTSOFTMAX,
	NTTANH,
	NTRELU,
	NTCONV,
	NTPOOL
}	NEURONTYPE;

template<typename T>
void clampGradient(Mat<T>& m,T val );

template<typename T>
T softmaxNUM(const T& z)
{
	return (T)exp(z);
}

template<typename T>
Mat<T> softmaxM(const Mat<T>& z)
{
	Mat<T> r(z);
	
	for(int j=1;j<=r.getColumn();j++)
	{
		T denum = (T)numeric_limits<T>::epsilon();
		for(int i=1;i<=r.getLine();i++)
		{
			T val = softmaxNUM(r.get(i,j));
			r.set( val, i,j);
			denum = denum + val;
		}
		
		for(int i=1;i<=r.getLine();i++)
		{
			r.set( r.get(i,j)/denum, i,j);
		}
	}
	
	return r;
}

template<typename T>
Mat<T> softmaxGradM(const Mat<T>& z)
{
	Mat<T> one((T)(1), z.getLine(), z.getColumn());
	return softmaxM<T>(z) % (one - softmaxM<T>(z));
}

template<typename T>
Mat<T> tanhM(const Mat<T>& z)
{
	Mat<T> r(z);
	for(int i=1;i<=r.getLine();i++)
	{
		for(int j=1;j<=r.getColumn();j++)
		{
			T val = tanh(r.get(i,j));
			r.set( val, i,j);
		}
	}
	
	return r;
}

template<typename T>
Mat<T> tanhGradM(const Mat<T>& z)
{
	Mat<T> one((T)(1), z.getLine(), z.getColumn());
	return one - (tanhM<T>(z) % tanhM(z));
}

template<typename T>
Mat<T> identityM(const Mat<T>& z)
{
	return z;
}

template<typename T>
Mat<T> identityGradM(const Mat<T>& z)
{
	return Mat<T>(1.0f,z.getLine(),z.getColumn());
}

template<typename T>
T ReLU(const T& z)
{
	return ( z > (T)0 ? z : (T)0);
}

template<typename T>
Mat<T> ReLUM(const Mat<T>& z)
{
	Mat<T> r(z);
	for(int i=1;i<=r.getLine();i++)
	{
		for(int j=1;j<=r.getColumn();j++)
		{
			T val = ReLU(r.get(i,j));
			r.set( val, i,j);
		}
	}
	
	return r;
}


template<typename T>
T ReLUGrad(const T& z)
{
	return ( z > (T)0 ? (T)1 : (T)0);
}

template<typename T>
Mat<T> ReLUGradM(const Mat<T>& z)
{
	Mat<T> r(z);
	for(int i=1;i<=r.getLine();i++)
	{
		for(int j=1;j<=r.getColumn();j++)
		{
			T val = ReLUGrad(r.get(i,j));
			r.set( val, i,j);
		}
	}
	
	return r;
}

std::vector<std::string>& split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> parseString(const std::string& line, char delim)
{
    std::vector<std::string> elems;
    split(line, delim, elems);
    return elems;
}
	
typedef struct CLType
{
	CLTNORMAL,
	CLTINPUT,
	CLTCONV,
	CLTPOOL,
	CLTRELU
} CLType;

class convLayerTopoInfo
{
	public :
	
	convLayerTopoInfo(const CLType& cltype_=CLTCONV, const unsigned int& hi=28, const unsigned int& wi=28, const unsigned int& di=1, const unsigned int& hf=3, const unsigned int& wf=3, const unsigned int& df=1, const unsigned int& nbrf=32, const unsigned int& str=1, const unsigned int& p=0) : cltype(cltype_), Winput(wi),Hinput(hi), Dinput(di), Wfilter(wf),Hfilter(hf), Dfilter(df), nbrFilter(nbrf),Stride(str),Pad(p)
	{
		switch(cltype)
		{
			case CLTINPUT :
			{
				this->Woutput = this->Winput;
				this->Houtput = this->Hinput;
				this->Doutput = this->Dinput;
			}
			break;
			
			case CLTCONV:
			{
				this->Woutput = 1+ (Winput + 2*Pad - Wfilter)/Stride;
				this->Houtput = 1+ (Hinput + 2*Pad - Hfilter)/Stride;
				this->Doutput = nbrFilter;
			}
			break;
			
			case CLTPOOL :
			{
				this->Woutput = 1+ (Winput - Wfilter)/Stride;
				this->Houtput = 1+ (Hinput - Hfilter)/Stride;
				this->Doutput = this->Dinput;
			}
			break;
			
			case CLTRELU :
			{
				this->Woutput = this->Winput;
				this->Houtput = this->Hinput;
				this->Doutput = this->Dinput;
			}
			break;
			
			default:
			//CLTNORMAL? nothing to do : it is not a convolutionnal layer.
			break;
		}
	}
	
	~convLayerTopoInfo()
	{
	
	}
	
	
	//-------------------------------
	
	CLType cltype;
	
	unsigned int Winput;
	unsigned int Hinput;
	unsigned int Dinput;
	
	unsigned int Woutput;
	unsigned int Houtput;
	unsigned int Doutput;
	
	unsigned int Wfilter;
	unsigned int Hfilter;
	unsigned int Dfilter;	
	
	unsigned int nbrFilter;
	unsigned int Stride;
	unsigned int Pad;
};

//DUMMY instance used to initialize the normal layers that are not convolutionnal...
convLayerTopoInfo clti_NORMAL(CLTNORMAL,1,1,1,1,1,1);

class Topology
{
	public :
	
	Topology() : /*convLayer(false),nbrConvLayer(0),IX(0), IY(0), K0(0), nbrConvRELUperLayer(0)*/
	{
		
	}
	
	~Topology()
	{
	
	}
	
	void save( const std::string& path)	const
	{
		//TODO : save LTLayer and CLTILayer:
		
		std::ofstream myfile;
		myfile.open( path.c_str() );
		
		
		myfile << "NperNLayer ";
		for(int i=0;i<this->NperNLayer.size();i++)
		{
			myfile << this->NperNLayer[i] << " ";
		}
		
		myfile << std::endl;
		
		myfile << "NTLayer ";
		for(int i=0;i<this->NTLayer.size();i++)
		{
			myfile << (int)(this->NTLayer[i]) << " ";
		}
		
		myfile << std::endl;
		
		
		myfile.close();
	}
	
	void load( const std::string& path)
	{
		//TODO : load LTLayer and CLTILayer:
		std::ifstream myfile;
		myfile.open( path.c_str(),std::ifstream::in );
		
		std::string line;
		
		while(std::getline( myfile, line))
		{
			std::vector<std::string> el = parseString(line, (char)' ');
		
			for(int i=0;i<el.size();i++)
			{
				if(el[0] == "NperNLayer")
				{
					i++;
					while(i<el.size())
					{
						unsigned int val = std::stoi(el[i]);
						this->NperNLayer.push_back( val);
					
						i++;
					}
				}
		
				if(el[0] == "NTLayer")
				{
					i++;
					while(i<el.size() )
					{
						NEURONTYPE val = (NEURONTYPE) std::stoi(el[i]);
						this->NTLayer.push_back( val);
					
						i++;
					}
				}
			}
		
		}
		
		
		myfile.close();
	}
	
	void push_back( const unsigned int& nbrNeurons, const NEURONTYPE& ntype, const LAYERTYPE& ltype = LTNORMAL, const convLayerTopoInfo& clti = clti_NORMAL)
	{
		NperNLayer.push_back(nbrNeurons);
		NTLayer.push_back(ntype);
		LTLayer.push_back(ltype);
		CLTILayer.push_back(clti);
	}
	
	/*
	void addConvLayers(const unsigned int& IX, const unsigned int& IY, const unsigned int& K0, const std::vector<unsigned int>& K, const std::vector<unsigned int>& F, const std::vector<unsigned int>& S, const std::vector<unsigned int>& pF, const std::vector<unsigned int>& pS, const unsigned int& nbrConvLayer)
	{
		this->convLayer = true;
		
		this->IX = IX;
		this->IY = IY;
		this->K0 = K0;
		this->K = K;
		this->F = F;
		this->S = S;
		
		this->pF = pF;
		this->pS = pS;

		this->nbrConvLayer = nbrConvLayer;		
		this->nbrConvRELUperLayer = this->F.size()/this->nbrConvLayer;

	} 
	*/
	
	int getNumLayer()	const
	{
		return NperNLayer.size();
	}
	
	
	//from 0 to NperNLayer.size()-1
	int getNeuronNumOnLayer( unsigned int nl)	const
	{
		if( nl >= NperNLayer.size())
		{
			return 0;
		}
		
		return NperNLayer[nl];
	}
	
	NEURONTYPE getNeuronTypeOnLayer( unsigned int nl)	const
	{
		if( nl >= NTLayer.size())
		{
			throw;
		}
		
		return NTLayer[nl];
	}
	
	LAYERTYPE getLayerType(unsigned int nl)	const
	{
		if( nl >= LTLayer.size())
		{
			throw;
		}
		
		return LTLayer[nl];
	}
	
	convLayerTopoInfo* getCLTI(unsigned int nl)	const
	{
		if( nl >= CLTILayer.size() )
		{
			throw;
		}
		
		return CLTILayer[nl];
	}
	
	bool operator==( const Topology& otopo)
	{
		//TODO : handle the LTLayer and CLTILayer :
		
		bool ret = true;
		
		if( this->getNumLayer() == otopo.getNumLayer() )
		{
			int nbrlayer = NperNLayer.size();
			
			for(int i=nbrlayer;i--;)
			{
				if( this->getNeuronNumOnLayer(i) != otopo.getNeuronNumOnLayer(i) )
				{
					ret = false;
					break;
				}
			}
		}
		else
		{
			ret = false;
		}
		
		return ret;
	}
	
	//------------------------------------------------------
	//------------------------------------------------------
	//------------------------------------------------------
	//------------------------------------------------------
	//------------------------------------------------------
	//------------------------------------------------------
	
	std::vector<unsigned int> NperNLayer;
	std::vector<NEURONTYPE> NTLayer;
	
	
	//convolutions :
	//TODO : save those parameters...
	/*
	bool convLayer;
	unsigned int IX;
	unsigned int IY;
	unsigned int K0;
	
	unsigned int nbrConvRELUperLayer;
	unsigned int nbrConvLayer;
	
	std::vector<unsigned int> K;
	std::vector<unsigned int> F;
	std::vector<unsigned int> S;
	std::vector<unsigned int> pF;
	std::vector<unsigned int> pS;
	*/
	
	std::vector<LAYERTYPE> LTLayer;
	std::vector<convLayerTopoInfo> CLTILayer;
	
	
};


//---------------------------------------------
//---------------------------------------------
//---------------------------------------------
//early declaration...
template<typename T>
class Layer;

//---------------------------------------------
//---------------------------------------------
//---------------------------------------------




//---------------------------------------------
//---------------------------------------------
//---------------------------------------------
//early declaration :
template<typename T>
class NN;

NormalRand nrglobal(0.0f,1.0f,1234567);
std::mutex mutexNRGLOBAL;

template<typename T>
T randomWeight(void)	{	return (T) (rand()/T(RAND_MAX) )*1e-5f ;	}
template<typename T>
T randomWeightGaussian(void)	{	mutexNRGLOBAL.lock();
											T ret = (T) nrglobal.dev()*1e-3f ;
											mutexNRGLOBAL.unlock();
												return ret ;	}
template<typename T>
class Connection
{
	public :
	
	
	NormalRand* nr;
	std::mutex mutexNR;
	NN<T>* net;
	
	Connection() : idxConnection(0), previousLayer(NULL), nextLayer(NULL), W(Mat<T>((T)0,1,1)), dW(Mat<T>((T)0,1,1)), lastDeltaW(Mat<T>((T)0,1,1)),lr((T)LEARNINGRATE), decay((T)DECAY), lastBatchDeltaW(Mat<T>(1,1)), batchDeltaW(Mat<T>(1,1)), batchCounter(0), b(Mat<T>((T)0,1,1))
	{
		nr = new NormalRand( 0.0f, 10.0f, 1029);
		net = NULL;
	}
	
	
	Connection(Layer<T>* previousLayer_, Layer<T>* nextLayer_, unsigned int idxConnection_, const T& lr_ = (T)LEARNINGRATE ) : idxConnection(idxConnection_), previousLayer(previousLayer_), nextLayer(nextLayer_), W(Mat<T>(1,1)), dW(Mat<T>(1,1)), lastDeltaW(Mat<T>(1,1)),lr(lr_), decay((T)DECAY), lastBatchDeltaW(Mat<T>(1,1)), batchDeltaW(Mat<T>(1,1)), batchCounter(0)
	{
		net = previousLayer->net;
		
		nbrIN = previousLayer->getNbrNeurons();
		this->previousLayer->setBOOL_CLTI(true);
		previousLayer->setOutConnection(this);
		nbrOUT = nextLayer->getNbrNeurons();
		nextLayer->setInConnection(this);
		
		W = Mat<T>((T)0,nbrOUT,nbrIN+1);
		dW = Mat<T>((T)0,nbrOUT,nbrIN+1);
		lastDeltaW = W;
		batchDeltaW = W;
		lastBatchDeltaW = W;
		
		for(int i=1;i<=W.getLine();i++)
		{
			for(int j=1;j<=W.getColumn();j++)
			{
				W.set( (T)randomWeightGaussian(), i,j);
			}
		}
	
		nr = new NormalRand( 0.0f, 10.0f, 1029);
		
		b = Mat<T>((T)randomWeightGaussian(),1,1);
		
	}
	
	Connection(Layer<T>* previousLayer_, Layer<T>* nextLayer_, unsigned int idxConnection_, const Mat<T>& W_, const T& lr_ = (T)LEARNINGRATE) : idxConnection(idxConnection_), previousLayer(previousLayer_), nextLayer(nextLayer_), W(W_), dW(Mat<T>(0.0f,W_.getLine(),W_.getColumn())), lastDeltaW(Mat<T>(0.0f,W_.getLine(),W_.getColumn())), lr(lr_), decay((T)DECAY), lastBatchDeltaW(Mat<T>(0.0f,W_.getLine(),W_.getColumn())), batchDeltaW(Mat<T>(0.0f,W_.getLine(),W_.getColumn())), b(Mat<T>((T)0.0f,1,1))
	{
		net = previousLayer->net;
		nbrIN = previousLayer->getNbrNeurons();
		this->previousLayer->setBOOL_CLTI(false);
		previousLayer->setOutConnection(this);
		nbrOUT = nextLayer->getNbrNeurons();
		nextLayer->setInConnection(this);	
		
		nr = new NormalRand( 0.0f, 10.0f, 1029);
	}
	
	~Connection()
	{
		delete nr;
	}

	
	void backProp(const Mat<float> errorOut, const Mat<float>& actIn)
	{
		//errorOut must be a n(l+1)x1
		//actIn must be f(znl) = nlx1 so that
		// dW = n(l+1) x nl
		
		//Stochastic Gradient Descent  :

		Mat<float> sgd( dW.getLine(), dW.getColumn()-1);	//negative or positive random..
		this->dW = 0.0f*sgd;
		for(int k=1;k<=actIn.getColumn();k++)
		{
#ifdef sgd_use		
			for(int i=1;i<=sgd.getLine();i++)
			{
				for(int j=1;j<=sgd.getColumn();j++)
				{
					mutexNR.lock();
					float val = nr->dev();
					mutexNR.unlock();
					if( val > 	(float)0)
					{
						sgd.set( 1.0f, i,j);
					}
					else
					{
						sgd.set( 0.0f, i,j);
					}
				}
			}
		
			this->dW += sgd % ( Cola(errorOut,k) * transpose( Cola(actIn,k) ) );
#else
			this->dW += Cola(errorOut,k) * transpose( Cola(actIn,k) );
#endif		
			
		}

		
		//--------------------------------------------------------------------
		
		
		//Mat<float> ddW( operatorL( this->lr * (this->dW+ this->decay* extract(this->W,1,1,this->dW.getLine(),this->dW.getColumn()) ), this->lr * (errorOut) ) );
		this->dW = operatorL( this->lr * this->dW, this->lr * (errorOut) );
		Mat<float> deltaW( dW + this->decay * this->lastDeltaW  );

#ifdef gradient_check		
		std::cout << " NORME DELTA W : " << norme2(deltaW) << " CONNECTION : " << idxConnection << std::endl;
#endif		
		W -= deltaW;
		this->lastDeltaW = deltaW;
		//W -= Mat<float>( lr * (dW+decay* extract(W,1,1,dW.getLine(),dW.getColumn()) ), (float)0, 1,1, dW.getLine(), dW.getColumn()+1);
		//W -= Mat<float>( lr * (errorOut), (float)0, 1,dW.getColumn()+1, dW.getLine(), dW.getColumn()+1);
		NANregularizeWeights();
		
	}
	
	
	
	
	void backPropBATCH(const Mat<float> errorOut, const Mat<float>& actIn, int batchSize = 100)
	{
		//TODO : investigate the momentum formula ....
		//Stochastic Gradient Descent  :
#ifdef sgd_use		
		Mat<float> sgd( dW.getLine(), dW.getColumn());	//negative or positive random..
		for(int i=1;i<=sgd.getLine();i++)
		{
			for(int j=1;j<=sgd.getColumn();j++)
			{
				mutexNR.lock();
				float val = nr->dev();
				mutexNR.unlock();
				if( val > 	(float)0)
				{
					sgd.set( 1.0f, i,j);
				}
				else
				{
					sgd.set( 0.0f, i,j);
				}
			}
		}
		
		Mat<float> currentdW( sgd % (errorOut * transpose(actIn)) );
#else
		Mat<float> currentdW( errorOut * transpose(actIn) );
#endif		
		
		//--------------------------------------------------------------------
		
#ifndef outer_momentum		
		Mat<float> currentDeltaW( operatorL( 
				this->lr * currentdW,
				this->lr * (errorOut) ) );
				Mat<float> currentBatchDeltaW( currentDeltaW + this->decay * this->lastBatchDeltaW );
		//Mat<float> currentBatchDeltaW( (1.0f-this->decay) * currentDeltaW + this->decay * this->lastBatchDeltaW );
		this->lastBatchDeltaW = currentBatchDeltaW;
		
		this->batchDeltaW += currentBatchDeltaW;
#else
		Mat<float> currentBatchDeltaW( operatorL( 
				this->lr * currentdW,
				this->lr * (errorOut) ) );
		
		this->batchDeltaW += currentBatchDeltaW;
#endif
		this->batchCounter++;
		
		if( batchCounter >= batchSize)
		{			
#ifdef outer_momentum			
			//Mat<float> currentBatchDeltaW( (1.0f-this->decay) * this->batchDeltaW + this->decay * this->lastBatchDeltaW );
			Mat<float> currentBatchDeltaW( this->batchDeltaW + this->decay * this->lastBatchDeltaW );
			this->W -= currentBatchDeltaW;
			this->lastBatchDeltaW = currentBatchDeltaW;
#else
			this->W -= this->batchDeltaW;
#endif			
			
#ifdef gradient_check		
		std::cout << " NORME DELTA W : " << norme2(this->batchDeltaW) << " CONNECTION : " << idxConnection << std::endl;
#endif		
			this->net->rsNN.tadd( std::string("NORME dW : connection : ")+std::to_string(idxConnection), norme2(this->batchDeltaW) );			
			this->batchDeltaW *= 0.0f;
			this->batchCounter = 0;
			NANregularizeWeights();
		}
	}
	
	void applyGradient(const Mat<T>& grad)
	{
		if(grad.getLine() == this->W.getLine() && grad.getColumn() == this->W.getColumn())
		{
			this->W += NANregularize(grad);
		}
		else
		{
			throw;
		}
		
		clampGradient(this->W, 3e0f);
	}
	
	Mat<T> getWeights()	const
	{
		return W;
	}
	
	Mat<T> getBias()	const
	{
		return b;
	}
	
	Mat<T> getDeltaWeight()	const
	{
		return this->dW;
	}
	
	bool setWeights(const Mat<T>& newW)
	{
		/*
		if( newW.getLine() == this->W.getLine() && newW.getColumn() == this->W.getColumn() )
		{
			this->W = newW;
			return true;
		}
		
		return false;
		*/
		this->W = newW;
		return true;
	}
	
	bool setBias(const Mat<T>& newb)
	{
		if( newb.getLine() == this->b.getLine() && newb.getColumn() == this->b.getColumn() )
		{
			this->b = newb;
			return true;
		}
		
		return false;
	}
	
	Mat<T> getWeights(unsigned int idx)	const
	{
		return Line(W,idx);
	}
	
	
	
	void save( const std::string& path)	const
	{
		std::ofstream myfile;
		myfile.open( path.c_str() );
		
		for(int i=1;i<=this->W.getLine();i++)
		{
			for(int j=1;j<=this->W.getColumn();j++)
			{
				myfile << this->W.get(i,j) << " ";
			}
			
			myfile << std::endl;
		}
		
		myfile.close();
	}
	
	void load( const std::string& path)
	{
		std::ifstream myfile;
		myfile.open( path.c_str(),std::ifstream::in );
		
		std::string line;
		
		int i = 1;
		
		while( std::getline(myfile,line) )
		{
			std::vector<std::string> el = parseString(line, (char)' ' );
		
			for(int j=0;j<el.size();j++)
			{
				float val =0.0f;
				
				try
				{
					val = std::stof(el[j]);
				}
				catch(std::exception e)
				{
					std::cout << " EXCEPTION ON LOADING A VALUE : " << e.what() << " :: --" << el[j] << "--" << std::endl; 
				}
				
				this->W.set( val, i,j+1);		
			}
			
			i++;
			
		}
		
		
		myfile.close();
	}
	
	void NANregularizeWeights()
	{
		for(size_t i=1;i<=W.getLine();i++)
		{
			for(size_t j=1;j<=W.getColumn();j++)
			{
				if( isnan( W.get(i,j) ) )
				{
					//W.set( numeric_limits<T>::epsilon(), i,j);
					//let us reinitialized this weight :
					W.set( randomWeightGaussian(), i,j);
				}
			}
		}
	}
	
	private :
	
	unsigned int idxConnection;
	unsigned int nbrIN;		//without counting the bias.
	unsigned int nbrOUT;	//no bias to count since there is no connection to it.
	
	Layer<T>* previousLayer;
	Layer<T>* nextLayer;
	
	Mat<T> W;	// nbr Neuron on nextLayer X nbr Neuron on previousLayer+the biasNeuron... == nbrOUT X nbrIN+1
	Mat<T> dW;
	Mat<T> lastDeltaW;
	
	T lr;	//learning rate.
	T decay;
	
	int batchCounter;
	Mat<T> batchDeltaW;
	Mat<T> lastBatchDeltaW;
	
	
	//v2 parameter :
	Mat<T> b;
};


//---------------------------------------------
//---------------------------------------------
//---------------------------------------------



//---------------------------------------------
//---------------------------------------------
//---------------------------------------------


template<typename T>
class Neuron
{
	public :
		
	Neuron() : m_output((T)0), m_activation((T)0), layer(NULL), idxNeuronOnLayer(0)
	{
	
	}
	
	Neuron(Layer<T>* layer_, const unsigned int& idxNeuronOnLayer_, const NEURONTYPE& ntype_) : m_output((T)0), m_activation((T)0), layer(layer_), idxNeuronOnLayer(idxNeuronOnLayer_), ntype(ntype_)
	{
		//Initialization of the correct function to call :
		switch(ntype)
		{
			case NTSIGMOID :
			{
				this->function = sigmoid<T>;
			}
			break;
			
			case NTSOFTMAX :
			{
				this->function = softmaxNUM<T>;
			}
			break;
		}
	}
	
	~Neuron()
	{
	
	}
	
	virtual T computeActivation(const Mat<T>& inputs)
	{
		switch(layer->ltype)
		{
			default :
			{
			//let us retrieve the list of associate weights : idxNeuronOnLayer € [1;nbrNeurons]
			Mat<T> Wi( layer->getInConnection()->getWeights(idxNeuronOnLayer) );
		
			//Wi is a line, inputs are to be columns.
			m_activation = (Wi*inputs).get(1,1);
			}
			break;
			
			case LTINPUT :
			{
				m_activation = inputs.get(idxNeuronOnLayer,1);
			}
			break;
			
		}
			
		return m_activation;
	}
	
	virtual T computeOutput()
	{
		m_output = this->function( m_activation);
		return m_output;
	}
	
	T update(const Mat<T>& inputs)
	{
		computeActivation(inputs);
		return computeOutput();
	}
	
	T getActivation()	const
	{
		return m_activation;
	}
	
	T getOutput()	const
	{
		return m_output;
	}
	
	protected :
	
	NEURONTYPE ntype;
	T (*function)(const T&);
	
	T m_output;
	T m_activation;
	
	//T (ptrFuncAct2Out*)(void);
	//for now on, the activation2output function is a sigmoid :
	
	Layer<T>* layer;
	unsigned int idxNeuronOnLayer;
};


template<typename T>
class BiasNeuron : public Neuron<T>
{
	public :
	
	BiasNeuron() : Neuron<T>()
	{
		this->m_output = (T)1;
	}
	
	BiasNeuron( Layer<T>* layer_, const unsigned int& idxNeuronOnLayer_,const NEURONTYPE& ntype_) : Neuron<T>(layer_, idxNeuronOnLayer_, ntype_)
	{
		this->m_output = (T)1;
	}
	
	
	virtual T computeActivation(const Mat<T>& inputs)	override
	{
		this->m_activation = (T)0;
		return this->m_activation;
	}
	
	virtual T computeOutput()	override
	{
		this->m_output = (T)1;
		return this->m_output;
	}
	
	
	private :
	
	
};
//---------------------------------------------
//---------------------------------------------
//---------------------------------------------




//---------------------------------------------
//---------------------------------------------
//---------------------------------------------
template<typename T>
class NN;

template<typename T>
class Layer
{
	public :
	NN<T>* net;
	
	protected :
	
	unsigned int nbrNeurons;
	//does not take into account the bias neuron...
	
	Mat<T> outputs;
	Mat<T> activations;
	Mat<T> inputs;
	
	Mat<T> error;
	Connection<T>* inConnection;
	Connection<T>* outConnection;
	
	private:
	//std::vector< std::unique_ptr<Neuron<T> > > neurons;
	std::vector< Neuron<T>* > neurons;
		
	public :
	
	NEURONTYPE ntype;
	Mat<T> (*function)(const Mat<T>&);
	Mat<T> (*functionDerivative)(const Mat<T>&);
	
	LAYERTYPE ltype;
	unsigned int idxLayer;	
	
	
	bool initCLTI;
		
	
	Layer(NN<T>* net_, const LAYERTYPE& ltype_, const NEURONTYPE& ntype_, const unsigned int& nbrNeurons_, unsigned int idxLayer_, bool inherited = false) : net(net_), idxLayer(idxLayer_), ltype(ltype_), ntype(ntype_), nbrNeurons(nbrNeurons_), outputs(Mat<T>((T)0,nbrNeurons,1)), activations(Mat<T>((T)0,nbrNeurons+1,1)), error(Mat<T>((T)0,nbrNeurons+1,1))
	{
		if(!inherited)
		{
			for(int i=1;i<=nbrNeurons;i++)	neurons.push_back( new Neuron<T>(this,i, this->ntype) );
			//add the bias :
			neurons.push_back( (Neuron<T>*)( new BiasNeuron<T>(this,nbrNeurons+1, this->ntype) ) );
		}
		
		
		//Initialization of the correct function to call :
		switch(ntype)
		{
			case NTSIGMOID :
			{
				this->function = sigmoidM<T>;
				this->functionDerivative = sigmoidGradM<T>;
			}
			break;
			
			case NTSOFTMAX :
			{
				this->function = softmaxM<T>;
				this->functionDerivative = softmaxGradM<T>;
			}
			break;
			
			case NTTANH :
			{
				this->function = tanhM<T>;
				this->functionDerivative = tanhGradM<T>;
			}
			break;
			
			case NTRELU :
			{
				this->function = ReLUM<T>;
				this->functionDerivative = ReLUGradM<T>;
			}
			break;
			
			case NTNONE :
			{
				this->function = identityM<T>;
				this->functionDerivative = identityGradM<T>;
			}
		}
	}
	
	~Layer()
	{
		for(int i=neurons.size();i--;)
		{
			delete neurons[i];
		}
	}
	
	
	
	virtual Mat<T> feedForward(const Mat<T>& inputs_)
	{
		inputs = inputs_;
		
		switch(this->ntype)
		{
			case NTSIGMOID :
			{
				for(int i=0;i<=nbrNeurons;i++)
				{
					outputs.set( neurons[i]->update(inputs), i+1,1);
					activations.set( neurons[i]->getActivation(), i+1,1);
				}
			}
			break;
			
			case NTTANH :
			{
				for(int i=0;i<=nbrNeurons;i++)
				{
					outputs.set( neurons[i]->update(inputs), i+1,1);
					activations.set( neurons[i]->getActivation(), i+1,1);
				}
			}
			break;
			
			case NTSOFTMAX :
			{
				//first computation :
				for(int i=0;i<=nbrNeurons;i++)
				{
					outputs.set( neurons[i]->update(inputs), i+1,1);
					activations.set( neurons[i]->getActivation(), i+1,1);
				}
				
				//let us compute the normalizing value :
				T denum = 0.0f;	
				for(int i=nbrNeurons;i--;)	denum+=outputs.get(i,1);
				//do not take the bias neuron...
				
				//let us normalize those outputs values :
				for(int i=1;i<=nbrNeurons;i++)
				{
					outputs.set( outputs.get(i,1)/denum, i,1);
				}
				
				
				//TODO : do not forget that the neurons' output values are not normalized.
				//		only the layers' output are normalized.
			}
			break;
			
		}
		
		if(this->net->learning)
		{
			//--------------------------------------
		
			//TODO : implement dropout : on outputs and activities :
			unsigned int dropout = 0;
			int idx = 1;
			while( dropout != outputs.getLine()/2)
			{
				int idxbegin = idx;
				while( outputs.get(idx,1) == (float)0)
				{
					idx++;
					if( idx==idxbegin)	break;
					if( idx>outputs.getLine() )	idx = 1;
				}
				// assert that we are not dropping something that we have already dropped.
			
				//drop :
				outputs.set( (float)0, idx,1);
				//TODO : determine if we have to do it on the activations or not : apparently there is no need to.
				//activations.set( (float)0, idx, 1);
				dropout++;
				//-----------------------
			
				idx++;
				if( idx>outputs.getLine())	idx = 1;
			}
			
			outputs *= 2.0f;
			//TODO : determine if we have to do it on the activations or not : but apparently, we shouldn't.
			// or at least it cannot be multiplayed if it is not zeroed.
			//activations *= 2.0f;
			//-------------------------------------------
		}		
		
		//--------------------------------------
		
#ifdef debuglvl1
		std::cout << " LAYER : " << idxLayer << " : " << std::endl;
		transpose(outputs).afficher();
#endif		

		return outputs;
	}
	
	
	virtual Mat<T> backProp(const Mat<T>& errorPrev_)
	{
		Mat<float> actThis( extract(this->getActivations(), 1,1, nbrNeurons,1) );
		
		if(this->ltype == LTOUTPUT )
		{
			this->error = errorPrev_;
		}
		else
		{
				Mat<float> w( extract( outConnection->getWeights(), 1,1, errorPrev_.getLine(), actThis.getLine() ) );
				this->error = ( transpose(w) * errorPrev_) % functionDerivative( actThis );
				
				outConnection->backProp( errorPrev_, actThis);
		}
		
		return this->error;
	}
	
	virtual Mat<T> getDeltaWeight()	const
	{
		return this->outConnection->getDeltaWeight();
	}
	
	virtual void applyGradient(const Mat<T>& grad)
	{
		this->outConnection->applyGradient( grad);
	}
	
	
	
	virtual Mat<T> backPropBATCH(const Mat<T>& errorPrev_, int batchSize = 100)
	{
		Mat<float> actThis( extract(this->getActivations(), 1,1, nbrNeurons,1) );
		
		if(this->ltype == LTOUTPUT )
		{
			this->error = errorPrev_;
		}
		else
		{
				Mat<float> w( extract( outConnection->getWeights(), 1,1, errorPrev_.getLine(), actThis.getLine() ) );
				this->error = ( transpose(w) * errorPrev_) % functionDerivative( actThis );
				
				outConnection->backPropBATCH( errorPrev_, actThis, batchSize);
		}
		
		return this->error;
	}
	
	
	
	void setInConnection(Connection<T>* inc)
	{
		inConnection = inc;
	}
	
	void setOutConnection(Connection<T>* outc)
	{
		outConnection = outc;
		
		this->handleConvLayerTopoInfo();
	}
	
	virtual void setBOOL_CLTI(const bool& init)
	{
		initCLTI = init;
	}
	
	virtual void handleConvLayerTopoInfo() const
	{
	
	}
	
	Connection<T>* getInConnection()	const
	{
		return inConnection;
	}
	
	Connection<T>* getOutConnection()	const
	{
		return outConnection ;
	}
	
	
	unsigned int getNbrNeurons()	const
	{
		return nbrNeurons;
	}
	
	Mat<T> getOutputs()	const
	{
		return outputs;
	}
	
	Mat<T> getActivations()	const
	{
		return activations;
	}
};


template<typename T>
class Layer2 : public Layer<T>
{
	private :
	
	convLayerTopoInfo clti;
	Mat<Mat<T> > POOLmaxIDX;

	
	public :
	
	static Mat<T> max(const Mat<T> m)
	{
		Mat<T> ret(m(1,1,1),1,4);
		ret(1,2) = 1; 	//line
		ret(1,3) = 1;	//column
		ret(1,4) = 1;	//depth
		
		for(int k=1;k<=m.getDepth();k++)
		{
			for(int i=1;i<=m.getLine();i++)
			{
				for(int j=1;j<=m.getColumn();j++)
				{
					if( ret(1,1) < m(i,j,k))
					{
						ret(1,1) = m(i,j,k);
						ret(1,2) = i;
						ret(1,3) = j;
						ret(1,4) = k;
					}
				}
			}
		}
		
		return ret;
	}
	
	Layer2(NN<T>* net_, const LAYERTYPE& ltype_, const NEURONTYPE& ntype_, const unsigned int& nbrNeurons_, unsigned int idxLayer_,convLayerTopoInfo clti_ = clti_NORMAL ) : Layer<T>(net_,ltype_,ntype_, nbrNeurons_, idxLayer_,true), clti(clit_)
	{
		
	}
	
	~Layer2()
	{
		
	}
	
	virtual void handleConvLayerTopoInfo() const override
	{
		if(this->initCLTI && this->ltype == LTCONV)
		{
			if(this->clti != clti_NORMAL)
			{
				Mat<T> newWeights(clti.Hfilter,clti.Wfilter, clti.nbrFilter);
				Mat<T> newBias((T)randomWeightGaussian(), 1,1, clti.nbrFilter);
				
				for(int k=1;k<=newWeights.getDepth();k++)
				{
					for(int i=1;i<=newWeights.getLine();i++)
					{
						for(int j=1;j<=newWeights.getColumn();j++)
						{
							newWeights.set( (T)randomWeightGaussian(), i,j,k);
						}
					}
				}
				
				this->outConnection->setWeights(newWeights);
				this->outConnection->setBias(newBias);
			}
			else
			{
				throw;
			}
		}
	}
	
	virtual Mat<T> feedForward(const Mat<T>& inputs_)	override
	{
		this->inputs = inputs_;
		
		if(this->ltype != LTCONV)
		{
			switch(this->ntype)
			{
				default :
				{
					switch(this->ltype)
					{
						default :
						{
							this->activations = this->inConnection->getWeights()*this->inputs;
						}
						break;
			
						case LTINPUT :
						{
							
							if(this->inputs.getColumn() != 1)
							{
								//let us go from a convolutionnal layer to a fully connected layer :
								unsigned int l = this->inputs.getLine();
								unsigned int c = this->inputs.getColumn();
								unsigned int d = this->inputs.getDepth();
								Mat<T> temp( l*c*d, 1, 1);
								int idx = 1;
								for(int k=1;k<=d;k++)
								{
									for(int j=1;j<=c;j++)
									{
										for(int i=1;i<=l;i++)
										{
											temp(idx,1,1) = inputs(i,j,k);
											idx++; 
										}
									}
								}
								
								this->inputs = temp;
							}
							
							this->activations = this->inputs;
						}
						break;
			
					}
				
					this->outputs = this->function(this->activations);
				
				}
				break;
			
			
				case NTSOFTMAX :
				{
					switch(this->ltype)
					{
						default :
						{
							this->activations = this->inConnection->getWeights()*this->inputs;
						}
						break;
			
						case LTINPUT :
						{
							this->activations = this->inputs;
						}
						break;
			
					}
				
					this->outputs = this->function(this->activations);
				
					//let us compute the normalizing value :
					T denum = 0.0f;	
					for(int i=this->nbrNeurons;i--;)	denum+=this->outputs.get(i,1);
					//do not take the bias neuron...
				
					//let us normalize those outputs values :
					for(int i=1;i<=this->nbrNeurons;i++)	this->outputs.set( this->outputs.get(i,1)/denum, i,1);
				
				}
				break;
			
			}
		}
		else
		{
			//TODO : handle conv relu pool in clti.cltype...
			switch(this->clti.CLType)
			{
				case CLTPOOL :
				{
					unsigned int Ho = this->clti.Houtput;
					unsigned int Wo = this->clti.Woutput;
					unsigned int Do = this->clti.Doutput;
					unsigned int S = this->clti.Stride;
					
					this->activations = Mat<T>(Ho,Wo,Do);
					for(int k=1;k<=Do;k++)
					{
						for(int i=1;i<=Ho;i++)
						{
							for(int j=1;j<=Wo;j++)
							{
								this->POOLmaxIDX(i,j,k) = max( extract( this->inputs, (i-1)*S+1, (j-1)*S+1, c, (i-1)*S+1, (j-1)*S+1, c ) ) ;
								this->activations(i,j,k) = POOLmaxIDX(i,j,k)(1,1);
							}
						}
					}
					
					this->outputs = this->activations;
				}
				break;
				
				case CLTRELU :
				{
					this->activations = this->inputs;
					
					this->outputs = ReLUM(this->activations);	
				}
				break;
				
				case CLTINPUT :
				{
					this->activations = this->inputs;
					
					this->outputs = this->activations;	
				}
				break;
				
				case CLTCONV :
				{
					unsigned int Ho = this->clti.Houtput;
					unsigned int Wo = this->clti.Woutput;
					unsigned int Do = this->clti.Doutput;
					unsigned int S = this->clti.Stride;
					
					this->activations = padding(this->input,this->clti.Pad);
					Mat<T> W(this->inConnection->getWeights());
					Mat<T> b(this->inConnection->getBias());

					this->outputs = Mat<T>(Ho,Wo,Do);
					
					computeConv(outputs,activations,W,b,S);
					
					
				}
				break;
			}
		}
		
		if(this->net->learning)
		{
			//--------------------------------------
		
			//TODO : implement dropout : on outputs and activities :
			unsigned int dropout = 0;
			int idx = 1;
				
			while( dropout != this->outputs.getLine()/2 && !( this->outputs == Mat<float>(0.0f, this->outputs.getLine(), this->outputs.getColumn() ) ) )
			{
				int idxbegin = idx;
				while( this->outputs.get(idx,1) == (float)0 && !( this->outputs == Mat<float>(0.0f, this->outputs.getLine(), this->outputs.getColumn() ) ) ) 
				{
					idx++;
					if( idx==idxbegin)	break;
					if( idx>this->outputs.getLine() )	idx = 1;
				}
				// assert that we are not dropping something that we have already dropped.
		
				//drop :
				this->outputs.set( (float)0, idx,1);
				//TODO : determine if we have to do it on the activations or not : apparently there is no need to.
				//activations.set( (float)0, idx, 1);
				dropout++;
				//-----------------------
		
				idx++;
				if( idx>this->outputs.getLine())	idx = 1;
			}
		
			this->outputs *= 2.0f;
			//TODO : determine if we have to do it on the activations or not : but apparently, we shouldn't.
			// or at least it cannot be multiplayed if it is not zeroed.
			//activations *= 2.0f;
			//-------------------------------------------
		}		
		
		//--------------------------------------
#ifdef debuglvl1
		std::cout << " LAYER : " << this->idxLayer << " : " << std::endl;
		transpose(operatorL(this->activations,this->outputs)).afficher();
#endif		
		
		return this->outputs;
	}
	
	Mat<T> padding( const Mat<T>& m, const unsigned int P)
	{
		Mat<T> r((T)0,m.getLine()+P*2,m.getColumn()+P*2, m.getDepth());
		
		for(int k=1;k<=m.getDepth();k++)
		{
			for(int i=1;i<=m.getLine();i++)
			{
				for(int j=1;j<=m.getColumn();j++)
				{
					r(i+P,j+P,k) = m(i,j,k);
				}
			}
		}
		
		return r;
	}
	void computeConv(Mat<T>& outputs, const Mat<T>& activations, const Mat<T>& W, const Mat<T>& b, const unsigned int& S)
	{
		//each filter can be found along the depth indexes : so at each depth, the filter for the channels are stacked along the lines.
		unsigned int Ffilter = W.getColumn();
		unsigned int HW = W.getLine();
		unsigned int nbrChannel = HW/Ffilter;
		//assuming that filter are square !!!!
		unsigned int nbrFilter = W.getDepth();
		
		std::vector<Mat<T> > Ws;
		for(int i=1;i<=nbrChannel;i++)	Ws.push_back( extract(W, (i-1)*Ffilter+1, 1,1, i*Ffilter, Ffilter, nbrFilter) );
		
		for(int k=1;k<=outputs.getDepth();k++)
		{
			for(int i=1;i<=outputs.getLine();i++)
			{
				for(int j=1;j<=outputs.getColumn();j++)
				{
					T sum = b(1,1,k);
					
					for(int kk=1;kk<=activations.getDepth();kk++)
					{
						for(int ii=1;ii<=Ffilter;ii++)
						{
							for(int jj=1;jj<=Ffilter;jj++)
							{
								sum += Ws[kk](ii,jj,k)*activations(ii+(i-1)*S,jj+(j-1)*S, kk); 
							}
						}
					}
					
					outputs(i,j,k) = sum;
				}
			}
		}
	}
	
	virtual Mat<T> backProp(const Mat<T>& errorPrev_)	override
	{
		Mat<float> actThis( extract(this->activations, 1,1, this->nbrNeurons,this->activations.getColumn() ) );
		
		if(this->ltype == LTOUTPUT )
		{
			this->error = errorPrev_;
		}
		else
		{
				Mat<float> w( extract( this->outConnection->getWeights(), 1,1, errorPrev_.getLine(), actThis.getLine() ) );
				this->error = ( transpose(w) * errorPrev_) % this->functionDerivative( actThis );
				//error is of size : nl x 1
				//TODO : figuring out why is the delta left multiplied ?
				
				//this->outConnection->backProp( errorPrev_, actThis );
				this->outConnection->backProp( errorPrev_, this->function(actThis) );
		}
		
		return this->error;
	}
	
	
	
	virtual Mat<T> backPropBATCH(const Mat<T>& errorPrev_, int batchSize = 100)	override
	{
		Mat<float> actThis( extract(this->activations, 1,1, this->nbrNeurons,1) );
		
		if(this->ltype == LTOUTPUT )
		{
			this->error = errorPrev_;
		}
		else
		{
				Mat<float> w( extract( this->outConnection->getWeights(), 1,1, errorPrev_.getLine(), actThis.getLine() ) );
				this->error = ( transpose(w) * errorPrev_) % this->functionDerivative( actThis );
				
				this->outConnection->backPropBATCH( errorPrev_, actThis, batchSize);
		}
		
		return this->error;
	}
};
//---------------------------------------------
//---------------------------------------------
//---------------------------------------------















//---------------------------------------------
//---------------------------------------------
//---------------------------------------------

template<typename T>
class NN
{
	public :
	std::mutex mutexNN;
	
	
	//-----------------------------
	bool learning;
	Topology topology;
	T lr;
	
	std::string filepathRS;
	RunningStats<T> rsNN;
	//-----------------------------
	
	NN(const Topology& topo,  const T& lr_ = (T)LEARNINGRATE, const std::string& filepathRS_ = std::string("./NN.txt") ) : topology(topo), learning(false), numLayer(topo.getNumLayer()),outputs(Mat<T>((T)0,topo.getNeuronNumOnLayer(numLayer-1),1)), lr(lr_), filepathRS(filepathRS_),rsNN(RunningStats<T>(filepathRS))
	{
		
		//-------------------------------------------
		//-------------------------------------------
		
		//-------------------------------------------
		//-------------------------------------------
		
		//let us create the layers :
		
		for(unsigned int nl=0;nl<numLayer;nl++)
		{
			
			unsigned int currentLayerNbrNeuron = topo.getNeuronNumOnLayer(nl);
			NEURONTYPE currentLayerNeuronType = topo.getNeuronTypeOnLayer(nl);
			LAYERTYPE currentLayerType = topo.getLayerType(nl);
			convLayerTopoInfo currentCLTI = topo.getCLTI(nl);
			m_layers.push_back( new Layer2<T>(this, currentLayerType, currentLayerNeuronType, currentLayerNbrNeuron, nl, currentCLTI ) );
			std::cout << "Made a layer of " << currentLayerNbrNeuron << " Neurons and 1 BiasNeuron." << std::endl;
		}
		
		m_layers[0]->ltype = LTINPUT;
		m_layers[numLayer-1]->ltype = LTOUTPUT;
		
		//bias neurons are automatically inserted by the layers...
		
		
		//-------------------------------------------
		//-------------------------------------------
		
		//-------------------------------------------
		//-------------------------------------------
		
		//Let us create the connections :
		
		for(unsigned int nl=0;nl<numLayer-1;nl++)
		{
			m_connections.push_back( new Connection<T>( m_layers[nl], m_layers[nl+1], nl, this->lr ) );
			
			std::cout << "Made connections number " << nl << " between layers  " << nl << " and " << nl+1 << "." << std::endl;
			//std::cout << "Weights are : " << std::endl;
			//m_connections[nl]->getWeights().afficher();
			
		}	
		
	}
	
	NN(NN<T>* onet, const std::string& filepathRS_ = std::string("./NN.txt") ) : topology(onet->topology),learning(false), numLayer(topology.getNumLayer()),outputs(Mat<T>((T)0,topology.getNeuronNumOnLayer(numLayer-1),1)), lr(onet->lr), filepathRS(filepathRS_)
	{
		
		rsNN = RunningStats<T>(filepathRS);
		//-------------------------------------------
		//-------------------------------------------
		
		//-------------------------------------------
		//-------------------------------------------
		
		//let us create the layers :
		
		for(unsigned int nl=0;nl<numLayer;nl++)
		{
			unsigned int currentLayerNbrNeuron = topology.getNeuronNumOnLayer(nl);
			NEURONTYPE currentLayerNeuronType = topology.getNeuronTypeOnLayer(nl);
			LAYERTYPE currentLayerType = topology.getLayerType(nl);
			convLayerTopoInfo currentCLTI = topology.getCLTI(nl);
			m_layers.push_back( new Layer2<T>(this, currentLayerType, currentLayerNeuronType, currentLayerNbrNeuron, nl, currentCLTI ) );
			std::cout << "Made a layer of " << currentLayerNbrNeuron << " Neurons and 1 BiasNeuron." << std::endl;
		}
		
		m_layers[0]->ltype = LTINPUT;
		m_layers[numLayer-1]->ltype = LTOUTPUT;
		
		//bias neurons are automatically inserted by the layers...
		
		
		//-------------------------------------------
		//-------------------------------------------
		
		//-------------------------------------------
		//-------------------------------------------
		
		//Let us create the connections :
		
		std::vector<Connection<T>*> oconn = onet->getConnections();
		for(unsigned int nl=0;nl<numLayer-1;nl++)
		{
			m_connections.push_back( new Connection<T>( m_layers[nl], m_layers[nl+1], nl, oconn[nl]->getWeights(), this->lr ) );
			
			std::cout << "Made connections number " << nl << " between layers  " << nl << " and " << nl+1 << "." << std::endl;
			//std::cout << "Weights are : " << std::endl;
			//m_connections[nl]->getWeights().afficher();
			
		}	
		
	}
	
	NN(const std::string& filepath, const T& lr_ = (T)LEARNINGRATE, const std::string& filepathRS_ = std::string("./NN.txt") ) : topology(Topology()), learning(false), outputs(Mat<T>((T)0,1,1)), lr(lr_), filepathRS(filepathRS_)
	{
		rsNN = RunningStats<T>(filepathRS);
		this->load(filepath);
	}
	
	~NN()
	{
		for(int i=m_layers.size();i--;)
		{
			delete m_layers[i];
		}
		
		for(int i=m_connections.size();i--;)
		{
			delete m_connections[i];
		}
	}
	
	
	NN<T>& operator=( const NN<T>& onet)
	{
		if( this != &onet)
		{
			this->~NN();
			
			//filepath RS remains the same, but we create a new history of datas :
			rsNN = RunningStats<T>(filepathRS);
			this->topology = onet.topology;
			this->learning = false;
			this->numLayer = this->topology.getNumLayer();
			this->outputs = Mat<T>((T)0,this->topology.getNeuronNumOnLayer(numLayer-1),1);
			this->lr = onet.lr;				
		
			//-------------------------------------------
			//-------------------------------------------
		
			//-------------------------------------------
			//-------------------------------------------
		
			//let us create the layers :
		
			for(unsigned int nl=0;nl<numLayer;nl++)
			{
				unsigned int currentLayerNbrNeuron = topology.getNeuronNumOnLayer(nl);
				NEURONTYPE currentLayerNeuronType = topology.getNeuronTypeOnLayer(nl);
				m_layers.push_back( new Layer2<T>(this, LTNORMAL, currentLayerNeuronType, currentLayerNbrNeuron, nl ) );
				std::cout << "Made a layer of " << currentLayerNbrNeuron << " Neurons and 1 BiasNeuron." << std::endl;
			}
		
			m_layers[0]->ltype = LTINPUT;
			m_layers[numLayer-1]->ltype = LTOUTPUT;
		
			//bias neurons are automatically inserted by the layers...
		
		
			//-------------------------------------------
			//-------------------------------------------
		
			//-------------------------------------------
			//-------------------------------------------
		
			//Let us create the connections :
		
			std::vector<Connection<T>*> oconn = onet.getConnections();
			for(unsigned int nl=0;nl<numLayer-1;nl++)
			{
				m_connections.push_back( new Connection<T>( m_layers[nl], m_layers[nl+1], nl, oconn[nl]->getWeights(), this->lr ) );
			
				std::cout << "Made connections number " << nl << " between layers  " << nl << " and " << nl+1 << "." << std::endl;
			}
		}
			
		
		
		return *this;
	}
	
	Mat<T> feedForward(const Mat<T>& inputs)
	{
		mutexNN.lock();
		
		outputs = inputs;
		
		for(int i=0;i<numLayer;i++)
		{
			outputs = m_layers[i]->feedForward(outputs);
			if(m_layers[i]->
			outputs = operatorC(outputs, Mat<T>((T)1,1,outputs.getColumn()) );
		}
		
		Mat<float> ret( extract(outputs,1,1, outputs.getLine()-1,outputs.getColumn()) );
		mutexNN.unlock();
		
		return ret;
	}
	
	void backProp(const Mat<T>& target)
	{
		mutexNN.lock();
		
		Mat<float> actLast( extract( m_layers[numLayer-1]->getActivations(), 1,1, target.getLine(), target.getColumn() )  );
		//Mat<float> errorBackProp( ( (-1.0f)*(target-actLast) ) % ( m_layers[numLayer-1]->functionDerivative(actLast) )  );
		Mat<float> errorBackProp(  (m_layers[numLayer-1]->function(actLast) - target )  % ( m_layers[numLayer-1]->functionDerivative(actLast) )  );
		
		for(int i=numLayer;i--;)
		{
			errorBackProp = m_layers[i]->backProp(errorBackProp);
			//std::cout << " i = " << i << std::endl;
			//transpose(errorBackProp).afficher();
		}
		
		mutexNN.unlock();
	}
	
	void backPropBATCH(const Mat<T>& target, int batchSize = 100)
	{
		mutexNN.lock();
		
		Mat<float> actLast( extract( m_layers[numLayer-1]->getActivations(), 1,1, target.getLine(), 1)  );
		//Mat<float> errorBackProp( ( (-1.0f)*(target-actLast) ) % ( m_layers[numLayer-1]->functionDerivative(actLast) )  );
		Mat<float> errorBackProp(  (m_layers[numLayer-1]->function(actLast) - target )  % ( m_layers[numLayer-1]->functionDerivative(actLast) )  );
		
		for(int i=numLayer;i--;)
		{
			errorBackProp = m_layers[i]->backPropBATCH(errorBackProp, batchSize);
			//std::cout << " i = " << i << std::endl;
			//transpose(errorBackProp).afficher();
		}
		
		mutexNN.unlock();
	}
	
	void backPropDelta(const Mat<T>& delta)
	{
		Mat<float> actLast( extract( m_layers[numLayer-1]->getActivations(), 1,1, this->outputs.getLine()-1, 1)  );
		Mat<float> errorBackProp(  delta  % ( m_layers[numLayer-1]->functionDerivative(actLast) )  );
		
		for(int i=numLayer;i--;)
		{
			errorBackProp = m_layers[i]->backProp(errorBackProp);
			//std::cout << " i = " << i << std::endl;
			//transpose(errorBackProp).afficher();
		}
	}
	
	void backPropDeltaBATCH(const Mat<T>& delta, int batchSize = 100)
	{
		Mat<float> actLast( extract( m_layers[numLayer-1]->getActivations(), 1,1, this->outputs.getLine()-1, 1)  );
		Mat<float> errorBackProp(  delta  % ( m_layers[numLayer-1]->functionDerivative(actLast) )  );
		
		for(int i=numLayer;i--;)
		{
			errorBackProp = m_layers[i]->backPropBATCH(errorBackProp, batchSize);
			//std::cout << " i = " << i << std::endl;
			//transpose(errorBackProp).afficher();
		}
	}
	
	void backPropCrossEntropy(const Mat<T>& target)
	{
		Mat<float> errorBackProp( (target-extract(this->outputs,1,1,target.getLine(),1) )  );
		
		for(int i=numLayer;i--;)
		{
			errorBackProp = m_layers[i]->backProp(errorBackProp);
			//std::cout << " i = " << i << std::endl;
			//transpose(errorBackProp).afficher();
		}
	}
	
	
	Mat<T> getGradientWRTinput(Mat<T>* input = NULL)
	{
		if( input != NULL)	this->feedForward(*input);
		
		mutexNN.lock();
		
		Mat<T> actLast( extract( m_layers[numLayer-1]->getActivations(), 1,1, outputs.getLine()-1, 1)  );	//nbrOut x 1
		Mat<T> WLast( m_connections[numLayer-2]->getWeights() );	//nbrOut x nbrIn+1
		WLast = extract(WLast, 1,1, WLast.getLine(), WLast.getColumn()-1);	//nbrOut x nbrIn
		Mat<T> deltaF( m_layers[numLayer-1]->functionDerivative(actLast)  );
		Mat<T> gradF(0.0f, deltaF.getLine(), deltaF.getLine());
		for(int i=1;i<=deltaF.getLine();i++)	gradF.set( deltaF.get(i,1), i,i);
		
		Mat<T> grad( gradF*WLast);
		
		for(int i=numLayer-2;i>0;i--)
		{
			actLast = extract( m_layers[i]->getActivations(), 1,1, m_layers[i]->getNbrNeurons(), 1) ;	//nbrOut x 1
			WLast = m_connections[i-1]->getWeights();	//nbrOut x nbrIn+1
			WLast = extract(WLast, 1,1, WLast.getLine(), WLast.getColumn()-1);	//nbrOut x nbrIn
			deltaF = m_layers[i]->functionDerivative(actLast)  ;
			gradF = Mat<T>(0.0f, deltaF.getLine(), deltaF.getLine());
			for(int i=1;i<=deltaF.getLine();i++)	gradF.set( deltaF.get(i,1), i,i);
		
			grad = grad*(gradF*WLast);
		}
		
		mutexNN.unlock();
		
		return grad;
	}
	
	
	void getOutputs(Mat<T>& outputs_)	const
	{
		outputs_ = m_layers[numLayer-1]->getOutputs();
	}
	
	std::vector<Connection<T>*> getConnections()	const
	{
		return m_connections;
	}
	
	
	void save(const std::string& filepath)	const
	{
		//let us write the topology first :
		this->topology.save( filepath+std::string(".topo") );
		
		//let us save the weights :
		for(int i=m_connections.size();i--;)
		{
			this->m_connections[i]->save( filepath+std::string(".weights_")+std::to_string(i)  );
		}	
	}
	
	void load(const std::string& filepath)
	{
		this->topology.load( filepath+std::string(".topo") );
		
		
		//-------------------------------------------
		//-------------------------------------------
		
		this->learning = true;
		this->numLayer = this->topology.getNumLayer();
		this->outputs = Mat<T>((T)0, this->topology.getNeuronNumOnLayer(numLayer-1),1) ;		
		
		//-------------------------------------------
		//-------------------------------------------
		
		
		//let us create the layers :
		
		for(unsigned int nl=0;nl<numLayer;nl++)
		{
			unsigned int currentLayerNbrNeuron = this->topology.getNeuronNumOnLayer(nl);
			NEURONTYPE currentLayerNeuronType = this->topology.getNeuronTypeOnLayer(nl);
			LAYERTYPE currentLayerType = this->topology.getLayerType(nl);
			convLayerTopoInfo currentCLTI = this->topology.getCLTI(nl);
			m_layers.push_back( new Layer2<T>(this, currentLayerType, currentLayerNeuronType, currentLayerNbrNeuron, nl, currentCLTI ) );
			std::cout << "Made a layer of " << currentLayerNbrNeuron << " Neurons and 1 BiasNeuron." << std::endl;
		}
		
		m_layers[0]->ltype = LTINPUT;
		m_layers[numLayer-1]->ltype = LTOUTPUT;
		
		//bias neurons are automatically inserted by the layers...
		
		
		//-------------------------------------------
		//-------------------------------------------
		
		//-------------------------------------------
		//-------------------------------------------
		
		//Let us create the connections :
		
		for(unsigned int nl=0;nl<numLayer-1;nl++)
		{
			m_connections.push_back( new Connection<T>( m_layers[nl], m_layers[nl+1], nl, this->lr ) );
			
			std::cout << "Made connections number " << nl << " between layers  " << nl << " and " << nl+1 << "." << std::endl;
			//std::cout << "Weights are : " << std::endl;
			//m_connections[nl]->getWeights().afficher();
			
		}	
		
		//Let us load the connections'weights :
		
		for(int i=m_connections.size();i--;)
		{
			this->m_connections[i]->load( filepath+std::string(".weights_")+std::to_string(i)  );
		}
	}
	
	
	bool updateToward( NN<T>* onet, const T& momentum)
	{
		mutexNN.lock();
		
		//let us write the topology first :
		if( this->topology == onet->topology )
		{
			std::vector<Connection<T>*> onetconn = onet->getConnections();
			
			for(int i=m_connections.size();i--;)
			{
				Mat<T> newW( momentum * onetconn[i]->getWeights() );
				newW += (T)(1.0f-momentum) * this->m_connections[i]->getWeights();
				
				this->m_connections[i]->setWeights(newW);
			}
			
			mutexNN.unlock();
			
			return true;
		}
		
		std::cout << "ERROR : The Neural Networks are not of the same architecture...!" << std::endl;
		mutexNN.unlock();
		
		return false;	
	}
	
	
	protected :
	
	unsigned int numLayer;
	Mat<T> outputs;
	
	vector<Layer<T>* > m_layers;	//m_layers[layerNum]
	vector<Connection<T>* > m_connections; //m_connections[connectionNum] / connectionNum == previousLayerNum

};

//---------------------------------------------
//---------------------------------------------
//---------------------------------------------

//---------------------------------------------
//---------------------------------------------
//---------------------------------------------

typedef enum GRADIENTCOMP
{
	GCVanilla,
	GCSGD,
	GCRMSProp,
	GCSGDMomentum
}GRADIENTCOMP;

template<typename T>
class NNTrainer : public NN<T>
{
	public :
	
	NNTrainer( NN<T>* net_) : net(net_), NN<T>(net_->topology, net_->lr, net_->filepathRS+"TRAINER")
	{
		//initialization of the weights :
		this->updateToward(this->net, 1.0f);
		ConnGrad.resize(this->numLayer-1);
		dWs.resize(this->numLayer-1);
		for(int i=this->numLayer;i--;)
		{
			if( i< this->numLayer-1)
			{
				ConnGrad[i] =  this->m_layers[i]->getDeltaWeight();
			}
		}
		
	}
	
	~NNTrainer()
	{
	
	}
	
	
	bool updateNet( const T& momentum)
	{
		return this->net->updateToward(this, momentum);
	}
	
	void accumulateGradient2(const Mat<T>& input, const Mat<T>& delta)
	{
		//vanilla gradient computation :
		this->feedForward(input);
		
		//let us compute the deltas :
		this->deltas.clear();
		this->deltas.resize(this->numLayer+1);
		
		Mat<float> actLast( extract(this->m_layers[this->numLayer-1]->getActivations(), 1,1, this->outputs.getLine()-1, 1)  );
		//Mat<float> errorBackProp(  transpose(delta)  % ( this->m_layers[this->numLayer-1]->functionDerivative(actLast) )  );
		Mat<float> errorBackProp(  delta  * ( this->m_layers[this->numLayer-1]->functionDerivative(actLast) )  );
		deltas[this->numLayer] = errorBackProp;
		
		for(int i=this->numLayer;i--;)
		{
			deltas[i] = this->m_layers[i]->backProp( deltas[i+1]);
			
			if(isnanM(deltas[i]))
			{
				regularizeNanM( &deltas[i] );
			}
				
			if( i< this->numLayer-1)
			{
				dWs[i].push_back( this->m_layers[i]->getDeltaWeight() );
				
				if( isnanM(dWs[i][dWs[i].size()-1]) )
					regularizeNanM( &(dWs[i][dWs[i].size()-1]) );
			}
		}
		
		
	}
	
	void accumulateGradient(const Mat<T>& input, const Mat<T>& delta)
	{
		//delta must be nl x 1
		
		//vanilla gradient computation :
		this->feedForward(input);
		
		//let us compute the deltas :
		this->deltas.clear();
		this->deltas.resize(this->numLayer+1);
		
		Mat<float> actLast( extract(this->m_layers[this->numLayer-1]->getActivations(), 1,1, this->outputs.getLine()-1, 1)  );
		Mat<float> errorBackProp(  transpose(delta)  % ( this->m_layers[this->numLayer-1]->functionDerivative(actLast) )  );
		//Mat<float> errorBackProp(  delta  * ( this->m_layers[this->numLayer-1]->functionDerivative(actLast) )  );
		deltas[this->numLayer] = errorBackProp;
		
		for(int i=this->numLayer;i--;)
		{
			//deltas (nl) = nl x 1
			//deltas (n(l-1) ) = nl-1 x 1
			deltas[i] = this->m_layers[i]->backProp( deltas[i+1]);
			
			if(isnanM(deltas[i]))
			{
				regularizeNanM( &deltas[i] );
			}
				
			if( i< this->numLayer-1)
			{
				dWs[i].push_back( this->m_layers[i]->getDeltaWeight() );
				
				if( isnanM(dWs[i][dWs[i].size()-1]) )
					regularizeNanM( &(dWs[i][dWs[i].size()-1]) );
			}
		}
		
		
	}
	
	void operator()(GRADIENTCOMP gradComputationType = GCSGD,const T& momentum=(T)1.0f)
	{
		//computation of the gradients from the dWs:
		Mat<float> lastGrad;
		for(int i=this->numLayer-1;i--;)
		{
			//saving history...
			lastGrad = ConnGrad[i];
			ConnGrad[i] =((T)0.0f)*dWs[i][0];
			
			//computation of the new gradient :
			
			switch(gradComputationType)
			{
				case GCVanilla :
				{
					//vanilla update :
					ConnGrad[i] *= 0.0f;
					for(int j=dWs[i].size();j--;)
					{
						ConnGrad[i] += dWs[i][j];
					}
				}
				break;
				
				case GCSGD :
				{
					//SGD :
					ConnGrad[i] *= 0.0f;
					Mat<float> sgd(dWs[i][0]);
					for(int j=dWs[i].size();j--;)
					{
						//let us compute the sgd hadamard product :						
						for(int ii=1;ii<=sgd.getLine();ii++)
						{
							for(int jj=1;jj<=sgd.getColumn();jj++)
							{
								int val = rand()%2;
								if(val)
								{
									sgd.set( 1.0f,ii,jj);
								}
								else
								{
									sgd.set( 0.0f,ii,jj);
								}
							}
						}
						
						ConnGrad[i] += sgd % dWs[i][j];
					}
				}
				break;
				
				case GCSGDMomentum :
				{
					//SGD Momentum :
					float momentumMAJ = 0.9f;
					ConnGrad[i] = momentumMAJ*lastGrad;
					
					Mat<float> sgd(dWs[i][0]);
					for(int j=dWs[i].size();j--;)
					{
						//let us compute the sgd hadamard product :						
						for(int ii=1;ii<=sgd.getLine();ii++)
						{
							for(int jj=1;jj<=sgd.getColumn();jj++)
							{
								int val = rand()%2;
								if(val)
								{
									sgd.set( 1.0f,ii,jj);
								}
								else
								{
									sgd.set( 0.0f,ii,jj);
								}
							}
						}
						
						ConnGrad[i] += (1.0f-momentumMAJ) * ( sgd % dWs[i][j] );
					}
				}
				break;
				
				default :
				{
					//vanilla update :
					ConnGrad[i] *= 0.0f;
					for(int j=dWs[i].size();j--;)
					{
						ConnGrad[i] += dWs[i][j];
					}
				}
				break;
				
				//TODO : RMSprop ; MOMENTUM ; ...
				
			}
		}
		
		
		//application of the gradients :
		//beware of the -1 : update on the outConnection, no outConnection on the last layer.
		float clampingGrad = 1e-1f;
		for(int i=this->numLayer-1;i--;)
		{
			float norm = norme2(ConnGrad[i]);
			std::cout << " NORM GRAD : " << i << " :: " << norm << std::endl;
			
			if( !isfinite(norm) || norm > clampingGrad )
			{
				regularizeNanInfM(&(ConnGrad[i]));
				clampGradient(ConnGrad[i],clampingGrad);
				
				norm = norme2(ConnGrad[i]);
				std::cout << " REGULARIZED NORM GRAD : " << i << " :: " << norm << std::endl;
			}
			
			this->m_layers[i]->applyGradient( ConnGrad[i] );
			
		}
		
		//let us update the real net :
		bool success = this->updateNet(momentum);
		
		
		// reinitialization of the dW :
		dWs.clear();
		dWs.resize( this->numLayer-1);
		
		//ConnGrad are initialized at the old values, it is okay...
	}
	
	//TODO : batch version...
	
	
	private :
	
	NN<T>* net;									//pointer to the NN to train.
	std::vector<Mat<T> > deltas;				//delta of each each connection.
	std::vector<std::vector<Mat<T> > > dWs;		//deltaWeights associated with each connection.
	std::vector<Mat<T> > ConnGrad;				//gradient associated with each connection.
};





template<typename T>
void clampGradient(Mat<T>& m,T val )
{

	for(int i=m.getLine();i--;)
	{
	    for(int j=m.getColumn();j--;)
	    {
	        if( fabs(m.get(i+1,j+1)) > val )
	        {
	            if( m.get(i+1,j+1) > 0.0f)
	            {
	            	m.set( val, i+1,j+1);
	            }
	            else
	            {
	            	m.set( -val, i+1,j+1);
	            }
	        }
	    }
	}

}

//---------------------------------------------
//---------------------------------------------
//---------------------------------------------

//---------------------------------------------
//---------------------------------------------
//---------------------------------------------


template<typename T>
class NN2 : public NN<T>
{
	private :
	
	public :
	
	
};


template<typename T>
class CLayer
{
	public :
	
	CLayer(const convLayerTopoInfo& topo_) : topo(topo_), input(Mat<T>(topo.Hinput,topo.Winput,topo.Dinput)), output(Mat<T>(topo.Houtput,topo.Woutput,topo.Doutput)), activation(Mat<T>(topo.Houtput,topo.Woutput,topo.Doutput))
	{
	
		//DECLARATION :
		
		W.clear();
		for(int k=1;k<=topo.nbrFilter;k++)
		{
			W.push_back( Mat<T>(topo.Hfilter,topo.Wfilter,topo.Dfilter) );
		}
		
		b.clear();
		for(int k=1;k<=topo.nbrFilter;k++)
		{
			b.push_back( Mat<T>((T)randowmWeightGaussian(),1,1) );
		}
		
		
		
		//INITIALIZATION :
		
		for(int k=0;k<topo.nbrFilter;k++)
		{
			for(int i=1;i<=W[k].getLine();i++)
			{
				for(int j=1;j<=W[k].getColumn();j++)
				{
					W[k].set( (T)randomWeightGaussian(), i,j);
				}
			}
		}
		
		
		
		
	}
	
	protected :

	topoCLayer topo;
	
	std::vector<Mat<T> > W;
	std::vector<Mat<T> > b;
	
	Mat<T> input;
	Mat<T> activation;
	Mat<T> output;
	
	
	
};
//---------------------------------------------
//---------------------------------------------
//---------------------------------------------
//---------------------------------------------
//---------------------------------------------
//---------------------------------------------

#endif//NN_H


