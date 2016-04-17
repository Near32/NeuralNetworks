#ifndef NN_H
#define NN_H

#include <iostream>
//#include <memory>
#include <vector>

#include "MAT/Mat.h"

typedef enum LAYERTYPE{
	LTINPUT,
	LTNORMAL,
	LTOUTPUT
}	LAYERTYPE;

	
class Topology
{
	public :
	
	Topology()
	{
		
	}
	
	~Topology()
	{
	
	}
	
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
	
	
	
	std::vector<unsigned int> NperNLayer;
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

template<typename T>
class Connection
{
	public :
	
	static T randomWeight(void)	{	return (T) (rand()/T(RAND_MAX) ) ;	}
	
	
	Connection() : idxConnection(0), previousLayer(NULL), nextLayer(NULL), W(Mat<T>((T)0,1,1)), dW(Mat<T>((T)0,1,1))
	{
	
	}
	
	
	Connection(Layer<T>* previousLayer_, Layer<T>* nextLayer_, unsigned int idxConnection_) : idxConnection(idxConnection_), previousLayer(previousLayer_), nextLayer(nextLayer_), W(Mat<T>(1,1)), dW(Mat<T>(1,1))
	{
		nbrIN = previousLayer->getNbrNeurons();
		previousLayer->setOutConnection(this);
		nbrOUT = nextLayer->getNbrNeurons();
		nextLayer->setInConnection(this);
		
		W = Mat<T>((T)0,nbrOUT,nbrIN+1);
		dW = Mat<T>((T)0,nbrOUT,nbrIN+1);
		
		for(int i=1;i<=W.getLine();i++)
		{
			for(int j=1;j<=W.getColumn();j++)
			{
				W.set( (T)randomWeight(), i,j);
			}
		}
	
	}
	
	
	Mat<T> getWeights()	const
	{
		return W;
	}
	
	Mat<T> getWeights(unsigned int idx)	const
	{
		return Line(W,idx);
	}
	
	private :
	
	unsigned int idxConnection;
	unsigned int nbrIN;		//without counting the bias.
	unsigned int nbrOUT;	//no bias to count since there is no connection to it.
	
	Layer<T>* previousLayer;
	Layer<T>* nextLayer;
	
	Mat<T> W;	// nbr Neuron on nextLayer X nbr Neuron on previousLayer+the biasNeuron... == nbrOUT X nbrIN+1
	Mat<T> dW;
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
	
	Neuron(Layer<T>* layer_, const unsigned int& idxNeuronOnLayer_) : m_output((T)0), m_activation((T)0), layer(layer_), idxNeuronOnLayer(idxNeuronOnLayer_)
	{
	
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
		m_output = sigmoid<T>( m_activation);
		return m_output;
	}
	
	T update(const Mat<T>& inputs)
	{
		computeActivation(inputs);
		return computeOutput();
	}
	
	protected :
	
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
	
	BiasNeuron()
	{
		this->m_output = (T)1;
	}
	
	BiasNeuron(const Layer<T>* layer_, const unsigned int& idxNeuronOnLayer_)
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
class Layer
{
	private :
	

	unsigned int nbrNeurons;
	//does not take into account the bias neuron...
	
	//std::vector< std::unique_ptr<Neuron<T> > > neurons;
	std::vector< Neuron<T>* > neurons;
	Mat<T> outputs;
	Mat<T> inputs;
	
	Connection<T>* inConnection;
	Connection<T>* outConnection;
	
	
	public :
	
	LAYERTYPE ltype;
	unsigned int idxLayer;	
	
	
	Layer(const LAYERTYPE& ltype_,const unsigned int& nbrNeurons_, unsigned int idxLayer_) : idxLayer(idxLayer_), ltype(ltype_), nbrNeurons(nbrNeurons_), outputs(Mat<T>((T)0,nbrNeurons,1))
	{
		for(int i=1;i<=nbrNeurons;i++)	neurons.push_back( new Neuron<T>(this,i) );
		
		//add the bias :
		neurons.push_back( (Neuron<T>*)( new BiasNeuron<T>(this,nbrNeurons+1) ) );
	}
	
	~Layer()
	{
		for(int i=neurons.size();i--;)
		{
			delete neurons[i];
		}
	}
	
	
	
	Mat<T> feedForward(const Mat<T>& inputs_)
	{
		inputs = inputs_;
		for(int i=0;i<=nbrNeurons;i++)
		{
			outputs.set( neurons[i]->update(inputs), i+1,1);
		}
		
		return outputs;
	}
	
	
	
	
	
	void setInConnection(Connection<T>* inc)
	{
		inConnection = inc;
	}
	
	void setOutConnection(Connection<T>* outc)
	{
		outConnection = outc;
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
	
	NN(const Topology& topo) : numLayer(topo.getNumLayer()),outputs(Mat<T>((T)0,topo.getNeuronNumOnLayer(numLayer-1),1))
	{
		
		
		//-------------------------------------------
		//-------------------------------------------
		
		//-------------------------------------------
		//-------------------------------------------
		
		//let us create the layers :
		
		for(unsigned int nl=0;nl<numLayer;nl++)
		{
			unsigned int currentLayerNbrNeuron = topo.getNeuronNumOnLayer(nl);
			m_layers.push_back( new Layer<T>( LTNORMAL, currentLayerNbrNeuron, nl ) );
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
			m_connections.push_back( new Connection<T>( m_layers[nl], m_layers[nl+1], nl ) );
			
			std::cout << "Made connections number " << nl << " between layers  " << nl << " and " << nl+1 << "." << std::endl;
			//std::cout << "Weights are : " << std::endl;
			//m_connections[nl]->getWeights().afficher();
			
		}	
		
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
	
	Mat<T> feedForward(const Mat<T>& inputs)
	{
		outputs = Mat<T>(inputs,(T)1,1,1,inputs.getLine()+1,1);
		for(int i=0;i<numLayer;i++)
		{
			outputs = m_layers[i]->feedForward(outputs);
			outputs = Mat<T>( outputs, (T)1, 1,1, outputs.getLine()+1, 1);
		}
		
		return outputs;
	}
	
	void backProp(const Mat<T>& target)
	{
	
	}
	
	void getOutputs(Mat<T>& outputs_)	const
	{
		outputs_ = m_layers[numLayer-1]->getOutputs();
	}
	
	
	private :
	
	unsigned int numLayer;
	Mat<T> outputs;
	
	vector<Layer<T>* > m_layers;	//m_layers[layerNum]
	vector<Connection<T>* > m_connections; //m_connections[connectionNum] / connectionNum == previousLayerNum
	
};

//---------------------------------------------
//---------------------------------------------
//---------------------------------------------


#endif//NN_H


