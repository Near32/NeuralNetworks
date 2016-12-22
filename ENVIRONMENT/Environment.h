#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

//#include "../MAT/Mat.h"
#include "../MAT/Mat2.h"
#include "../RK4/RK4.h"
#include <python2.7/Python.h>
#include <mutex>
#include <thread>


template<typename T>
class Environment
{
	public :
	
	Environment(const float& EOE, const unsigned int& dss) : ENDOFEPISODE(EOE),dimStateSpace(dss), state(Mat<T>((T)0,dss,1))
	{
	
	}
	
	virtual ~Environment()
	{
	
	}
	
	virtual void initialize(bool random = false) =0;
	
	
	virtual Mat<T> executeAction(const Mat<T>& action) = 0;
	//TODO : return the reward due to the action...

	virtual bool isTerminal() =0;
	
	
	virtual Mat<T> getCurrentState()
	{
		return state;
	}

	virtual float getEOE()	const
	{
		return ENDOFEPISODE;
	}

	protected :
	
	float ENDOFEPISODE;
	unsigned int dimStateSpace;
	Mat<T> state;
	Mat<T> action;
	Mat<T> reward;
};



class SimulatorRK : public Environment<float>
{
	public :
	
    //swarmOfRobots;
    Mat<float> centerOfRotation;
    std::vector<float> desiredR;
    unsigned int nbrRobots;
    
    float currentTime;
    float timeStep;
    
    RK4* rk;
    
    Mat<float> initState;
    Mat<float> SIMstate;
    
    bool endSIMULATIONBAD;
    
    /*--------------------------------*/
    
    /* the state of the learning algo will contain :
    	-desired R1
    	- ...
    	- desired R_nbrRobot
    	-phi12
    	-...
    	-phi nbrRobot1
    	-a
    	-sigma
    	-kw
    	-kv
    	-couplingStrength
    	*/
    //SimulatorRK(const float& EOE, unsigned int nbrRobots_, const Mat<float>& CoR, std::vector<float> desiredR_) : Environment<float>(EOE,nbrRobots_*2+5),SIMstate(Mat<float>(0.0f,4*nbrRobots_+5,1))
    SimulatorRK(const float& EOE, unsigned int nbrRobots_, const Mat<float>& CoR, std::vector<float> desiredR_) : Environment<float>(EOE,nbrRobots_*2+5+2*nbrRobots_),SIMstate(Mat<float>(0.0f,4*nbrRobots_+5,1))
    {
       this->centerOfRotation = CoR;
       this->nbrRobots = nbrRobots_;
       this->desiredR = desiredR_;
       //this->swarmOfRobots = SwarmOfRobots(this->nbrRobots,this->centerOfRotation, this->desiredR);
       
       this->currentTime = 0.0f;
       this->timeStep = 0.01f;
       this->endSIMULATIONBAD = false;   
       
       this->rk = new RK4( SRupdate1four, this);
	}
	
	SimulatorRK(const SimulatorRK& sim) : Environment<float>(sim.getEOE(), sim.nbrRobots*2+5+2*sim.nbrRobots), SIMstate(Mat<float>(0.0f,4*sim.nbrRobots+5,1))
    {
       this->centerOfRotation = sim.centerOfRotation;
       this->nbrRobots = sim.nbrRobots;
       this->desiredR = sim.desiredR;
       //this->swarmOfRobots = SwarmOfRobots(this->nbrRobots,this->centerOfRotation, this->desiredR);
       
       this->currentTime = 0.0f;
       this->timeStep = 0.01f;
       this->endSIMULATIONBAD = false;   
       
       this->rk = new RK4( SRupdate1four, this);
	}
	
	~SimulatorRK()
	{
		delete this->rk;
	}
    
    void run(const float& timeStep, const float& EndTime)
    {
        if(timeStep > 0)
        {
            this->timeStep = timeStep;
        }
        
        //let us run the simulation :
        this->SIMstate = this->rk->solve(this->SIMstate,this->timeStep,EndTime);
        this->currentTime = EndTime;
        
/*% %             %let us retrieve the last result into the swarm of robots...
% %             %FORMAT : 4 x nbrRobots
% %             for i=1:this->nbrRobots
% %                 ristate = stateVector(: , i );
% %                 this->swarmOfRobots.setState(i, ristate );
% %             end
*/

        //let us retrieve the results into the robotStates :
        /*Mat<float> solvedRobotStates( this->rk.getRecording() );
        this->robotsStates = Parameter.empty(size(solvedRobotStates,2)/this->nbrRobots,0);
        for(int t=1;t<size(solvedRobotStates,2)/this->nbrRobots;t++)
        {
            int idxcolBegin = (t-1)*this->nbrRobots+1;
            int idxcolEnd = t*this->nbrRobots;
            Mat<float> currentSolvedRobotsStates( extract( solvedRobotStates, 1,idxcolBegin, solvedRobotStates.getLine(),idxcolEnd ) ); 
            this->robotsStates(t) = Parameter(currentSolvedRobotsStates);
        }*/
    }
    
    virtual void initialize(bool random = false)	override
    {
    	this->currentTime = 0.0f;
    	this->endSIMULATIONBAD = false;
    	//let us build the current stateVector :
        //FORMAT : 4 x nbrRobots
        //TODO : compute the random initialization state :
        float l = 1;
        float angleStep = 2*PI*l/this->nbrRobots;
		
		float initphi = angleStep;
		float inittheta = angleStep;
		float initrho = 0;
		//posStep = [desiredR(1)+rand(1,1)/rand(1,1);inittheta;initphi;initrho];
		Mat<float> posStep(4,1);
		posStep.set(this->desiredR[0]+0.3,1,1);
		//posStep.set(this->desiredR[1]+(float)(rand()%100)/100.0f,1,1);
		posStep.set(inittheta,2,1);
		posStep.set(initphi,3,1);
		posStep.set(initrho,4,1);
		
		Mat<float> initPos( posStep);
		for(int a=1;a<=nbrRobots-1;a++)
		{
		    //vec = [0;0;0;angleStep+rand(1,1)/rand(1,1)];
		    Mat<float> vec(0.0f,4,1);
		    vec.set(angleStep+0.1,4,1);
		    //vec.set(angleStep+(float)(rand()%100)/100.0f,4,1);
		    initPos = operatorL( initPos , extract(initPos, 1,initPos.getColumn(), initPos.getLine(),initPos.getColumn() ) + vec );
		}
		this->initState = operatorL(initPos, Mat<float>(0.0f,4,5+this->nbrRobots));
		//let us initialized the desired R values :
		for(int i=1;i<=this->nbrRobots;i++)	this->initState.set( this->desiredR[0], 1,this->nbrRobots+i);
		
		//--------------------
		// add the rest of the state :
		//initialize the state of the learning algo :
		//for now, the desired R are not changing...
		for(int i=1;i<=this->nbrRobots;i++)	this->state.set( this->desiredR[0], i,1);
		
		for(int i=1;i<=this->nbrRobots;i++)
		{
			float rhoi = this->SIMstate.get(4*i,1);
		    int ni = i+1;
		    if(ni > nbrRobots)
		    {
		        ni = 1;
			}
		    
		    float rhoni = this->SIMstate.get(4*ni,1);
		    float phiini = rhoni-rhoi;
		    
		    while( phiini > PI)
		    {
		    	phiini -= 2*PI;
		    }
		    while( phiini < -PI)
		    {
		    	phiini += 2*PI;
		    }
		    
		    this->state.set( phiini, this->nbrRobots+i,1);
		}
		
		//---------------------------------------------
		//adding the rest of the state :
		for(int i=1;i<=5;i++)
		{
			this->state.set( ( (float)(rand()%100) - (float)(rand()%100)) /100.0f, 2*this->nbrRobots+i,1);
		}
		
		/*
		-a
    	-sigma
    	-kw
    	-kv
    	-couplingStrength
    	*/
    	
		this->state.set(0.5f, 2*this->nbrRobots+2,1);
		this->state.set( (float)(rand()%100)/100.0f, 2*this->nbrRobots+3,1);	//kw must be positive if sigma is positive. 
		this->state.set(0.5f, 2*this->nbrRobots+5,1);	//l=1:N=3==> l/N = 1/3 epsilon must be positive.

		if(this->state.get(2*this->nbrRobots+5,1) >0)
		{
			this->state.set( (float)(rand()%100)/100.0f, 2*this->nbrRobots+4,1);
			this->state.set( (float)(rand()%100)/100.0f, 2*this->nbrRobots+1,1);
		}
		
		
		for(int i=1;i<=5;i++)
		{
			this->initState.set( this->state.get(2*this->nbrRobots+i,1), 1, 2*this->nbrRobots+i);
		}
		//----------------------------------------------
		
		//SECOND PART to State St : let us add the coordinate of the robots :
		Mat<float> coordRob(this->nbrRobots*2,1);
		for(int i=1;i<=coordRob.getLine();i++)
		{
			int idxRobot = i;
			coordRob.set( initState.get(1,idxRobot) ,i,1);
			i++;
			coordRob.set( initState.get(2,idxRobot) ,i,1);
		}
		for(int i=1;i<=this->nbrRobots*2;i++)
		{
			this->state.set( coordRob.get(i,1), i+2*this->nbrRobots+5, 1);
		}
		
		//--------------------------------------------
		
        this->SIMstate = this->initState;
        this->SIMstate.afficher();
        this->state.afficher();
        
        
        //let us initialize the solver :
        this->rk->initialize();
    }
	
	
	/* action space :
		-vdR1
		-vdR2
		-...
		-vdRnbrRobot
		-a
		-sigma
		-kw
		-kv
		-couplingStrength
	*/
		
		
		/*
		previously :
		-va
		-vsigma
		-vkw
		-vkv
		-vcouplingStrength
		*/
	virtual Mat<float> executeAction(const Mat<float>& action)	override
	{
		this->action = action;
		//TODO : figured out the granularity of time needed :
		float simTime = 0.1f;
		float EndTime = this->currentTime + simTime;
		
		
		//------------------------------
		// compute changes with regard to the action in the SIMstate and store it :
		for(int i=1;i<=this->nbrRobots+5;i++)
		{
			this->SIMstate.set( this->action.get(i,1), 2,this->nbrRobots+i);
		}
		
		//let us learn on the values and not the velocities...
		for(int i=1;i<=5;i++)
		{
			this->SIMstate.set( this->action.get(this->nbrRobots+i,1), 1,2*this->nbrRobots+i);
			this->SIMstate.set( 0.0f, 2,2*this->nbrRobots+i);
		}
		//------------------------------
		
		//------------------------------
		this->run( this->timeStep, EndTime); 
		//------------------------------
		
		//--------------------------------
		//compute the state and store it :
		//for now, the desired R are not changing...
		for(int i=1;i<=this->nbrRobots;i++)	this->state.set( this->SIMstate.get(1,this->nbrRobots+i), i,1);
		
		std::vector<float> phii(this->nbrRobots+1);
		float dPhi = 2*PI/this->nbrRobots;
		
		for(int i=1;i<=this->nbrRobots;i++)
		{
			float rhoi = this->SIMstate.get(4,i);
		    int ni = i+1;
		    if(ni > nbrRobots)
		    {
		        ni = 1;
			}
		    
		    float rhoni = this->SIMstate.get(4,ni);
		    float phiini = rhoni-rhoi;
		    
		    if( fabs_(phiini) > 100)
		    {
		    	phiini = 0.0f;
		    	this->SIMstate.set( 0.0f, 4,ni);
		    	this->SIMstate.set( 0.0f, 4,i);
		    	
		    	//update fail.... system instable : let us end the simulation here.
		    	endSIMULATIONBAD = true;
		    }
		    
		    while( phiini > PI)
		    {
		    	phiini -= 2*PI;
		    }
		    while( phiini < -PI)
		    {
		    	phiini += 2*PI;
		    }
		    
		    phii[i] =  phiini;
		    this->state.set( phiini, this->nbrRobots+i,1);
		}
		
		//let us retrieve the five parameters :
		for(int i=1;i<=5;i++)	this->state.set( this->SIMstate.get( 1, this->nbrRobots*2+i), this->nbrRobots*2+i, 1);
		
		Mat<float> coordRob(this->nbrRobots*2,1);
		for(int i=1;i<=coordRob.getLine();i++)
		{
			int idxRobot = i;
			coordRob.set( this->SIMstate.get(1,idxRobot) ,i,1);
			i++;
			coordRob.set( this->SIMstate.get(2,idxRobot) ,i,1);
		}
		for(int i=1;i<=this->nbrRobots*2;i++)
		{
			this->state.set( coordRob.get(i,1), i+2*this->nbrRobots+5, 1);
		}
		
		//state computation done.
		//--------------------------------
		
		//--------------------------------
		// compute the reward and store it
		float rewardCumulativePhi = 0.0f;
		for(int i=0;i<phii.size();i++)
		{
			rewardCumulativePhi += pow( phii[i]-dPhi, 2);
		}
		//rewardCumulativePhi = 1.0f/(1.0f+rewardCumulativePhi/this->nbrRobots);
		rewardCumulativePhi = -sqrt(rewardCumulativePhi/this->nbrRobots);
		//rewardCumulativePhi = 1.0f/(1.0f+rewardCumulativePhi);
		this->reward = Mat<float>(rewardCumulativePhi, 1,1);
		this->reward *= (float)(1.0f/(this->nbrRobots*PI));
		
		if(endSIMULATIONBAD)
		{
			this->reward *= 1e2f;
		}
		
		if(this->currentTime < this->ENDOFEPISODE)
		{
			this->reward *= 0.0f;
		}
		//--------------------------------
			
		
		return this->reward;
	}
	
	virtual bool isTerminal()	override
	{
		/*
		if(endSIMULATIONBAD)
		{
			return true;
		}
		*/
		
		return (this->currentTime > this->ENDOFEPISODE ? true : false);
	}
	
};



class SimulatorRKCARTPOLE : public Environment<float>
{
	protected :
	
    
    float currentTime;
    float timeStep;
    
    RK4* rk;
    
    Mat<float> initState;
    Mat<float> SIMstate;
    
    bool endSIMULATIONBAD;
    

    
    public :

    int idxAssociatedThread;  
    bool write;  
    /* the state of the learning algo will contain :
    	-x
    	-theta
    	-xdot
    	-thetadot
    	*/
    SimulatorRKCARTPOLE(const float& EOE)  : Environment<float>(EOE,4),SIMstate(Mat<float>(0.0f,5,1))
    {
       
       this->currentTime = 0.0f;
       this->timeStep = 0.01f;
       this->endSIMULATIONBAD = false;   
       
       this->rk = new RK4( CARTPOLEUPDATE, this);
       
       this->idxAssociatedThread = 0;
       this->write = false;
	}
	
	~SimulatorRKCARTPOLE()
	{
		delete this->rk;
	}
    
    void run(const float& timeStep, const float& EndTime)
    {
        if(timeStep > 0)
        {
            this->timeStep = timeStep;
        }
        
        //let us run the simulation :
        this->SIMstate = this->rk->solve(this->SIMstate,this->timeStep,EndTime);
        this->currentTime = EndTime;
        
    }
    
    /*Mat<float> robotsStates = getRobotsStates(obj)
    {
       robotsStates = this->robotsStates; 
    }*/
    
    virtual void initialize(bool random = false)	override
    {
    	this->currentTime = 0.0f;
    	this->endSIMULATIONBAD = false;
    	//let us build the current stateVector :
        //FORMAT : 4 x nbrRobots
        //TODO : compute the random initialization state :
        float initTheta = PI;
        //float initTheta = 0.0f;
        
        if(random)
        	initTheta = ((float)(rand()%360))/360.0f*PI;
        
        while( initTheta > 2*PI)
        {
        	initTheta -= 2*PI;
        }
        while( initTheta < 0.0f)
        {
        	initTheta += 2*PI;
        }
        
        //std::cout << " ENVIRONMENT :: " << " THETA INIT = " << initTheta << std::endl;
        
		this->initState = Mat<float>(0.0f,4,1);
		this->initState.set( initTheta, 2,1);
		//TODO : no random initial velocity...
		this->initState.set( ((float)(rand()%100))/100.0f, 4,1);
		//--------------------
		// add the rest of the state :
		//initialize the state of the learning algo :
		this->state = initState;
		
		//---------------------------------------------
		//----------------------------------------------
		
        this->SIMstate = operatorC(this->initState,Mat<float>(0.0f,1,1) );
        //this->SIMstate.afficher();
        //this->state.afficher();
        
        
        //let us initialize the solver :
        this->rk->initialize(this->idxAssociatedThread,this->write);
    }
	
	
	/* action space :
		-F
	*/
	virtual Mat<float> executeAction(const Mat<float>& action)	override
	{
		this->action = action;
		//TODO : figured out the granularity of time needed :
		float simTime = 0.1f;
		float EndTime = this->currentTime + simTime;
		
		
		//------------------------------
		// compute changes with regard to the action in the SIMstate and store it :
		this->SIMstate.set( this->action.get(1,1), 5,1);
		//------------------------------
		
		//------------------------------
		this->run( this->timeStep, EndTime); 
		//------------------------------
		
		//--------------------------------
		//compute the state and store it :
		for(int i=1;i<=4;i++)	this->state.set( this->SIMstate.get(i,1), i,1);
		//--------------------------------
		//--------------------------------
		
		//--------------------------------
		// compute the reward and store it
		float theta = this->state.get(2,1);
		float xpos = this->state.get(1,1);
		
		if( fabs_(xpos) > 100.0f)
		{
			endSIMULATIONBAD = false;
		}
		
		while( theta > 2*PI)
		{
			theta-=2*PI;
		}
		while( theta < 0.0f)
		{
			theta += 2*PI;
		}
		state.set( theta, 2,1);
		//regularize in the SIMSTATE...
		this->SIMstate.set(theta, 2,1);
		
		float rewardCumulative = -pow(PI-theta,2);
#ifdef ENVrestrictXPOS
		rewardCumulative /= pow(PI,2);
		float usedxpos = xpos;
		/*
		if(fabs_(xpos) < 10.0f)
		{
			if(xpos > 0)
			{
				usedxpos = 10.0f;
			}
			else
			{
				usedxpos = -10.0f;
			}
		}
		*/
		float beta = 5.0f;
		rewardCumulative += -pow(usedxpos,2)*beta;
#endif	

#ifdef ENVpenalizeFORCE
		float force = action.get(1,1);
		float sigma = 5.0f;
		rewardCumulative += -fabs_(force*sigma);
#endif	
		//rewardCumulative = 1.0f/(1.0f+rewardCumulative);
		this->reward = Mat<float>(rewardCumulative, 1,1);
		
		if(endSIMULATIONBAD )
		{
			//this->reward *= 0.0f;
			
			for(int ii=1;ii<=this->reward.getLine();ii++)
			{
				for(int jj=1;jj<=this->reward.getColumn();jj++)
				{
					this->reward.set( -2e3f, ii,jj);
				}
			}
			
		}
		
		//this->reward *= 1e0f/2.0f;
		
		//reward only if it is greater than 0.4 :
		if( false )// this->reward.get(1,1) < 0.30f)
		{
			this->reward *= 0.0f;
			//this->reward.set( (0.4f-this->reward.get(1,1))*(-1.0f), 1,1);
		}
		
		//--------------------------------
			
		
		return this->reward;
	}
	
	virtual bool isTerminal()	override
	{
		if(endSIMULATIONBAD)
		{
			return true;
		}
		
		return (this->currentTime > this->ENDOFEPISODE ? true : false);
	}
	
};


//3 possible actions : -1 Newton, 0 , 1 Newton of Force applied.
class SimulatorRKCARTPOLE_Discrete : public SimulatorRKCARTPOLE
{
	protected :
	   
    public :
    
    /* the state of the learning algo will contain :
    	-x
    	-theta
    	-xdot
    	-thetadot
    	*/
    SimulatorRKCARTPOLE_Discrete(const float& EOE) : SimulatorRKCARTPOLE(EOE)
    {
      
	}
	
	~SimulatorRKCARTPOLE_Discrete()
	{
		
	}
	
	
	/* action space :
		-F
	*/
	virtual Mat<float> executeAction(const Mat<float>& action)	override
	{
		//discrete case :
		int max = idmin( (-1.0f)*action ).get(1,1);
		switch(max)
		{
			case 1 :
			{
				this->action = Mat<float>((-1.0f), 1,1);
			}
			break;
			
			case 3 :
			{
				this->action = Mat<float>((1.0f), 1,1);
			}
			break;
			
			default :
			{
				this->action = Mat<float>(0.0f, 1,1);
			}
			break;
			
		}
		
		//TODO : figured out the granularity of time needed :
		float simTime = 0.01f;
		float EndTime = this->currentTime + simTime;
		
		
		//------------------------------
		// compute changes with regard to the action in the SIMstate and store it :
		this->SIMstate.set( this->action.get(1,1), 5,1);
		//------------------------------
		
		//------------------------------
		this->run( this->timeStep, EndTime); 
		//------------------------------
		
		//--------------------------------
		//compute the state and store it :
		for(int i=1;i<=4;i++)	this->state.set( this->SIMstate.get(i,1), i,1);
		//--------------------------------
		//--------------------------------
		
		//--------------------------------
		// compute the reward and store it
		float theta = this->state.get(2,1);
		float xpos = this->state.get(1,1);
		
		if( fabs_(xpos) > 100.0f)
		{
			endSIMULATIONBAD = true;
		}
		
		while( theta > 2*PI)
		{
			theta-=2*PI;
		}
		while( theta < 0.0f)
		{
			theta += 2*PI;
		}
		state.set( theta, 2,1);
		
		float rewardCumulative = pow(PI-theta,2);
		rewardCumulative = 1.0f/(1.0f+rewardCumulative);
		this->reward = Mat<float>(rewardCumulative, 1,1);
		
		if(endSIMULATIONBAD )//|| rewardCumulative < 0.8f)
		{
			this->reward *= 0.0f;
			
			for(int ii=1;ii<=this->reward.getLine();ii++)
			{
				for(int jj=1;jj<=this->reward.getColumn();jj++)
				{
					this->reward.set( -1.0f, ii,jj);
				}
			}
			
		}
		
		this->reward *= 1e0f/2.0f;
		//--------------------------------
			
		
		return this->reward;
	}
	
	
};



class OPENAIGYM_Environment : public Environment<float>
{

	public :
	
	bool continuer;
	std::mutex rmutex;
	int nbrit;
	int counterit;
	
	OPENAIGYM_Environment(const std::string& env_name_, int nbriteration, int nbrstate_, int nbraction_, int argc_, char* argv_[]) : Environment<float>(10.0f,nbrstate_), env_name(env_name_), A(Mat<float>(0.0f,nbraction_,1)), nbrstate(nbrstate_),nbraction(nbraction_), argc(argc_), argv(argv_), operationhandled(false), nbrit(nbriteration), counterit(0)
	{
		this->initEnv = std::string("initEnvironment");
		this->init = std::string("init");
		this->execAction = std::string("execAction");
		this->getState = std::string("getState");
		
		this->operation[initEnv] = true;
		this->operation[init] = false;
		this->operation[execAction] = false;
		this->operation[getState] = false;
		
		std::cout << " OPENAIGYM ENVIRONMENT ::::::::::::::::::: INITIALIZED :::::::::::::::::::::::::::" << std::endl;
		
		this->threadLoop = new std::thread( &OPENAIGYM_Environment::loop, std::ref(*this) );
		
		std::cout << " OPENAIGYM ENVIRONMENT ::::::::::::::::::: LAUNCHED :::::::::::::::::::::::::::" << std::endl;
	}
	
	~OPENAIGYM_Environment()
	{
		delete this->threadLoop;
	}
	
	virtual void initialize(bool random = false) override
	{
		rmutex.lock();
		this->operation[init] = true;
		this->counterit = 0;
		
		this->operationhandled = false;
		rmutex.unlock();
		
		rmutex.lock();
		while( !(this->operationhandled) )
		{
			rmutex.unlock();
			rmutex.lock();
		}
		rmutex.unlock();
		
		
	}
	
	virtual Mat<float> executeAction(const Mat<float>& action) override
	{
		rmutex.lock();
		this->operation[execAction] = true;
		this->A = action;
		this->counterit++;
		
		this->operationhandled = false;
		rmutex.unlock();
		
		rmutex.lock();
		while( !(this->operationhandled) )
		{
			rmutex.unlock();
			rmutex.lock();
		}
		rmutex.unlock();
		
		return this->reward;
	}
	
	virtual Mat<float> getCurrentState()	override
	{
		rmutex.lock();
		this->operation[this->getState] = true;		
		
		this->operationhandled = false;
		rmutex.unlock();
		
		rmutex.lock();
		while( !(this->operationhandled) )
		{
			rmutex.unlock();
			rmutex.lock();
		}
		rmutex.unlock();
		
		return this->state;
	}
	
	virtual bool isTerminal()
	{
		return this->isDone;
	}
	
	
	void loop()
	{
		Py_SetProgramName(argv[0]);  /* optional but recommended */
		Py_Initialize();
		PySys_SetArgv(argc, argv); // must call this to get sys.argv and relative imports
		PyRun_SimpleString("import os, sys\n"
				         "print sys.argv, \"\\n\".join(sys.path)\n"
				         "print os.getcwd()\n"
				         "import list2numpy\n" );
				         
		std::cout << " OPENAIGYM ENVIRONMENT :::::::::::::::::::::::: LOOP STARTED ::::::::::::::::::::" << std::endl;
		
		while(continuer)
		{
			if( this->operation[initEnv] )
			{
		
				PyObject *pName = PyString_FromString("gym");   
				this->pl2n = PyImport_ImportModule("list2numpy");
				this->pFuncl2n = PyObject_GetAttrString(pl2n,"list2numpy");              
						         
				this->pModule = PyImport_Import(pName);
	
				if (this->pModule != NULL) 
				{
					PyObject *pFunc = PyObject_GetAttrString(this->pModule, "make");
				
					/* pFunc is a new reference */

					if (pFunc && PyCallable_Check(pFunc)) 
					{
						PyObject *pArgs = PyTuple_New(1);
						PyTuple_SetItem(pArgs, 0, PyString_FromString(this->env_name.c_str()) );
						
						
						this->pEnv = PyObject_CallObject(pFunc, pArgs);
						
						if (this->pEnv != NULL) 
						{
						    this->pFuncEnvReset = PyObject_GetAttrString(this->pEnv, "reset");
						    
						    if(pFuncEnvReset && PyCallable_Check(pFuncEnvReset))
						    {

						    	
						    	pFuncEnvRender = PyObject_GetAttrString(pEnv,"render");
						    	
						    	if(pFuncEnvRender && PyCallable_Check(pFuncEnvRender))
						    	{
						    		pFuncEnvStep = PyObject_GetAttrString(pEnv,"step");
						    		
						    		if( pFuncEnvStep && PyCallable_Check(pFuncEnvStep) )
						    		{
						    		
						    			pEnvActionSpace = PyObject_GetAttrString(pEnv,"action_space");
						    			pFuncEnvActionSpaceSample = PyObject_GetAttrString(pEnvActionSpace,"sample");
						    			if( pFuncEnvActionSpaceSample && PyCallable_Check(pFuncEnvActionSpaceSample) )
						    			{
						    			
											std::cout << "INITIALIZATION OF THE ENVIRONMENT :: DONE." << std::endl;
										}
									}							
								}
						    }		            
						    
						    Py_DECREF(pEnv);
						}
					}
					Py_XDECREF(pFunc);
					Py_DECREF(pModule);
				}
				else
				{
					std::cout << "INITIALIZATION OF THE ENVIRONMENT :: ERROR !!!!!!!!!!!!!!!!" << std::endl;
				}
				
				this->operationhandled = true;
				this->operation[initEnv] = false;
			}
			
			if( this->operation[init] )
			{
				std::cout << "RESETTING :::: " << std::endl;
				PyObject_CallObject(this->pFuncEnvReset,NULL);
				this->isDone = false;
				
				this->operationhandled = true;
				this->operation[init] = false;
			}
			
			if( this->operation[execAction] )
			{
				PyObject_CallObject(pFuncEnvRender,NULL);
				//std::cout << "rendering in progress... " << std::endl;

				PyObject *pArgActionSample = PyObject_CallObject( pFuncEnvActionSpaceSample, NULL);

				//std::cout << "OBS : ";
				//PyObject_Print(pArgActionSample,stdout,0);
				//std::cout << std::endl;
				
				//--------------------------
				
				//creation of the action :
				PyObject *actionlist = PyList_New(this->nbraction);
				for(int i=1;i<=this->nbraction;i++)
				{
					PyList_SetItem(actionlist, i-1 , PyFloat_FromDouble( this->A.get(i,1) ) );
				}
				
				//std::cout << "ACTIONLIST : ";
				//PyObject_Print(actionlist,stdout,0);
				//std::cout << std::endl;
				
				PyObject *argtupleactionlist = PyTuple_New(1);
				PyTuple_SetItem(argtupleactionlist,0,actionlist );

				PyObject *NPactionlist = PyObject_CallObject(pFuncl2n,argtupleactionlist);
				//std::cout << "NP ACTIONLIST : ";
				//PyObject_Print(NPactionlist,stdout,0);
				//std::cout << std::endl;
				
				//--------------------------
				
				PyObject *pArgAction = PyTuple_New(1);
				//PyTuple_SetItem(pArgAction, 0, pArgActionSample);
				PyTuple_SetItem(pArgAction, 0, NPactionlist);
				//std::cout << "ACTION : ";
				//PyObject_Print(pArgAction,stdout,0);
				//std::cout << std::endl;


				PyObject *pObsRewardDoneInforStepValue = PyObject_CallObject( pFuncEnvStep, pArgAction);
				//std::cout << "RETURN :: " ;
				//PyObject_Print(pObsRewardDoneInforStepValue,stdout,0);
				//std::cout << std::endl;
				PyObject *Reward = PyTuple_GetItem(pObsRewardDoneInforStepValue, 1);
				float rewardval = float( PyFloat_AsDouble(Reward) );
				this->reward.set( rewardval, 1,1);

				PyObject *obsarr = PyTuple_GetItem(pObsRewardDoneInforStepValue, 0);
				PyObject * pFuncToListOBSARR = PyObject_GetAttrString(obsarr,"tolist");
				PyObject *obs = PyObject_CallObject(pFuncToListOBSARR,NULL);
				//PyObject_Print(obs,stdout,0);
				//std::cout << std::endl;
				
				std::vector<float> obss(PyList_GET_SIZE(obs));
				for(int k=0;k<obss.size();k++)
				{
					obss[k] = PyFloat_AsDouble(PyList_GetItem(obs,k));
					this->state.set( obss[k], k,1);
				}
				
				
				PyObject *done = PyTuple_GetItem(pObsRewardDoneInforStepValue, 2);
			
				//if( Py_True == done )
				if(this->counterit >= this->nbrit)
				{
					this->isDone = true;
					this->counterit = 0;
				}
				else
				{
					this->isDone = false;
				}
				
				this->operationhandled = true;	
				this->operation[execAction] = false;
			}
			
			if( this->operation[getState] )
			{
				this->operationhandled = true;	
				this->operation[execAction] = false;
			}
			
		}	
		
		Py_Finalize();
	}
	
	private :
	
	std::string env_name;
	
	PyObject *pModule;
	PyObject *pEnv;
	PyObject *pEnvName;
	PyObject *pFuncEnvReset;
	PyObject *pFuncEnvRender;
	PyObject *pFuncEnvStep;
	PyObject *pEnvActionSpace;
	PyObject *pFuncEnvActionSpaceSample;
	
	PyObject *pl2n;
	PyObject *pFuncl2n;
	
	std::map<std::string,bool> operation;
	std::string initEnv;
	std::string init;
	std::string execAction;
	std::string getState;
	Mat<float> A;
	int nbrstate;
	int nbraction;
	
	int argc;
	char** argv;
	
	bool isDone;
	bool operationhandled;
	
	std::thread* threadLoop;
};

#endif
