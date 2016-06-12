#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "../MAT/Mat.h"
#include "../RK4/RK4.h"


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
	
	
	virtual Mat<T> getCurrentState()	const
	{
		return state;
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
	protected :
	
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
    
    public :
    
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
			float rhoi = this->SIMstate.get(4,i);
		    int ni = i+1;
		    if(ni > nbrRobots)
		    {
		        ni = 1;
			}
		    
		    float rhoni = this->SIMstate.get(4,ni);
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
		rewardCumulativePhi = 1.0f/(1.0f+rewardCumulativePhi/this->nbrRobots);
		//rewardCumulativePhi = 1.0f/(1.0f+rewardCumulativePhi);
		this->reward = Mat<float>(rewardCumulativePhi, 1,1);
		
		if(endSIMULATIONBAD)
		{
			this->reward *= 0.0f;
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
    
    /* the state of the learning algo will contain :
    	-x
    	-theta
    	-xdot
    	-thetadot
    	*/
    SimulatorRKCARTPOLE(const float& EOE)  : Environment<float>(EOE,4),SIMstate(Mat<float>(0.0f,5,1))
    {
       
       this->currentTime = 0.0f;
       this->timeStep = 0.001f;
       this->endSIMULATIONBAD = false;   
       
       this->rk = new RK4( CARTPOLEUPDATE, this);
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
        float initTheta = 0.0f;
        if(random)
        	initTheta = ( ((float)(-rand()%100))+((float)(rand()%100)) )/PI;
        
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
        this->rk->initialize();
    }
	
	
	/* action space :
		-F
	*/
	virtual Mat<float> executeAction(const Mat<float>& action)	override
	{
		this->action = action;
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
	
	virtual bool isTerminal()	override
	{
		if(endSIMULATIONBAD)
		{
			return true;
		}
		
		return (this->currentTime > this->ENDOFEPISODE ? true : false);
	}
	
};

#endif
