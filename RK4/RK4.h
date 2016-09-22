#ifndef RK4_H
#define RK4_H


//#include "../MAT/Mat.h"
#include "../MAT/Mat2.h"
#include <vector>
#include "../RunningStats/RunningStats.h"


class SimulatorRKCARTPOLE;


class RK4
{
    protected :
  	
  	Mat<float> (*func)(const Mat<float>&, void*);
    void* objUsed;
    
    Mat<float> stateVector;
    
    float timeStep;
    float currentTime;
    float endTime;
    
    Mat<float> an;
    Mat<float> bn;
    Mat<float> cn;
    Mat<float> dn;
    
    std::vector<Mat<float> > recording;
	RunningStats<float>* rs;
	int countRecord;
	int idxThread;
	
    public :
    
    RK4(Mat<float> (*func_)(const Mat<float>&, void*) = NULL, void* objUsed_ = NULL) : func(func_)
    {
        this->currentTime = 0;
        this->timeStep = 1e-2f;
        this->endTime = 1;
        
        this->objUsed = objUsed_;
        
        this->countRecord = 0;
        this->idxThread = 0;
		rs = new RunningStats<float>(std::string("datas"), 100 );
    }
    
    ~RK4()
    {
    	delete rs;
    }
    
    void initialize(int idxThread_= 0, bool write = false)
    {
    	this->currentTime = 0;
    	this->timeStep = 1e-2f;
    	this->endTime = 1;

		this->countRecord++;
		this->idxThread = idxThread_;
		
		if(write)
		{
    		writeInFile(std::string("./DATAS/DATA_EPISODE_RK_")+std::to_string(this->countRecord)+std::string(".txt")+std::to_string(this->idxThread), this->recording );
    	}
    	
    	this->recording.clear();

    }
    
    Mat<float> solve(const Mat<float>& initState, const float& timeStep_, const float& endTime_)
    {
       this->timeStep = timeStep_;
       this->endTime = endTime_;
       this->stateVector = initState;
       //this->plotter.add(this->stateVector);
       this->recording.push_back( transpose(this->stateVector));
       
       //this->printState();
       
       Mat<float> velocity(1,1);
       
       while(this->currentTime < this->endTime)
       {
           this->an = this->func(this->stateVector,this->objUsed);
           this->bn = this->func(this->stateVector+(this->timeStep/2.0f)*this->an,this->objUsed);
           this->cn = this->func(this->stateVector+(this->timeStep/2.0f)*this->bn,this->objUsed);
           this->dn = this->func(this->stateVector+this->timeStep*this->cn,this->objUsed);
           
           velocity = (this->timeStep/6.0f)*(this->an+2.0f*this->bn+2.0f*this->cn+this->dn) ;

           //regularizeNanM( &velocity);
           regularizeNanInfM( &velocity);
           
           this->stateVector += velocity;
           
           this->recording.push_back( transpose(this->stateVector) );
           //this->plotter.add(this->stateVector);
           
           this->currentTime = this->currentTime + this->timeStep;
       }
       
		return this->stateVector;
       
    }
    
    std::vector<Mat<float> > getRecording()	const
    {
        return this->recording;
    }
    
    void printState()
    {
    	for(int i = 1;i<=stateVector.getLine();i++)
    	{
    		for(int j=1;j<=stateVector.getColumn();j++)
    		{
    			rs->tadd( i+100*j, stateVector.get(i,j) );
    		}
    	}
    	
    	for(int j=1;j<=3;j++)
    	{
			rs->tadd( -1-j*10, stateVector.get(1,j)*cos(stateVector.get(4,j) ) );
			rs->tadd( -2-j*10, stateVector.get(1,j)*sin(stateVector.get(4,j) ) );
		}
    }
    
    float getCurrentTime()	const
    {
    	return currentTime;
    }
    
    Mat<float>& getStateVector()
    {
    	return stateVector;
    }
    
};

class RK42 : public RK4
{
	protected :
	Mat<float> (*func2)(const Mat<float>&,Mat<float>&, void*);
	
	public :
	
	
	RK42(Mat<float> (*func2_)(const Mat<float>&,Mat<float>&, void*), void* objUsed_) : RK4(NULL,objUsed_),func2(func2_)
    {

    }
    
    ~RK42()
    {
    
    }
    
	Mat<float> solve(const Mat<float>& initState, const float& timeStep_, const float& endTime_)
    {
       this->timeStep = timeStep_;
       this->endTime = endTime_;
       this->stateVector = initState;
       //this->plotter.add(this->stateVector);
       this->recording.push_back( transpose(this->stateVector));
       
       //this->printState();
       
       Mat<float> velocity(1,1);
       
       while(this->currentTime < this->endTime)
       {
           this->stateVector = this->func2(this->stateVector,this->an,this->objUsed);
           this->func2(this->stateVector+(this->timeStep/2.0f)*this->an, this->bn,this->objUsed);
           this->func2(this->stateVector+(this->timeStep/2.0f)*this->bn, this->cn,this->objUsed);
           this->func2(this->stateVector+this->timeStep*this->cn, this->dn, this->objUsed);
           
           velocity = (this->timeStep/6.0f)*(this->an+2.0f*this->bn+2.0f*this->cn+this->dn) ;

           //regularizeNanM( &velocity);
           regularizeNanInfM( &velocity);
           
           this->stateVector += velocity;
           //this->stateVector = this->func2(this->stateVector,this->an,this->objUsed);
           
           this->recording.push_back( transpose(this->stateVector) );
           //this->plotter.add(this->stateVector);
           
           this->currentTime = this->currentTime + this->timeStep;
       }
       
		return this->stateVector;
       
    }
    
};


class RK43 : public RK4
{
	protected :
	Mat<float> (*func2)(const float&, const Mat<float>&,Mat<float>&, void*);
	void (*callback)( void*,void*);
	
	public :
	
	
	RK43(Mat<float> (*func2_)(const float&, const Mat<float>&,Mat<float>&, void*), void* objUsed_) : RK4(NULL,objUsed_),func2(func2_)
    {
		callback = NULL;
    }
    
    ~RK43()
    {
    
    }
    
	Mat<float> solve(const Mat<float>& initState, const float& timeStep_, const float& endTime_)
    {
       this->timeStep = timeStep_;
       this->endTime = endTime_;
       this->stateVector = initState;
       //this->plotter.add(this->stateVector);
       this->recording.push_back( transpose(this->stateVector));
       
       //this->printState();
       
       Mat<float> velocity(1,1);
       
       while(this->currentTime < this->endTime)
       {
           this->stateVector = this->func2(this->currentTime,this->stateVector,this->an,this->objUsed);
           this->func2(this->currentTime+this->timeStep/2.0f, this->stateVector+(this->timeStep/2.0f)*this->an, this->bn,this->objUsed);
           this->func2(this->currentTime+this->timeStep/2.0f, this->stateVector+(this->timeStep/2.0f)*this->bn, this->cn,this->objUsed);
           this->func2(this->currentTime+this->timeStep, this->stateVector+this->timeStep*this->cn, this->dn, this->objUsed);
           
           velocity = (this->timeStep/6.0f)*(this->an+2.0f*this->bn+2.0f*this->cn+this->dn) ;

           //regularizeNanM( &velocity);
           regularizeNanInfM( &velocity);
           
           this->stateVector += velocity;
           //this->stateVector = this->func2(this->stateVector,this->an,this->objUsed);
           
           this->recording.push_back( transpose(this->stateVector) );
           //this->plotter.add(this->stateVector);
           
           //callback :
           if(callback != NULL)
           {
           		this->callback(this,this->objUsed);
           }
           
           
           this->currentTime = this->currentTime + this->timeStep;
       }
       
		return this->stateVector;
       
    }
    
    void setCallbackFunction( void (*cb)(void*,void*) )
    {
    	this->callback = cb;
    }
    
};


Mat<float> SRupdate1four(const Mat<float>& state, void* swarmObj)
{
	/*% %     state ought to be partitionned the following way :
	% %     on each colums : state of a given robot of the Swarm
	% %     a state of a robot is : [r ; theta; phi]
	% %     dimensions are : 4 x nbrRobots
    */
    float eps = numeric_limits<float>::epsilon();
    int nbrRobots = (state.getColumn()-5)/2;
    Mat<float> nstateDot( 0.0f*state );
    
    float a = state.get(1,nbrRobots+1);
    float sigma = state.get(1,nbrRobots+2);
    float kw = state.get(1,nbrRobots+3);
    float kv = state.get(1,nbrRobots+4);
    float couplingStrength = state.get(1,nbrRobots+5);
    
    std::vector<float> v(nbrRobots+1);
    
	//let us compute rdot(n+1) for each i :
	//let us compute thetadot(n+1) for each i :
    for(int i=1;i<=nbrRobots;i++)
    {
        float desiredR = state.get(1,nbrRobots+i);
        float desiredRdot = state.get(2,nbrRobots+i);
        float r = state.get(1,i);
        float fr = a*r*(1-r*r/(desiredR*desiredR+eps));
        
        nstateDot.set(fr, 1,i);
        //handle the zero order holder for the velocity of the desired radius.
		nstateDot.set(desiredRdot, 1,nbrRobots+i);
        
        float theta = state.get(2,i);
        //phii = state(3,i);
        float rhoi = state.get(4,i);
        int ni = i+1;
        if(ni > nbrRobots)
        {
            ni = 1;
		}
        
        float rhoni = state.get(4,ni);
        float phii = rhoni-rhoi;
        float gphi = sigma+couplingStrength*sin(phii);
        nstateDot.set( kw*(r*gphi*cos(theta)-fr*sin(theta)) , 2,i);
        
		//let us compute v for each i:
        v[i] =  kv * ( fr*cos(theta)+r*gphi*sin(theta) ) ;
        
        //let us compute rhodot(n+1) for each i :
        float x = r*cos(rhoi);
        float y = r*sin(rhoi);
        float xdot = v[i]*cos(rhoi+theta);
        float ydot = v[i]*sin(rhoi+theta);
        nstateDot.set( ( 1 / (1+ (y/(x+eps))*(y/(x+eps)) ) ) * (ydot*x-xdot*y)/(x*x+eps) , 4,i);
    }
    
    
	//let us compute phidot(n+1) for each i :
    for(int i=1;i<=nbrRobots;i++)
    {
        int ni = i+1;
        if(ni > nbrRobots)
        {
            ni = 1;
        }
        
        float thetani = state.get(2,ni);
        float thetai = state.get(2,i);
        float rni = state.get(1,ni)+eps;
        float ri = state.get(1,i)+eps;
        
        float partni = v[ni]*cos(thetani)/rni;
        float parti = v[i]*cos(thetai)/ri;
        
        /*
        if(isnan(partni))
        {
        	//then we divided by zero...
           	partni = v[ni]*cos(thetani)/1e-10;
        }
        if(isnan(parti))
        {
		   //then we divided by zero...
           parti = v[i]*cos(thetai)/1e-10;
        }
        */
        
        nstateDot.set( partni-parti, 3,i);
    }
    
    //let us final handle the zero order older for the velocity of the parameters.
    for(int i=1;i<=5;i++)
    {
    	nstateDot.set( state.get(2,2*nbrRobots+i), 1,2*nbrRobots+i);
    }
    
    return nstateDot;
}


Mat<float> CARTPOLEUPDATE(const Mat<float>& state, void* obj)
{
	/*% %     state :
	% %     x
	% %     theta
	% %     xdot
	% %		thetadot
	% %		F
    */
    Mat<float> dupstate(state);
    float eps = numeric_limits<float>::epsilon();
    Mat<float> nstateDot( 0.0f*state );
	float m1 = 1.0f;
	float m2 = 0.1f;
	float l = 0.5f;
	float g = -9.81f;
	float F = dupstate.get(5,1);
	float thetadot = dupstate.get(4,1);
	float theta = dupstate.get(2,1);
	if( !isfinite(theta))
	{
		theta = 0.0f;
		dupstate.set( theta, 2,1);
	}
	
	while( theta > 2*PI)
	{
		theta-=2*PI;
	}
	while( theta < 0.0f)
	{
		theta += 2*PI;
	}
		
	float stheta = sin(theta);
	float ctheta = cos(theta);
	
	float a = ctheta;
	float b = l;
	float c = m1+m2;
	float d = ctheta*m2*l;
	float det = a*d-c*b;
	Mat<float> inv(2,2);
	inv.set( d, 1,1);
	inv.set( -c, 1,2);
	inv.set( -b, 2,1);
	inv.set( a, 2,2);
	inv *= (float)(1.0f/(det+eps));
	
	Mat<float> y(2,1);
	y.set( -g*stheta, 1,1);
	y.set( F+m2*l*thetadot*thetadot*stheta, 2,1);
	
	y = inv*y;	//xdotdot, thetadotdot
    
    nstateDot.set( dupstate.get(3,1), 1,1);
    nstateDot.set( dupstate.get(4,1), 2,1);
    nstateDot.set( y.get(1,1), 3,1);
    nstateDot.set( y.get(2,1), 4,1);
    nstateDot.set( 0.0f, 5,1);
    
    return nstateDot;
}



Mat<float> KURAMOTOMODEL1(const Mat<float>& state, void* obj)
{
	/*% %     state :
	% %     theta1 w1 K1
	% %     theta2 w2 K2
	% %     ...
	% %		thetaN wN kN
	% %		
    */
    int N = state.getLine();
    Mat<float> dupstate(state);
    float eps = numeric_limits<float>::epsilon();
    Mat<float> nstateDot( 0.0f*state );
	
	//let us regularize the thetas and compute psi and r:
	float psi = 0.0f;
	Mat<float> repsi(0.0f,1,2);
	
	for(int i=1;i<=N;i++)
	{
		float theta = state.get(i,1);
		
		if( !isfinite(theta))
		{
			theta = 0.0f;
		}
	
		while( theta > 2*PI)
		{
			theta-=2*PI;
		}
		while( theta < 0.0f)
		{
			theta += 2*PI;
		}
		
		dupstate.set( theta, 2,1);		
		
		psi += theta/N;
		float stheta = sin(theta);
		float ctheta = cos(theta);
		
		repsi.set( repsi.get(1,1)+ctheta/N, 1,1);
		repsi.set( repsi.get(1,2)+stheta/N, 1,2);
	}
    
    float r = sqrt( pow(repsi.get(1,1),2)+pow(repsi.get(1,2),2) );
    
    for(int i=1;i<=N;i++)
	{
		float theta = dupstate.get(i,1);
		float thetadot = state.get(i,2)+state.get(i,3)*r*sin(theta-psi);
		
		nstateDot.set( thetadot, i,1);		
	}
	
    return nstateDot;
}

Mat<float> KURAMOTOMODEL1_RK42(const Mat<float>& state, Mat<float>& stateDot, void* obj)
{
	/*% %     state :
	% %     theta1 w1 K1
	% %     theta2 w2 K2
	% %     ...
	% %		thetaN wN kN
	% %		
    */
    int N = state.getLine();
    
    Mat<float> dupstate(state);
    float eps = numeric_limits<float>::epsilon();
    Mat<float> nstateDot( 0.0f*state );
	
	//let us regularize the thetas and compute psi and r:
	float psi = 0.0f;
	Mat<float> repsi(0.0f,1,2);
	
	for(int i=1;i<=N;i++)
	{
		float theta = state.get(i,1);
		
		if( !isfinite(theta))
		{
			theta = 0.0f;
		}
	
		while( theta > PI)
		{
			theta-=2*PI;
		}
		while( theta < -PI)
		{
			theta += 2*PI;
		}
		
		dupstate.set( theta, i,1);		
		
		psi += theta/N;
		float stheta = sin(theta);
		float ctheta = cos(theta);
		
		repsi.set( repsi.get(1,1)+ctheta/N, 1,1);
		repsi.set( repsi.get(1,2)+stheta/N, 1,2);
	}
    
    float r = sqrt( pow(repsi.get(1,1),2)+pow(repsi.get(1,2),2) );
    
    for(int i=1;i<=N;i++)
	{
		float theta = dupstate.get(i,1);
		float thetadot = state.get(i,2)+state.get(i,3)*r*sin(psi-theta);
		
		nstateDot.set( thetadot, i,1);		
	}
	
	stateDot = nstateDot;
	
    return dupstate;
}


class DDEHistory
{
	protected :
	std::map<float,Mat<float> > phi;
	float discreteStep;
	float currentTime;
	float delay;
	
	public :
	
	DDEHistory( const float& delay_ = 1.0f, const float& dstep = 0.1f, std::map<float, Mat<float> >* historyPhi = NULL) : discreteStep(dstep), delay(delay_)
	{
		if( historyPhi != NULL)
		{
			phi = *historyPhi;
		}
		else
		{
			phi.clear();
		}
		
		currentTime = 0.0f;
		
	}
	
	~DDEHistory()
	{
	
	}
	
	Mat<float>& get( const float& t)
	{
		if( phi.find(t) != phi.end() )
		{
			return phi[t];
		}
		else
		{
			//we have to interpolate the quantity : LINEAR INTERPOLATION 
			float tm,tp;
			Mat<float> sm,sp;
			std::map<float,Mat<float> >::iterator it = phi.begin();
		
			while( it->first < t && it != phi.end() )
			{
				it++;
			}
			it--;
			
			tm = it->first;
			sm = it->second;
			
			while( it->first <= t && it != phi.end() )
			{
				it++;
			}
			
			tp = it->first;
			sp = it->second;
			
			phi[t] = sm+((float) ((t-tm)/pow(tp-tm,2)) ) * (sp-sm); 
			
			return phi[t];
			
		}
	}
	
	void add( const float& t, const Mat<float>& state)
	{
		this->phi[t] = state;
		
		if( t > this->currentTime)
		{
			this->currentTime = t;
		}
	}
	
	float getCurrentTime()	const
	{
		return currentTime;
	}
	
	float getDelay() const
	{
		return delay;
	}
	
};



Mat<float> KURAMOTOMODEL_TIME_DELAY_RK43(const float& currentTime, const Mat<float>& state, Mat<float>& stateDot, void* obj)
{
	/*% %     state :
	% %     theta1 w1 K1
	% %     theta2 w2 K2
	% %     ...
	% %		thetaN wN kN
	% %		
    */
    int N = state.getLine();
    
    DDEHistory* previousStates = (DDEHistory*)obj;
	
    float delay = previousStates->getDelay();
    float previousTime = currentTime-delay;
    Mat<float>& previousStateTau = previousStates->get(previousTime);
    
    Mat<float> dupstate(state);
    float eps = numeric_limits<float>::epsilon();
    Mat<float> nstateDot( 0.0f*state );
	
	//let us regularize the thetas and compute psi and r:
	float psi = 0.0f;
	Mat<float> repsi(0.0f,1,2);
	
	for(int i=1;i<=N;i++)
	{
		
		float theta = previousStateTau.get(i,1);
		
		if( !isfinite(theta))
		{
			theta = 0.0f;
		}
	
		while( theta > PI)
		{
			theta-=2*PI;
		}
		while( theta < -PI)
		{
			theta += 2*PI;
		}
		
		dupstate.set( theta, i,1);
		
		psi += theta/N;
		float stheta = sin(theta);
		float ctheta = cos(theta);
		
		repsi.set( repsi.get(1,1)+ctheta/N, 1,1);
		repsi.set( repsi.get(1,2)+stheta/N, 1,2);
	}
    
    float r = sqrt( pow(repsi.get(1,1),2)+pow(repsi.get(1,2),2) );
    
    try
    {
    	previousStates->add( previousTime, dupstate);
    }
    catch( std::exception& e)
    {
    	std::cout << e.what() << std::endl;
    }
    
    
    for(int i=1;i<=N;i++)
	{
		float theta = dupstate.get(i,1);
		float thetadot = state.get(i,2)+state.get(i,3)*r*sin(psi-theta);
		
		nstateDot.set( thetadot, i,1);		
	}
	
	stateDot = nstateDot;
	
    return dupstate;
}

void CALLBACK_KURAMOTOMODEL_TIME_DELAY_RK43( void* simulator, void* obj)
{
	RK43* sim = (RK43*)simulator;
	DDEHistory* ddeh = (DDEHistory*)obj;
	
	Mat<float> state( sim->getStateVector() );
	float currentTime = sim->getCurrentTime();
	
	try
	{
		ddeh->add( currentTime, state);
	}
	catch( std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	
}
#endif


