#ifndef RK4_H
#define RK4_H


#include "../MAT/Mat.h"
#include <vector>
#include "../RunningStats/RunningStats.h"

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
	
    public :
    
    RK4(Mat<float> (*func_)(const Mat<float>&, void*), void* objUsed_) : func(func_)
    {
        this->currentTime = 0;
        this->timeStep = 1e-2f;
        this->endTime = 1;
        
        this->objUsed = objUsed_;
		rs = new RunningStats<float>(std::string("datas"), 100 );
    }
    
    ~RK4()
    {
    	delete rs;
    }
    
    void initialize()
    {
    	this->currentTime = 0;
    	this->timeStep = 1e-2f;
    	this->endTime = 1;
    }
    
    Mat<float> solve(const Mat<float>& initState, const float& timeStep_, const float& endTime_)
    {
       this->timeStep = timeStep_;
       this->endTime = endTime_;
       this->stateVector = initState;
       //this->plotter.add(this->stateVector);
       //this->recording.push_back( this->stateVector);
       this->printState();
       
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
           
           this->recording.push_back( this->stateVector);
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
    float eps = numeric_limits<float>::epsilon();
    Mat<float> nstateDot( 0.0f*state );
	float m1 = 1.0f;
	float m2 = 0.5f;
	float l = 1.0f;
	float g = -9.81f;
	float F = state.get(5,1);
	float thetadot = state.get(4,1);
	float theta = state.get(2,1);
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
    
    nstateDot.set( state.get(3,1), 1,1);
    nstateDot.set( state.get(4,1), 2,1);
    nstateDot.set( y.get(1,1), 3,1);
    nstateDot.set( y.get(2,1), 4,1);
    nstateDot.set( 0.0f, 5,1);
    
    return nstateDot;
}


#endif


