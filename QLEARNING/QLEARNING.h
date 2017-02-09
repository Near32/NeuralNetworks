#ifndef QLEARNING_H
#define QLEARNING_H

#include "../PA/PA.h"
#include "../FA/FA.h"
#include "../ENVIRONMENT/Environment.h"
#include "../RunningStats/RunningStats.h"

#include <mutex>
#include <thread>

//#define debuglvl1

class QLEARNING
{
	private :
	
	FA<float>* fa;				//Function Approximator from which we retrieve the Q values.
	Environment<float>* env;	//environnement from which we retrieve the rewards and the state.
	
	float gamma;				//discount factor
	unsigned int nbrepisode;
	
	std::vector<Mat<float> > totalReturn;
	
	public :
	
	QLEARNING(const float& gamma_, Environment<float>* env_, FA<float>* fa_) : gamma(gamma_), env(env_), fa(fa_), nbrepisode(100)
	{
	
	}
	
	~QLEARNING()
	{
	
	}
	
	void run(unsigned int nbrepisode = 100)
	{
		this->nbrepisode = nbrepisode;
		
		this->fa->initialize();
		
		
		for(int i=0;i<nbrepisode;i++)
		{
			this->env->initialize();	//random initial state or predefined...
		
		    unsigned int iteration = 0;
			while( !this->env->isTerminal() )
			{
		
				Mat<float> S(this->env->getCurrentState() );
				Mat<float> A(this->fa->eps_greedy(S));
			
				Mat<float> R(this->env->executeAction(A));
				Mat<float> S1(this->env->getCurrentState());
			
				this->fa->update(S,A,S1,R);
				
				
				//totalReturn :
				if(iteration == 0)
				{
					this->totalReturn.push_back(R);
				}
				else
				{
					this->totalReturn[iteration]+=R;
				}
				
				iteration++;
			}
		}
	}
	
	Mat<float> getTotalReturn(unsigned int idxEpisode)
	{
		if(idxEpisode < this->nbrepisode)
		{
			return this->totalReturn[idxEpisode];
		}
		
		return Mat<float>((float)0,1,1);
	}
};


class XP
{
	public :
	
	XP(const Mat<float>& S_,const Mat<float>& A_,const Mat<float>& S1_,const Mat<float>& R_) : S(S_),A(A_),S1(S1_),R(R_)
	{
	
	}
	
	~XP()
	{
	
	}


	//---------------------------------
	
	Mat<float> S;
	Mat<float> A;
	Mat<float> S1;
	Mat<float> R;
	
};

class XPActorCritic
{
	public :
	
	XPActorCritic(const Mat<float>& S_ = Mat<float>(1,1),const Mat<float>& A_ = Mat<float>(1,1),const Mat<float>& S1_ = Mat<float>(1,1), const Mat<float>& A1_ = Mat<float>(1,1), const Mat<float>& R_ = Mat<float>(1,1)) : S(S_),A(A_),S1(S1_), A1(A1_), R(R_)
	{
	
	}
	
	~XPActorCritic()
	{
	
	}


	//---------------------------------
	
	Mat<float> S;
	Mat<float> A;
	Mat<float> S1;
	Mat<float> A1;
	Mat<float> R;
	
};

class QLEARNINGXPReplay
{
	private :
	
	FA<float>* fa;				//Function Approximator from which we retrieve the Q values.
	Environment<float>* env;	//environnement from which we retrieve the rewards and the state.
	
	float gamma;				//discount factor
	unsigned int nbrepisode;
	
	std::vector<Mat<float> > totalReturn;
	
	unsigned int nbrReplay;
	std::vector<XP> bankXP;
	int nbrEpisodeBankXPHolder;
	
	public :
	
	QLEARNINGXPReplay(const unsigned int& nbrepi, const float& gamma_, Environment<float>* env_, FA<float>* fa_) : gamma(gamma_), env(env_), fa(fa_), nbrepisode(nbrepi),nbrReplay(100), nbrEpisodeBankXPHolder(20)
	{
		//TODO : evaluate the number of REPLAY NEEDED.
		//TODO : evaluate the number of EPISODE NEEDED.
	}
	
	~QLEARNINGXPReplay()
	{
		
	}
	
	void run(unsigned int nbrepisode = 100)
	{
		this->nbrepisode = nbrepisode;
		
		this->fa->initialize();
		
		int batchSize = 10;
		
		for(int i=0;i<nbrepisode;i++)
		{
			this->env->initialize();	//random initial state or predefined...
			
		    unsigned int iteration = 0;
			while( !this->env->isTerminal() )
			{
				clock_t time = clock();
		
				Mat<float> S(this->env->getCurrentState() );
				Mat<float> A(this->fa->eps_greedy(S));
				Mat<float> Qsa( this->fa->getQvalue(S,A) );
#ifdef debuglvl1
				A *= 0.0f;
#endif			
				Mat<float> R(this->env->executeAction(A));
				Mat<float> S1(this->env->getCurrentState());
			
				//METHOD 1 :
				/*
				this->fa->update(S,A,S1,R);
				*/
				
				//METHOD 2 :
				Mat<float> A1( this->fa->greedy(S1) ); 
				Mat<float> Qs1a1(  this->fa->getQvalue(S1,A1) );
				this->fa->updateDelta(operatorC(S,A), R+gamma*Qs1a1-Qsa );
				
				//totalReturn :
				if(iteration == 0)
				{
					this->totalReturn.push_back(R);
				}
				else
				{
					this->totalReturn[i]+=R;
				}
				
				
				iteration++;
				
				//---------------------------------------
				//---------------------------------------
				//---------------------------------------
				//XP REPLAY :
				//add the new exp :
				this->bankXP.push_back( XP(S,A,S1,R) );
				//let us replay them :
				for(int i=(nbrReplay<bankXP.size()?nbrReplay:bankXP.size());i--;)
				{
					int idx = rand()%bankXP.size();
					
					//this->fa->update( bankXP[idx].S, bankXP[idx].A, bankXP[idx].S1, bankXP[idx].R );
					//METHOD 1:
					/*
					this->fa->updateBATCH( bankXP[idx].S, bankXP[idx].A, bankXP[idx].S1, bankXP[idx].R, batchSize );
					*/
					
					//METHOD 2 :
					Mat<float> bankQsa( this->fa->getQvalue(bankXP[idx].S, bankXP[idx].A) );
					Mat<float> bankA1( this->fa->greedy(bankXP[idx].S1) ); 
					Mat<float> bankQs1a1(  this->fa->getQvalue(bankXP[idx].S1,bankA1) );
					this->fa->updateDelta(operatorC(bankXP[idx].S,bankXP[idx].A), bankXP[idx].R+gamma*bankQs1a1-bankQsa );
				}
				//---------------------------------------
				//---------------------------------------
				//---------------------------------------
				
				std::cout << " Iteration : " << iteration << " : " << (float)(clock()-time)/CLOCKS_PER_SEC << " seconds." << " Rt = " << R.get(1,1) << " ; Q(s,a) = " << Qsa.get(1,1) << std::endl;
				transpose(A).afficher();
			}
			
			std::cout << "EPISODE : " << i << " REWARD : " << this->totalReturn[this->totalReturn.size()-1].get(1,1) << std::endl;
			
			if( i%nbrEpisodeBankXPHolder)
			{
				for(int i=iteration;i--;)	this->bankXP.erase(this->bankXP.begin());
			}
			
			writeInFile(std::string("./totalreturn.txt"), totalReturn);
		}
		
		for(int i=0;i<this->totalReturn.size();i++)
		{
			std::cout << "EPISODE : " << i << " REWARD : " << this->totalReturn[i].get(1,1) << std::endl;
		}
	}
	
	Mat<float> getTotalReturn(unsigned int idxEpisode)
	{
		if(idxEpisode < this->nbrepisode)
		{
			return this->totalReturn[idxEpisode];
		}
		
		return Mat<float>((float)0,1,1);
	}
};




class QLEARNINGXPReplayActorCriticFixedTarget
{
	private :
	
	PA<float>* pa;				//ACTOR		//Function Approximator from which we retrieve the policy.
	FA<float>* fa;				//CRITIC	//Function Approximator from which we retrieve the Q values.
	PA<float>* targetPA;
	FA<float>* targetFA;
	Environment<float>* env;	//environnement from which we retrieve the rewards and the state.
	
	float gamma;				//discount factor
	unsigned int nbrepisode;
	
	std::vector<Mat<float> > totalReturn;
	
	unsigned int nbrReplay;
	std::vector<XPActorCritic> bankXP;
	
	XPActorCritic meanXP;
	XPActorCritic stdXP;
	bool batchNormInit;
	
	int nbrEpisodeBankXPHolder;
	
	float momentumUpdate;
	
	public :
	
	QLEARNINGXPReplayActorCriticFixedTarget(const unsigned int& nbrepi, const float& gamma_, Environment<float>* env_, FA<float>* fa_, PA<float>* pa_, const float& momentumUpdate_) : gamma(gamma_), env(env_), fa(fa_), pa(pa_), nbrepisode(nbrepi),nbrReplay(20), nbrEpisodeBankXPHolder(1000), momentumUpdate(momentumUpdate_), batchNormInit(false)
	{
		//TODO : evaluate the number of REPLAY NEEDED.
		targetFA = (FA<float>*) new QFANN<float>( fa_->getLR(), fa_->getEPS(), gamma_, ((QFANN<float>*)fa_)->dimActionSpace, fa_->getNetPointer()->topology, ((QFANN<float>*)fa)->filepathRSNN+std::string(".FAtarget"));
		//*(((QFANN<float>*)targetFA)->getNetPointer()) = *(((QFANN<float>*)fa)->getNetPointer());
		targetFA->updateToward( fa, 1.0f);
		
		targetPA = (PA<float>*) new QPANN<float>( pa_->getLR(), pa_->getEPS(), gamma_, ((QPANN<float>*)pa_)->dimActionSpace, pa_->getNetPointer()->topology, ((QPANN<float>*)pa)->filepathRSNN+std::string(".PAtarget") );
		//*(((QPANN<float>*)targetPA)->getNetPointer()) = *(((QPANN<float>*)pa)->getNetPointer());
		targetPA->updateToward( pa, 1.0f);
	}
	
	~QLEARNINGXPReplayActorCriticFixedTarget()
	{
		delete targetFA;
		delete targetPA;
	}
	
	void run(unsigned int nbrepisode = 100)
	{
		this->nbrepisode = nbrepisode;
		
		this->fa->initialize();
		
		std::vector<Mat<float> > QerrorBATCH;
		std::vector<Mat<float> > testTOTALRETURN;
		int counterTest = 0;
		int valTest = 100;
		
		int bankSizeNeeded = 1000;
		
		//int batchSize = 100;
		int batchSize = nbrReplay;
		
		for(int i=0;i<nbrepisode;i++)
		{
			this->env->initialize(false);	//random initial state or predefined...
			
		    unsigned int iteration = 0;
			while( !this->env->isTerminal() )
			{
				clock_t time = clock();
		
				Mat<float> S(this->env->getCurrentState() );
				//Mat<float> A(this->pa->estimateAction(S));
				Mat<float> A(this->pa->eps_greedy(S));
				Mat<float> Qsa( this->fa->getQvalue(S,A) );


				Mat<float> R(this->env->executeAction(A));
				Mat<float> S1(this->env->getCurrentState());
			
				//ACTOR CRITIC SCHEME :
				//METHOD 1 :
				/*
				Mat<float> A1( this->pa->estimateAction(S1) ); 
				Mat<float> Qs1a1(  this->fa->getQvalue(S1,A1) );
				this->fa->update(operatorC(S,A), R+gamma*Qs1a1 );
				*/
				
				//METHOD 2 : DDPG..
				
				Mat<float> A1( this->pa->estimateAction(S1) ); 
				Mat<float> Qs1a1(  this->fa->getQvalue(S1,A1) );
				Mat<float> error(R+gamma*Qs1a1-Qsa);
				//this->fa->updateDelta(operatorC(S,A), (-1.0f)*error );
				
				/*
				for(int kk=1000;kk--;)
				{
				Mat<float> A1( this->pa->estimateAction(S1) ); 
				Mat<float> Qs1a1(  this->fa->getQvalue(S1,A1) );
				Mat<float> error(R+gamma*Qs1a1-Qsa);
				this->fa->updateDelta(operatorC(S,A), error );
				
				std::cout << " iteration : " << kk << " ; ERROR NORM : " << norme2(error) << std::endl;
				
				}
				
				//throw;
				*/
				
				
				//POLICY GRADIENT : METHOD 1 :
				/*
				this->fa->getQvalue(S,A); 
				Mat<float> dQda( this->fa->getNetPointer()->getGradientWRTinput() );
				Mat<float> satoa( 0.0f, dQda.getColumn(), A.getColumn() );
				for(int k=1;k<=A.getColumn();k++)	satoa.set( 1.0f, S.getLine()+k, k);
				dQda = dQda * satoa;
				
				Mat<float> outputPA( this->pa->estimateAction(S) );
				//this element has to be cancelled from the computed term within the backpropagation function...
				this->pa->update(S,dQda+outputPA);
				//this->pa->update(S,dQda);
				*/
				
				//POLICY GRADIENT : METHOD 2: Stochastic gradient 
				/*
				Mat<float> delta( R+gamma*Qs1a1 - Qsa);				
				Mat<float> outputPA( this->pa->estimateAction(S) );
				Mat<float> dlogpa( inverseM(outputPA) );
				//this->pa->updateDelta(S, dlogpa*Qsa);
				*/
				
				//POLICY GRADIENT : METHOD 3 : DDPG : fixed target : idem for the policy...
				
				this->fa->getQvalue(S,A); 
				Mat<float> dQda( this->fa->getNetPointer()->getGradientWRTinput() );
				Mat<float> satoa( 0.0f, dQda.getColumn(), A.getColumn() );
				for(int k=1;k<=A.getColumn();k++)	satoa.set( 1.0f, S.getLine()+k, k);
				dQda = dQda * satoa;
				//this->pa->updateDelta(S, (1.0f/nbrReplay)*dQda);
				
				/*
				for(int kk=1000;kk--;)
				{
				this->fa->getQvalue(S,A); 
				Mat<float> dQda( this->fa->getNetPointer()->getGradientWRTinput() );
				Mat<float> satoa( 0.0f, dQda.getColumn(), A.getColumn() );
				for(int k=1;k<=A.getColumn();k++)	satoa.set( 1.0f, S.getLine()+k, k);
				dQda = dQda * satoa;
				Mat<float> error((1.0f/nbrReplay)*dQda);
				this->pa->updateDelta(S, error);
				
				std::cout << " iteration : " << kk << " ; ERROR NORM : " << norme2(error) << std::endl;
				
				}
				
				throw;
				*/
				
				
				
				//POLICY GRADIENT : METHOD 4: TD 0 ACtor Critic : stochastic gradient
				/*
				//TODO : this is not a probability...
				Mat<float> delta( R+gamma*Qs1a1 - Qsa);				
				Mat<float> dlogpa( inverseM(A) );
				//this->pa->updateDelta(S, (-1.0f)*dlogpa*delta);
				*/
				
				//VALIDATION OF THE ASCENT BY USING A MINUS DLOGDA * DELTA...
				/*
				Mat<float> Aprevious(A);
				Mat<float> Rprevious(R);
				for(int kk=1000;kk--;)
				{
					Mat<float> At(this->pa->estimateAction(S));
					Mat<float> Qsat( this->fa->getQvalue(S,At) );
					Mat<float> Rt(this->env->executeAction(At));
					Mat<float> S1t(this->env->getCurrentState());
					Mat<float> A1t( this->pa->estimateAction(S1t) ); 
					Mat<float> Qs1a1t(  this->fa->getQvalue(S1t,A1t) );
					Mat<float> delta( Rt+gamma*Qs1a1t - Qsat);				
					//Mat<float> delta( Rt);				
					Mat<float> dlogpa( inverseM(At) );
					Mat<float> error( (-1.0f)*(dlogpa*delta));
					this->pa->updateDelta(S, error);
				
					std::cout << " iteration : " << kk << " ; ERROR NORM : " << norme2(error) << " dif A1 - A  " << (At-Aprevious).get(1,1) << " REWARD : " << Rt.get(1,1) << " diff : " << (Rt-Rprevious).get(1,1) << "theta = " << S1t.get(2,1)*180.0f/PI <<  std::endl;
					
					Rprevious = Rt;
					Aprevious = At;
				}
				
				throw;
				*/
				
				//totalReturn :
				if(iteration == 0)
				{
					this->totalReturn.push_back(R);
				}
				else
				{
					this->totalReturn[i]+=R;
				}
				
				
				iteration++;
				
				//---------------------------------------
				//---------------------------------------
				//---------------------------------------
				//XP REPLAY :
				//add the new exp :
				this->bankXP.push_back( XPActorCritic(S,A,S1,A1,R) );
				//let us replay them :
				if(bankSizeNeeded < bankXP.size() )
				{
					this->batchNormalization();
					Mat<float> QerrorBATCHtemp(0.0f*R);
					
					for(int i=(nbrReplay<bankXP.size()?nbrReplay:bankXP.size());i--;)
					{
						int idx = rand()%bankXP.size();
					
						//ACTOR CRITIC SCHEME :
						//METHOD 1:
						/*
						Mat<float> bankAfromS1( this->pa->estimateAction( bankXP[idx].S1 ) ); 
						this->fa->update(operatorC( bankXP[idx].S, bankXP[idx].A), R+gamma*this->fa->getQvalue( bankXP[idx].S1 , bankAfromS1) );
						*/
					
						//METHOD 2 :
						/*
						Mat<float> bankAfromS1( this->pa->estimateAction(S1) ); 
						Mat<float> bankQs1a1(  this->fa->getQvalue(bankXP[idx].S1,bankAfromS1) );
						Mat<float> bankQsa( this->fa->getQvalue(bankXP[idx].S,bankXP[idx].A) );
						this->fa->updateDelta(operatorC(bankXP[idx].S,bankXP[idx].A), bankXP[idx].R+gamma*bankQs1a1-bankQsa );
						*/
					
						//METHOD DDPG :
						/*
						Mat<float> bankAfromS1( this->pa->estimateAction(S1) ); 
						Mat<float> bankQs1a1(  this->fa->getQvalue(bankXP[idx].S1, bankAfromS1) );
						Mat<float> bankQsa( this->fa->getQvalue(bankXP[idx].S,bankXP[idx].A) );
						this->fa->updateDelta(operatorC(bankXP[idx].S,bankXP[idx].A), bankXP[idx].R+gamma*bankQs1a1-bankQsa );
						*/
					
						//METHOD DDPG or DSPG: fixed target :
						Mat<float> bankAfromS1TARGET( this->targetPA->estimateAction(S1) ); 
						Mat<float> bankQs1a1TARGET(  this->targetFA->getQvalue(bankXP[idx].S1, bankAfromS1TARGET) );
						Mat<float> bankQsa( this->fa->getQvalue(bankXP[idx].S,bankXP[idx].A) );
						Mat<float> bankerror(bankXP[idx].R+gamma*bankQs1a1TARGET-bankQsa);
						this->fa->updateDeltaBATCH(operatorC(bankXP[idx].S,bankXP[idx].A), (-1.0f)*bankerror, batchSize );
					
						QerrorBATCHtemp += bankerror;
					
						//POLICY GRADIENT : METHOD 1 :
						/*
						this->fa->getQvalue( bankXP[idx].S, bankXP[idx].A); 
						Mat<float> bankdQda( this->fa->getNetPointer()->getGradientWRTinput() );
						Mat<float> satoa( 0.0f, bankdQda.getColumn(), A.getColumn() );
						for(int k=1;k<=A.getColumn();k++)	satoa.set( 1.0f, S.getLine()+k, k);
						bankdQda = bankdQda * satoa;
				
						Mat<float> bankoutputPA( this->pa->estimateAction( bankXP[idx].S) );
						//this element has to be cancelled from the computed term within the backpropagation function...
						this->pa->update( bankXP[idx].S,bankdQda+bankoutputPA);
						//this->pa->update( bankXP[idx].S,bankdQda);
						*/
					
						//POLICY GRADIENT : METHOD 2 :
						/*
						Mat<float> bankdelta( bankXP[idx].R+gamma*bankQs1a1 - bankQsa);				
						Mat<float> bankoutputPA( this->pa->estimateAction( bankXP[idx].S) );
						Mat<float> bankdlogpa( inverseM( bankoutputPA) );
						this->pa->updateDelta(bankXP[idx].S, bankdlogpa*bankQsa);
						*/
					
						//POLICY GRADIENT : METHOD 3:
						/*
						this->fa->getQvalue(bankXP[idx].S,bankXP[idx].A); 
						Mat<float> bankdQda( this->fa->getNetPointer()->getGradientWRTinput() );
						Mat<float> banksatoa( 0.0f, bankdQda.getColumn(), A.getColumn() );
						for(int k=1;k<=A.getColumn();k++)	banksatoa.set( 1.0f, S.getLine()+k, k);
						bankdQda = bankdQda * banksatoa;
						this->pa->updateDelta(bankXP[idx].S, bankdQda);
						*/
					
						//POLICY GRADIENT : METHOD 4: TD 0 ACtor Critic
						/*
						Mat<float> bankdelta( bankXP[idx].R+gamma*bankQs1a1 - bankQsa);				
						Mat<float> bankdlogpa( inverseM(bankXP[idx].A) );
						this->pa->updateDelta(bankXP[idx].S, bankdlogpa*bankdelta);
						*/
					
						//POLICY GRADIENT : METHOD DDPG:
						/*
						this->fa->getQvalue(bankXP[idx].S, this->pa->estimateAction(bankXP[idx].S) ); 
						Mat<float> bankdQda( this->fa->getNetPointer()->getGradientWRTinput() );
						Mat<float> banksatoa( 0.0f, bankdQda.getColumn(), A.getColumn() );
						for(int k=1;k<=A.getColumn();k++)	banksatoa.set( 1.0f, S.getLine()+k, k);
						bankdQda = bankdQda * banksatoa;
						this->pa->updateDelta(bankXP[idx].S, bankdQda);
						*/
					
						//POLICY GRADIENT : METHOD DDPG : fixed target : idem for the policy...
						
						this->fa->getQvalue(bankXP[idx].S, this->pa->estimateAction(bankXP[idx].S) ); 
						Mat<float> bankdQda( this->fa->getNetPointer()->getGradientWRTinput() );
						Mat<float> banksatoa( 0.0f, bankdQda.getColumn(), A.getColumn() );
						for(int k=1;k<=A.getColumn();k++)	banksatoa.set( 1.0f, S.getLine()+k, k);
						bankdQda = bankdQda * banksatoa;
						//if(i==0)	std::cout << " DELTA Q da : " << bankdQda.get(1,1) << std::endl;
						//this->pa->updateDeltaBATCH(bankXP[idx].S, (-1.0f/nbrReplay) * bankdQda, batchSize);
						this->pa->updateDeltaBATCH(bankXP[idx].S, (1.0f/nbrReplay) * bankdQda, batchSize);
						
					
						//POLICY GRADIENT : METHOD 6: TD 0 ACtor Critic : DSPG+fixed target
						/*
						Mat<float> bankdelta( bankXP[idx].R+gamma*bankQs1a1TARGET - bankQsa);				
						Mat<float> bankdlogpa( inverseM(bankXP[idx].A) );
						//TODO : this is not stochastic : not a probability...
						this->pa->updateDeltaBATCH(bankXP[idx].S, (-1.0f/nbrReplay)*(bankdlogpa*bankdelta), batchSize);
						//this->pa->updateDeltaBATCH(bankXP[idx].S, (1.0f/nbrReplay)*(bankdlogpa*bankdelta), batchSize);				
						*/
							
					
					
					
					
					}
					
					QerrorBATCH.push_back(QerrorBATCHtemp);
					writeInFile(std::string("./FAerrorBATCH.txt"), QerrorBATCH);
					
					
					//TEST :
					if(counterTest == valTest)
					{
						counterTest = 0;
						Mat<float> tempTR(0.0f*R);
						this->env->initialize(false);
						while( !this->env->isTerminal() )
						{
	
							Mat<float> St(this->env->getCurrentState() );
							Mat<float> At(this->pa->estimateAction(St));
							Mat<float> Qsat( this->fa->getQvalue(St,At) );

							Mat<float> Rt(this->env->executeAction(At));
							Mat<float> S1t(this->env->getCurrentState());
							
							Mat<float> A1t(this->pa->estimateAction(S1t));
							//XP REPLAY :
							//add the new exp :
							this->bankXP.push_back( XPActorCritic(St,At,S1t,A1t,Rt) );
							
							tempTR += Rt;
						}
						
						testTOTALRETURN.push_back( tempTR);
						
						writeInFile(std::string("./TEST_TotalReturn.txt"), testTOTALRETURN);
					}
					else
					{
						counterTest++;
					}
					
				}
				else
				{
					std::cout << " NO XP REPLAY YET : " << bankXP.size() << std::endl;
				}
				//---------------------------------------
				//---------------------------------------
				//---------------------------------------
				
				std::cout << " Iteration : " << iteration << " : " << (float)(clock()-time)/CLOCKS_PER_SEC << " seconds." << " Rt = " << R.get(1,1) << " ; Q(s,a) = " << Qsa.get(1,1) << " ; Action performed : " << std::endl;
				transpose(A).afficher();
			}
			
			//UPDATE THE TARGET NETWORKS :
			this->updateTARGET();
			
			std::cout << "EPISODE : " << i << " REWARD : " << this->totalReturn[this->totalReturn.size()-1].get(1,1) << std::endl;
			
			if( bankXP.size() > 2*nbrEpisodeBankXPHolder)
			{
				for(int i=nbrEpisodeBankXPHolder;i--;)	this->bankXP.erase(this->bankXP.begin());
				this->batchNormInit = false;
			}
			
			writeInFile(std::string("./totalreturn.txt"), totalReturn);
		}
		
		for(int i=0;i<this->totalReturn.size();i++)
		{
			std::cout << "EPISODE : " << i << " REWARD : " << this->totalReturn[i].get(1,1) << std::endl;
		}
	}
	
	Mat<float> getTotalReturn(unsigned int idxEpisode)
	{
		if(idxEpisode < this->nbrepisode)
		{
			return this->totalReturn[idxEpisode];
		}
		
		return Mat<float>((float)0,1,1);
	}
	
	void updateTARGET()
	{
		bool retfa = this->targetFA->updateToward( this->fa, momentumUpdate);
		bool retpa = this->targetPA->updateToward( this->pa, momentumUpdate);
		
		std::cout << " UPDATE THE TARGETS : momentum : " << momentumUpdate << " ; results fa pa : " << retfa << " " << retpa << std::endl;
	}
	
	void batchNormalization()
	{
		if(!this->batchNormInit)
		{
			this->batchNormInit = true;
			int nbrxp = bankXP.size();
			float inbrxp = 1.0f/((float)(nbrxp)+numeric_limits<float>::epsilon());
			Mat<float> Sm( 0.0f*bankXP[0].S);
			Mat<float> Am( 0.0f*bankXP[0].A);
		
		
			for(int i=nbrxp;i--;)
			{
				Sm += inbrxp*bankXP[i].S;
				Am += inbrxp*bankXP[i].A;
			}
		
			float inobias = 1.0f/((float)(nbrxp-1)+numeric_limits<float>::epsilon());
			Mat<float> Sv( 0.0f*Sm);
			Mat<float> Av( 0.0f*Am);
		
		
			for(int i=nbrxp;i--;)
			{
				Sv += inobias * ( Sm-bankXP[i].S) % ( Sm-bankXP[i].S) ;
				Av += inobias * ( Am-bankXP[i].A) % ( Am-bankXP[i].A) ;			
			}
		
			meanXP = XPActorCritic( Sm, Am);
			stdXP = XPActorCritic( sqrt(Sv), sqrt(Av));
		
			
			this->pa->setInputNormalization(meanXP.S,stdXP.S);
			this->targetPA->setInputNormalization(meanXP.S,stdXP.S);
			this->fa->setInputNormalization( operatorC(meanXP.S,meanXP.A) , operatorC(stdXP.S, stdXP.A) );
			this->targetFA->setInputNormalization( operatorC(meanXP.S,meanXP.A) , operatorC(stdXP.S, stdXP.A) );
			
			/*
			meanXP.S.afficher();
			meanXP.A.afficher();
			stdXP.S.afficher();
			stdXP.A.afficher();
			std::cout << inbrxp << " " << inobias << std::endl;
			throw;
			*/
		}
		
		
	}
};




class QLEARNINGXPReplayActorCritic
{
	private :
	
	PA<float>* pa;				//ACTOR		//Function Approximator from which we retrieve the policy.
	FA<float>* fa;				//CRITIC	//Function Approximator from which we retrieve the Q values.
	Environment<float>* env;	//environnement from which we retrieve the rewards and the state.
	
	float gamma;				//discount factor
	unsigned int nbrepisode;
	
	std::vector<Mat<float> > totalReturn;
	
	unsigned int nbrReplay;
	std::vector<XPActorCritic> bankXP;
	int nbrEpisodeBankXPHolder;
	
	public :
	
	QLEARNINGXPReplayActorCritic(const unsigned int& nbrepi, const float& gamma_, Environment<float>* env_, FA<float>* fa_, PA<float>* pa_) : gamma(gamma_), env(env_), fa(fa_), pa(pa_), nbrepisode(nbrepi),nbrReplay(100), nbrEpisodeBankXPHolder(100)
	{
		//TODO : evaluate the number of REPLAY NEEDED.
	}
	
	~QLEARNINGXPReplayActorCritic()
	{
		
	}
	
	void run(unsigned int nbrepisode = 100)
	{
		this->nbrepisode = nbrepisode;
		
		this->fa->initialize();
		
		int batchSize = 100;
		
		for(int i=0;i<nbrepisode;i++)
		{
			this->env->initialize();	//random initial state or predefined...
			
		    unsigned int iteration = 0;
			while( !this->env->isTerminal() )
			{
				clock_t time = clock();
		
				Mat<float> S(this->env->getCurrentState() );
				//Mat<float> A(this->pa->estimateAction(S));
				Mat<float> A(this->pa->eps_greedy(S));
				Mat<float> Qsa( this->fa->getQvalue(S,A) );
#ifdef debuglvl1
				A *= 0.0f;
#endif			
				Mat<float> R(this->env->executeAction(A));
				Mat<float> S1(this->env->getCurrentState());
			
				//ACTOR CRITIC SCHEME :
				//METHOD 1 :
				/*
				Mat<float> A1( this->pa->estimateAction(S1) ); 
				Mat<float> Qs1a1(  this->fa->getQvalue(S1,A1) );
				this->fa->update(operatorC(S,A), R+gamma*Qs1a1 );
				*/
				
				//METHOD 2 :
				Mat<float> A1( this->pa->estimateAction(S1) ); 
				Mat<float> Qs1a1(  this->fa->getQvalue(S1,A1) );
				//this->fa->updateDelta(operatorC(S,A), (R+gamma*Qs1a1-Qsa) );
				
				//POLICY GRADIENT : METHOD 1 :
				/*
				this->fa->getQvalue(S,A); 
				Mat<float> dQda( this->fa->getNetPointer()->getGradientWRTinput() );
				Mat<float> satoa( 0.0f, dQda.getColumn(), A.getColumn() );
				for(int k=1;k<=A.getColumn();k++)	satoa.set( 1.0f, S.getLine()+k, k);
				dQda = dQda * satoa;
				
				Mat<float> outputPA( this->pa->estimateAction(S) );
				//this element has to be cancelled from the computed term within the backpropagation function...
				this->pa->update(S,dQda+outputPA);
				//this->pa->update(S,dQda);
				*/
				
				//POLICY GRADIENT : METHOD 2:
				/*
				Mat<float> delta( R+gamma*Qs1a1 - Qsa);				
				Mat<float> outputPA( this->pa->estimateAction(S) );
				Mat<float> dlogpa( inverseM(outputPA) );
				this->pa->updateDelta(S, dlogpa*Qsa);
				*/
				
				//POLICY GRADIENT : METHOD 3:
				/*
				this->fa->getQvalue(S,A); 
				Mat<float> dQda( this->fa->getNetPointer()->getGradientWRTinput() );
				Mat<float> satoa( 0.0f, dQda.getColumn(), A.getColumn() );
				for(int k=1;k<=A.getColumn();k++)	satoa.set( 1.0f, S.getLine()+k, k);
				dQda = dQda * satoa;
				this->pa->updateDelta(S, 1e5f*dQda);
				*/
				
				//POLICY GRADIENT : METHOD 4: TD 0 ACtor Critic
				Mat<float> delta( R+gamma*Qs1a1 - Qsa);				
				Mat<float> dlogpa( inverseM(A) );
				//this->pa->updateDelta(S, dlogpa*delta);
				
				
				//totalReturn :
				if(iteration == 0)
				{
					this->totalReturn.push_back(R);
				}
				else
				{
					this->totalReturn[i]+=R;
				}
				
				
				iteration++;
				
				//---------------------------------------
				//---------------------------------------
				//---------------------------------------
				//XP REPLAY :
				//add the new exp :
				this->bankXP.push_back( XPActorCritic(S,A,S1,A1,R) );
				//let us replay them :
				if(10*nbrReplay < bankXP.size() )
				{
				for(int j=(nbrReplay<bankXP.size()?nbrReplay:bankXP.size());j--;)
				{
					int idx = rand()%bankXP.size();
					
					//ACTOR CRITIC SCHEME :
					//METHOD 1:
					/*
					Mat<float> bankAfromS1( this->pa->estimateAction( bankXP[idx].S1 ) ); 
					this->fa->update(operatorC( bankXP[idx].S, bankXP[idx].A), R+gamma*this->fa->getQvalue( bankXP[idx].S1 , bankAfromS1) );
					*/
					
					//METHOD 2 :
					/*
					Mat<float> bankAfromS1( this->pa->estimateAction(S1) ); 
					Mat<float> bankQs1a1(  this->fa->getQvalue(bankXP[idx].S1,bankAfromS1) );
					Mat<float> bankQsa( this->fa->getQvalue(bankXP[idx].S,bankXP[idx].A) );
					this->fa->updateDelta(operatorC(bankXP[idx].S,bankXP[idx].A), bankXP[idx].R+gamma*bankQs1a1-bankQsa );
					*/
					
					//METHOD DDPG :
					Mat<float> bankAfromS1( this->pa->estimateAction(S1) ); 
					Mat<float> bankQs1a1(  this->fa->getQvalue(bankXP[idx].S1, bankAfromS1) );
					Mat<float> bankQsa( this->fa->getQvalue(bankXP[idx].S,bankXP[idx].A) );
					this->fa->updateDelta(operatorC(bankXP[idx].S,bankXP[idx].A), bankXP[idx].R+gamma*bankQs1a1-bankQsa );
					
					//POLICY GRADIENT : METHOD 1 :
					/*
					this->fa->getQvalue( bankXP[idx].S, bankXP[idx].A); 
					Mat<float> bankdQda( this->fa->getNetPointer()->getGradientWRTinput() );
					Mat<float> satoa( 0.0f, bankdQda.getColumn(), A.getColumn() );
					for(int k=1;k<=A.getColumn();k++)	satoa.set( 1.0f, S.getLine()+k, k);
					bankdQda = bankdQda * satoa;
				
					Mat<float> bankoutputPA( this->pa->estimateAction( bankXP[idx].S) );
					//this element has to be cancelled from the computed term within the backpropagation function...
					this->pa->update( bankXP[idx].S,bankdQda+bankoutputPA);
					//this->pa->update( bankXP[idx].S,bankdQda);
					*/
					
					//POLICY GRADIENT : METHOD 2 :
					/*
					Mat<float> bankdelta( bankXP[idx].R+gamma*bankQs1a1 - bankQsa);				
					Mat<float> bankoutputPA( this->pa->estimateAction( bankXP[idx].S) );
					Mat<float> bankdlogpa( inverseM( bankoutputPA) );
					this->pa->updateDelta(bankXP[idx].S, bankdlogpa*bankQsa);
					*/
					
					//POLICY GRADIENT : METHOD 3:
					/*
					this->fa->getQvalue(bankXP[idx].S,bankXP[idx].A); 
					Mat<float> bankdQda( this->fa->getNetPointer()->getGradientWRTinput() );
					Mat<float> banksatoa( 0.0f, bankdQda.getColumn(), A.getColumn() );
					for(int k=1;k<=A.getColumn();k++)	banksatoa.set( 1.0f, S.getLine()+k, k);
					bankdQda = bankdQda * banksatoa;
					this->pa->updateDelta(bankXP[idx].S, bankdQda);
					*/
					
					//POLICY GRADIENT : METHOD 4: TD 0 ACtor Critic
					/*
					Mat<float> bankdelta( bankXP[idx].R+gamma*bankQs1a1 - bankQsa);				
					Mat<float> bankdlogpa( inverseM(bankXP[idx].A) );
					this->pa->updateDelta(bankXP[idx].S, bankdlogpa*bankdelta);
					*/
					
					//POLICY GRADIENT : METHOD DDPG:
					this->fa->getQvalue(bankXP[idx].S, this->pa->estimateAction(bankXP[idx].S) ); 
					Mat<float> bankdQda( this->fa->getNetPointer()->getGradientWRTinput() );
					Mat<float> banksatoa( 0.0f, bankdQda.getColumn(), A.getColumn() );
					for(int k=1;k<=A.getColumn();k++)	banksatoa.set( 1.0f, S.getLine()+k, k);
					bankdQda = bankdQda * banksatoa;
					this->pa->updateDelta(bankXP[idx].S, bankdQda);
					
				}
				}
				else
				{
					std::cout << " NO XP REPLAY YET : " << bankXP.size() << std::endl;
				}
				//---------------------------------------
				//---------------------------------------
				//---------------------------------------
				
				std::cout << " Iteration : " << iteration << " : " << (float)(clock()-time)/CLOCKS_PER_SEC << " seconds." << " Rt = " << R.get(1,1) << " ; Q(s,a) = " << Qsa.get(1,1) << " ; Action performed : " << std::endl;
				transpose(A).afficher();
			}
			
			std::cout << "EPISODE : " << i << " REWARD : " << this->totalReturn[this->totalReturn.size()-1].get(1,1) << std::endl;
			
			if( i%nbrEpisodeBankXPHolder)
			{
				for(int j=iteration;j--;)	this->bankXP.erase(this->bankXP.begin());
			}
			
			writeInFile(std::string("./totalreturn.txt"), totalReturn);
		}
		
		for(int i=0;i<this->totalReturn.size();i++)
		{
			std::cout << "EPISODE : " << i << " REWARD : " << this->totalReturn[i].get(1,1) << std::endl;
		}
	}
	
	Mat<float> getTotalReturn(unsigned int idxEpisode)
	{
		if(idxEpisode < this->nbrepisode)
		{
			return this->totalReturn[idxEpisode];
		}
		
		return Mat<float>((float)0,1,1);
	}
};




//---------------------------------------------------------------------------------------------------------








class DDPGA3C
{
	private :
	
	PA<float>* pa;				//ACTOR		//Function Approximator from which we retrieve the policy.
	FA<float>* fa;				//CRITIC	//Function Approximator from which we retrieve the Q values.
	PA<float>* targetPA;
	FA<float>* targetFA;
	Environment<float>* env;	//environnement from which we retrieve the rewards and the state.
	
	float gamma;				//discount factor
	unsigned int nbrepisode;
	
	std::vector<Mat<float> > totalReturn;
	
	unsigned int nbrReplay;
	std::vector<XPActorCritic> bankXP;
	std::mutex mutexXP;
	
	XPActorCritic meanXP;
	XPActorCritic stdXP;
	bool batchNormInit;
	
	int nbrEpisodeBankXPHolder;	
	float momentumUpdate;
	
	int nbrThread;
	int nbrThreadNeeded;
	std::mutex mInit;
	std::mutex mPAupdate;
	std::mutex mFAupdate;
	std::mutex mSOFTupdate;
	std::vector<std::mutex*> mXPrefresh;
	std::vector<std::thread> threadsA3C;
	int freqUpdate;
	int counterUpdate;
	int nbrIteration;
	std::vector<bool> NNupdateRequirement;
	bool needXP;
	
	int nbraction;
	
	public :
	
	DDPGA3C(const unsigned int& nbrepi, const float& gamma_, Environment<float>* env_, FA<float>* fa_, PA<float>* pa_, const float& momentumUpdate_, int freqUpdate_, int nbraction_ = 1) : gamma(gamma_), env(env_), fa(fa_), pa(pa_), nbrepisode(nbrepi),nbrReplay(100), nbrEpisodeBankXPHolder(20), momentumUpdate(momentumUpdate_), batchNormInit(false), nbrThread(0), freqUpdate(freqUpdate_), counterUpdate(0), needXP(false), nbraction(nbraction_)
	{
		//TODO : evaluate the number of REPLAY NEEDED.
		targetFA = (FA<float>*) new QFANN<float>( fa_->getLR(), fa_->getEPS(), gamma_, ((QFANN<float>*)fa_)->dimActionSpace, fa_->getNetPointer()->topology, ((QFANN<float>*)fa)->filepathRSNN+std::string(".FAtarget"));
		((QFANN<float>*)targetFA)->VvaluesOnly = ((QFANN<float>*)fa)->VvaluesOnly;
		
		//*(((QFANN<float>*)targetFA)->getNetPointer()) = *(((QFANN<float>*)fa)->getNetPointer());
		targetFA->updateToward( fa, 1.0f);
		
		targetPA = (PA<float>*) new QPANN<float>( pa_->getLR(), pa_->getEPS(), gamma_, ((QPANN<float>*)pa_)->dimActionSpace, pa_->getNetPointer()->topology, ((QPANN<float>*)pa)->filepathRSNN+std::string(".PAtarget") );
		//*(((QPANN<float>*)targetPA)->getNetPointer()) = *(((QPANN<float>*)pa)->getNetPointer());
		targetPA->updateToward( pa, 1.0f);
	}
	
	~DDPGA3C()
	{
		delete targetFA;
		delete targetPA;
	}
	
	void run(unsigned int nbrepisode = 100, unsigned int nbrThreadNeeded = 8)
	{
		this->nbrIteration = nbrepisode;
		
		this->nbrThreadNeeded = nbrThreadNeeded;
		this->nbrepisode = nbrepisode;
		
		this->fa->initialize();
		this->pa->initialize();
		
		std::vector<Mat<float> > QerrorBATCH;
		std::vector<Mat<float> > testTOTALRETURN;
		std::vector<Mat<float> > testMEANRETURN;
		std::vector<Mat<float> > testLOSS;
		std::vector<Mat<float> > testMINRETURN;
		std::vector<Mat<float> > testQSA_STD;
		std::vector<Mat<float> > testQSA_MEAN;
		std::vector<Mat<float> > testA_STD;
		std::vector<Mat<float> > testA_MEAN;
		
		int counterTest = 0;
		int valTest = 1;
		
		int bankSizeNeeded = 1000;
		
		int batchSize = nbrReplay;
		
		int itLOG = 0;
		int freqLOG = 100;
		
	#ifndef EVALONLY
		//CREATION OF THE THREADS :
		#ifdef ACTORCRITICDDPG
		this->needXP=true;
		for(int k=nbrThreadNeeded;k--;)	threadsA3C.push_back( std::thread( &DDPGA3C::runThread, std::ref(*this) ) ); 
		#endif
		//for(int k=nbrThreadNeeded;k--;)	threadsA3C.push_back( std::thread( &DDPGA3C::runThreadEpisode, std::ref(*this) ) ); 
		
		#ifdef ACTORCRITICVVALUES
		for(int k=nbrThreadNeeded;k--;)	threadsA3C.push_back( std::thread( &DDPGA3C::runThreadEpisode2, std::ref(*this) ) ); 
		#endif
	#endif
			
		bool continuer = true;
				
		for(int i=0;i<=nbrepisode;i++)
		{
			#ifdef ENVnorandomINIT
			this->env->initialize(false);
			#else
			this->env->initialize(true);
			#endif
			
		    unsigned int iteration = 0;
			while( !this->env->isTerminal() )
			{
				clock_t time = clock();
		
				Mat<float> S(this->env->getCurrentState() );
				//Mat<float> A(this->pa->estimateAction(S));
				Mat<float> A(this->pa->eps_greedy(S));
				Mat<float> Qsa( this->fa->getQvalue(S,A) );


				Mat<float> R(this->env->executeAction(A));
				Mat<float> S1(this->env->getCurrentState());
			
				
				//METHOD 2 : DDPG..
				
				Mat<float> A1( this->pa->estimateAction(S1) ); 
				/*
				Mat<float> Qs1a1(  this->fa->getQvalue(S1,A1) );
				Mat<float> error(R+gamma*Qs1a1-Qsa);
				//this->fa->updateDelta(operatorC(S,A), (-1.0f)*error );
				*/
				
				
				/*
				Mat<float> Aprevious(A);
				Mat<float> Rprevious(R);
				for(int kk=1000;kk--;)
				{
				this->env->initialize(false);
				Mat<float> St(this->env->getCurrentState());
				Mat<float> At( this->pa->estimateAction(St) ); 
				Mat<float> Qsat( this->fa->getQvalue(St,At) );
				Mat<float> Rt(this->env->executeAction(At));
				Mat<float> S1t(this->env->getCurrentState());
				Mat<float> A1t( this->pa->estimateAction(S1t) ); 
				Mat<float> Qs1a1t(  this->fa->getQvalue(S1t,A1t) );
				Mat<float> errort(Qsat-(R+gamma*Qs1a1t) );
				this->fa->updateDelta(operatorC(St,At), errort );
				
				std::cout << " iteration : " << kk << " ; ERROR NORM FA : " << norme2(errort) << std::endl;
				
				Mat<float> delta( Rt+gamma*Qs1a1t - Qsat);				
				//Mat<float> delta( Rt);				
				//Mat<float> dlogpa( inverseM(At) );
				Mat<float> dlogpa( 1.0f,1,1 );
				Mat<float> error( (1.0f)*(dlogpa*delta));
				this->pa->updateDelta(St, error);
			
				//std::cout << " iteration : " << kk << " ; ERROR NORM PA: " << norme2(error) << " dif A1 - A  " << (At-Aprevious).get(1,1) << " REWARD : " << Rt.get(1,1) << " diff : " << (Rt-Rprevious).get(1,1) << " theta = " << S1t.get(2,1)*180.0f/PI <<  std::endl;
				
				Rprevious = Rt;
				Aprevious = At;
				
				}
				
				throw;
				*/
				
				
				
				
				//POLICY GRADIENT : METHOD 3 : DDPG : fixed target : idem for the policy...
				/*
				this->fa->getQvalue(S,A); 
				Mat<float> dQda( this->fa->getNetPointer()->getGradientWRTinput() );
				Mat<float> satoa( 0.0f, dQda.getColumn(), A.getColumn() );
				for(int k=1;k<=A.getColumn();k++)	satoa.set( 1.0f, S.getLine()+k, k);
				dQda = dQda * satoa;
				//this->pa->updateDelta(S, (1.0f/nbrReplay)*dQda);
				/**/
				
				/*
				for(int kk=1000;kk--;)
				{
				this->fa->getQvalue(S,A); 
				Mat<float> dQda( this->fa->getNetPointer()->getGradientWRTinput() );
				Mat<float> satoa( 0.0f, dQda.getColumn(), A.getColumn() );
				for(int k=1;k<=A.getColumn();k++)	satoa.set( 1.0f, S.getLine()+k, k);
				dQda = dQda * satoa;
				Mat<float> error((1.0f/nbrReplay)*dQda);
				this->pa->updateDelta(S, error);
				
				std::cout << " iteration : " << kk << " ; ERROR NORM : " << norme2(error) << std::endl;
				
				}
				
				throw;
				*/
				
				
				
				//POLICY GRADIENT : METHOD 4: TD 0 ACtor Critic : stochastic gradient
				/**/
				//TODO : this is not a probability...
				/*
				Mat<float> delta( R+gamma*Qs1a1 - Qsa);				
				Mat<float> dlogpa( inverseM(A) );
				//this->pa->updateDelta(S, (-1.0f)*dlogpa*delta);
				/**/
				
				//VALIDATION OF THE ASCENT BY USING A MINUS DLOGDA * DELTA...
				/*
				//Mat<float> Aprevious(A);
				//Mat<float> Rprevious(R);
				for(int kk=1000;kk--;)
				{
					this->env->initialize(false);
					Mat<float> St(this->env->getCurrentState());
					Mat<float> At(this->pa->estimateAction(St));
					Mat<float> Qsat( this->fa->getQvalue(St,At) );
					Mat<float> Rt(this->env->executeAction(At));
					Mat<float> S1t(this->env->getCurrentState());
					Mat<float> A1t( this->pa->estimateAction(S1t) ); 
					Mat<float> Qs1a1t(  this->fa->getQvalue(S1t,A1t) );
					Mat<float> delta( Rt+gamma*Qs1a1t - Qsat);				
					//Mat<float> delta( Rt);				
					//Mat<float> dlogpa( inverseM(At) );
					Mat<float> dlogpa( 1.0f,1,1 );
					Mat<float> error( (-1.0f)*(dlogpa*delta));
					this->pa->updateDelta(St, error);
				
					std::cout << " iteration : " << kk << " ; ERROR NORM : " << norme2(error) << " dif A1 - A  " << (At-Aprevious).get(1,1) << " REWARD : " << Rt.get(1,1) << " diff : " << (Rt-Rprevious).get(1,1) << " theta = " << S1t.get(2,1)*180.0f/PI <<  std::endl;
					
					Rprevious = Rt;
					Aprevious = At;
				}
				
				throw;
				*/
				
				
				//totalReturn :
				if(iteration == 0)
				{
					this->totalReturn.push_back(R);
				}
				else
				{
					this->totalReturn[this->totalReturn.size()-1]+=R;
				}
				
				
				iteration++;
				
				//---------------------------------------
				//---------------------------------------
				//---------------------------------------
				//XP REPLAY :
				//add the new exp :
				if( this->needXP)
				{
					this->mutexXP.lock();
					this->bankXP.push_back( XPActorCritic(S,A,S1,A1,R) );
					this->mutexXP.unlock();
				}
				//---------------------------------------
				//---------------------------------------
				//---------------------------------------
				
				if( itLOG > freqLOG)
				{
					itLOG = 0;
					std::cout << " EXPLORER :::: " ;
					std::cout << " Iteration : " << iteration << " : " << (float)(clock()-time)/CLOCKS_PER_SEC << " seconds." << " Rt = " << R.get(1,1) << " ; Q(s,a) = " << Qsa.get(1,1) << " ; Action performed : " << std::endl;
					transpose(A).afficher();
				}
				else
				{
					itLOG++;
				}
				
			}
			
			
			//VALIDATION TEST :
			if(counterTest >= valTest)
			{
				counterTest = 0;
				Mat<float> tempTR(0.0f,1,1);
				Mat<float> tempLOSS(1.0f,1,1);
				Mat<float> tempTMR(-1e20f,1,1);
				Mat<float> tempQSAMEAN(0.0f,1,1);
				std::vector<Mat<float> > tempQSASTD;
				Mat<float> tempAMEAN(0.0f,nbraction,1);
				std::vector<Mat<float> > tempASTD;
				
				#ifdef ENVnorandomINIT
				this->env->initialize(false);
				#else
				this->env->initialize(true);
				#endif
				
				int nbrsteps = 1;
				while( !this->env->isTerminal() )
				{

					Mat<float> St(this->env->getCurrentState() );
					Mat<float> At(this->pa->estimateAction(St));
					Mat<float> Qsat( this->fa->getQvalue(St,At) );

					Mat<float> Rt(this->env->executeAction(At));
					Mat<float> S1t(this->env->getCurrentState());
					
					Mat<float> A1t(this->pa->estimateAction(S1t));
					//XP REPLAY :
					//add the new exp :
					//this->bankXP.push_back( XPActorCritic(St,At,S1t,A1t,Rt) );
					
					tempTR += Rt;
					Mat<float> Qs1a1t(  this->fa->getQvalue(S1t,A1t) );
					Mat<float> error(Rt+gamma*Qs1a1t-Qsat);
					tempLOSS += error % error;
					if( tempTMR.get(1,1) < Rt.get(1,1) )
					{
						tempTMR.set( Rt.get(1,1), 1,1);
					}
					
					tempQSAMEAN = ( ((float)(nbrsteps))/((float)(nbrsteps+1)) )*tempQSAMEAN + (1.0f/(nbrsteps+1))*Qsat;
					tempQSASTD.push_back( Qsat);
					
					tempAMEAN = ( ((float)(nbrsteps))/((float)(nbrsteps+1)) )*tempAMEAN + (1.0f/(nbrsteps+1))*At;
					tempASTD.push_back( At);
					
					nbrsteps++;
				}
				
				testMEANRETURN.push_back( (1.0f/nbrsteps)*tempTR );
				testTOTALRETURN.push_back( tempTR);
				tempLOSS.set( log(tempLOSS.get(1,1)),1,1);
				testLOSS.push_back( tempLOSS);
				testMINRETURN.push_back( tempTMR );
				testQSA_MEAN.push_back( tempQSAMEAN );
				testA_MEAN.push_back( transpose(tempAMEAN) );
				
				//compute std : QSA :
				Mat<float> tempQSASTDev(0.0f,1,1);
				for(int i=tempQSASTD.size();i--;)
				{
					tempQSASTDev += (tempQSASTD[i]-tempQSAMEAN) % (tempQSASTD[i]-tempQSAMEAN);
				}
				tempQSASTDev.set( sqrt( (1.0f/(nbrsteps-1)) * tempQSASTDev.get(1,1) ), 1,1);
				testQSA_STD.push_back(tempQSASTDev);
				
				//compute std : A :
				Mat<float> tempASTDev(0.0f,nbraction,nbraction);
				for(int i=tempASTD.size();i--;)
				{
					tempASTDev += (tempASTD[i]-tempAMEAN) * transpose(tempASTD[i]-tempAMEAN);
				}
				//tempASTDev.set( sqrt( (1.0f/(nbrsteps-1)) * tempASTDev.get(1,1) ), 1,1);
				Mat<float> tempASTDline( 1,nbraction);
				for(int l=1;l<=nbraction;l++)	tempASTDline.set( sqrt( (1.0f/(nbrsteps-1)) * tempASTDev.get(l,l) ), 1,l);
				testA_STD.push_back(tempASTDline);
				
				
				writeInFile(std::string("./TEST_MinReturn.txt"), testMINRETURN);
				writeInFile(std::string("./TEST_MeanReturn.txt"), testMEANRETURN);
				writeInFile(std::string("./TEST_TotalReturn.txt"), testTOTALRETURN);
				writeInFile(std::string("./TEST_QSA_MEAN.txt"), testQSA_MEAN);
				writeInFile(std::string("./TEST_QSA_STD.txt"), testQSA_STD);
				
				writeInFile(std::string("./TEST_A_MEAN.txt"), testA_MEAN);
				writeInFile(std::string("./TEST_A_STD.txt"), testA_STD);
				
				writeInFile(std::string("./TEST_Loss.txt"), testLOSS);
				
				
			}
			else
			{
				counterTest++;
			}
			
			
			std::cout << " EXPLORER :::: " ;
			std::cout << "EPISODE : " << i << " REWARD : " << this->totalReturn[this->totalReturn.size()-1].get(1,1) << std::endl;
			
			
			
			if( bankXP.size() > nbrEpisodeBankXPHolder*nbrReplay)
			{
				for(int t=this->nbrThread;t--;)	this->mXPrefresh[t]->lock();
				
				for(int t=nbrEpisodeBankXPHolder/2*nbrReplay;t--;)	this->bankXP.erase(this->bankXP.begin());
				
				for(int t=this->nbrThread;t--;)	this->mXPrefresh[t]->unlock();
				
				
				this->batchNormInit = false;
				this->batchNormalization();
			}
			
			writeInFile(std::string("./totalreturn.txt"), totalReturn);
			
			
			
			continuer = false;
			for(int t=nbrThread;t--;)
			{
				if( !threadsA3C[t].joinable())
				{
					continuer = true;
					break;
				}
			}
			
			if(continuer && i == nbrepisode)
			{
				nbrepisode++;
			}
		}		
		
		//let us print out the rewards of each exploration tests :
		for(int i=0;i<this->totalReturn.size();i++)
		{
			std::cout << " EXPLORER :::: " ;
			std::cout << "EPISODE : " << i << " REWARD : " << this->totalReturn[i].get(1,1) << std::endl;
		}
		
		for(int i=threadsA3C.size();i--;)
		{
			if(threadsA3C[i].joinable())
			{
				threadsA3C[i].join();
			}
		}
	}
	
	Mat<float> getTotalReturn(unsigned int idxEpisode)
	{
		if(idxEpisode < this->nbrepisode)
		{
			return this->totalReturn[idxEpisode];
		}
		
		return Mat<float>((float)0,1,1);
	}
	
	void updateTARGET()
	{
		bool retfa = this->targetFA->updateToward( this->fa, momentumUpdate);
		bool retpa = this->targetPA->updateToward( this->pa, momentumUpdate);
		
		std::cout << " UPDATE THE TARGETS : momentum : " << momentumUpdate << " ; results fa pa : " << retfa << " " << retpa << std::endl;
	}
	
	void batchNormalization()
	{
		if(!this->batchNormInit)
		{
			this->batchNormInit = true;
			int nbrxp = bankXP.size();
			float inbrxp = 1.0f/((float)(nbrxp)+numeric_limits<float>::epsilon());
			Mat<float> Sm( 0.0f*bankXP[0].S);
			Mat<float> Am( 0.0f*bankXP[0].A);
		
		
			for(int i=nbrxp;i--;)
			{
				Sm += inbrxp*bankXP[i].S;
				Am += inbrxp*bankXP[i].A;
			}
		
			float inobias = 1.0f/((float)(nbrxp-1)+numeric_limits<float>::epsilon());
			Mat<float> Sv( 0.0f*Sm);
			Mat<float> Av( 0.0f*Am);
		
		
			for(int i=nbrxp;i--;)
			{
				Sv += inobias * ( Sm-bankXP[i].S) % ( Sm-bankXP[i].S) ;
				Av += inobias * ( Am-bankXP[i].A) % ( Am-bankXP[i].A) ;			
			}
		
			meanXP = XPActorCritic( Sm, Am);
			stdXP = XPActorCritic( sqrt(Sv), sqrt(Av));
		
			
			this->pa->setInputNormalization(meanXP.S,stdXP.S);
			this->targetPA->setInputNormalization(meanXP.S,stdXP.S);
			this->fa->setInputNormalization( operatorC(meanXP.S,meanXP.A) , operatorC(stdXP.S, stdXP.A) );
			this->targetFA->setInputNormalization( operatorC(meanXP.S,meanXP.A) , operatorC(stdXP.S, stdXP.A) );
			
			/*
			meanXP.S.afficher();
			meanXP.A.afficher();
			stdXP.S.afficher();
			stdXP.A.afficher();
			std::cout << inbrxp << " " << inobias << std::endl;
			throw;
			*/
		}
		
		
	}
	
	
	void runThread()
	{
		mInit.lock();
		this->nbrThread++;
		int idxThread = this->nbrThread;
		this->NNupdateRequirement.push_back(false);
		this->mXPrefresh.push_back( new std::mutex() );
		mInit.unlock();
		
		NNTrainer<float> trainerPA( ((QPANN<float>*)this->pa)->getNetPointer());
		NNTrainer<float> trainerFA( ((QFANN<float>*)this->fa)->getNetPointer());
		
		for(int iteration=0;iteration<=nbrIteration;iteration++)
		{
			if(bankXP.size()>2*nbrReplay)
			{
				for(int i=(nbrReplay<bankXP.size()?nbrReplay:bankXP.size());i--;)
				{
					//NN UPDATE :
					//TODO:estimate the need of a mutex or not ?
					if(this->NNupdateRequirement[idxThread-1])
					{
						trainerPA.updateToward( ((QPANN<float>*)this->pa)->getNetPointer(), 1.0f );
						trainerFA.updateToward( ((QFANN<float>*)this->fa)->getNetPointer(), 1.0f );
						this->NNupdateRequirement[idxThread-1] = false;
					}
					
					
					
					this->mXPrefresh[idxThread-1]->lock();
					
					this->mutexXP.lock();
					int idx = rand()%bankXP.size();
					Mat<float> S(this->bankXP[idx].S);
					Mat<float> A(this->bankXP[idx].A);
					Mat<float> S1(this->bankXP[idx].S1);
					Mat<float> A1(this->bankXP[idx].A1);
					Mat<float> R(this->bankXP[idx].R);
					this->mutexXP.unlock();
					
					//METHOD DDPG :
					/*
					Mat<float> bankAfromS1( this->pa->estimateAction(S1) ); 
					Mat<float> bankQs1a1(  this->fa->getQvalue(bankXP[idx].S1, bankAfromS1) );
					Mat<float> bankQsa( this->fa->getQvalue(bankXP[idx].S,bankXP[idx].A) );
					this->fa->updateDelta(operatorC(bankXP[idx].S,bankXP[idx].A), bankXP[idx].R+gamma*bankQs1a1-bankQsa );
					*/
				
					//METHOD DDPG or DSPG: fixed target :
					//Mat<float> bankAfromS1TARGET( this->targetPA->estimateAction(bankXP[idx].S1) ); 
					Mat<float> bankAfromS1TARGET( this->targetPA->estimateAction(S1) ); 
					//Mat<float> bankQs1a1TARGET(  this->targetFA->getQvalue(bankXP[idx].S1, bankAfromS1TARGET) );
					Mat<float> bankQs1a1TARGET(  this->targetFA->getQvalue(S1, bankAfromS1TARGET) );
					//Mat<float> bankQsa( this->fa->getQvalue(bankXP[idx].S,bankXP[idx].A) );
					Mat<float> bankQsa( this->fa->getQvalue(S,A) );
					//Mat<float> bankerror(bankXP[idx].R+gamma*bankQs1a1TARGET-bankQsa);
					Mat<float> bankerror(R+gamma*bankQs1a1TARGET-bankQsa);
					//this->fa->updateDeltaBATCH(operatorC(bankXP[idx].S,bankXP[idx].A), (-1.0f)*bankerror, batchSize );
					//trainerFA.accumulateGradient( operatorC(bankXP[idx].S,bankXP[idx].A), (-1.0f)*bankerror);
					trainerFA.accumulateGradient( operatorC(S,A), (-2.0f)*bankerror);
				
				
					//POLICY GRADIENT : METHOD 4: TD 0 ACtor Critic
					/*
					Mat<float> bankdelta( bankXP[idx].R+gamma*bankQs1a1 - bankQsa);				
					Mat<float> bankdlogpa( inverseM(bankXP[idx].A) );
					this->pa->updateDelta(bankXP[idx].S, bankdlogpa*bankdelta);
					*/
					
					//POLICY GRADIENT : METHOD DDPG : fixed target : idem for the policy...
					/**/
					//this->fa->getQvalue( bankXP[idx].S, this->pa->estimateAction(bankXP[idx].S) ); 
					//Mat<float> input( operatorC( bankXP[idx].S, this->pa->estimateAction(bankXP[idx].S) ) );
					Mat<float> input( operatorC(S, this->pa->estimateAction(S) ) );
					Mat<float> bankdQda( this->fa->getNetPointer()->getGradientWRTinput( &input ) );
					//Mat<float> banksatoa( 0.0f, bankdQda.getColumn(), bankXP[idx].A.getColumn() );
					Mat<float> banksatoa( 0.0f, bankdQda.getColumn(),A.getLine() );
					//for(int k=1;k<=bankXP[idx].A.getColumn();k++)	banksatoa.set( 1.0f, bankXP[idx].S.getLine()+k, k);
					for(int k=1;k<=A.getLine();k++)	banksatoa.set( 1.0f, S.getLine()+k, k);
					
					
					bankdQda = bankdQda * banksatoa;
					
					
					float normedqda = sqrt( norme2(bankdQda) );
					//if(i==0)	std::cout << " DELTA Q da : " << bankdQda.get(1,1) << std::endl;
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (-1.0f/nbrReplay) * bankdQda, batchSize);
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (1.0f/nbrReplay) * bankdQda, batchSize);
					//trainerPA.accumulateGradient(bankXP[idx].S, (-1.0f/nbrReplay) * bankdQda);
					//trainerPA.accumulateGradient(S, (-1.0f/nbrReplay) * bankdQda);
					trainerPA.accumulateGradient(S, (-2.0f/normedqda) * bankdQda);
					
					float normeAcc = norme2( (-2.0f/normedqda) * bankdQda );
					
					/*
					//DEBUG :
					std::cout << " NORME grad : " << normedqda << std::endl;
					std::cout << " NORME ACCUMULATION : " << normeAcc << std::endl;
					*/
					
					/**/
				
					//POLICY GRADIENT : METHOD 6: TD 0 ACtor Critic : DSPG+fixed target
					/*
					Mat<float> bankdelta( bankXP[idx].R+gamma*bankQs1a1TARGET - bankQsa);				
					Mat<float> bankdlogpa( 2.0f,1,1 );
					//inverse of probability of 0.5;
					//TODO : this is not stochastic : not a probability...
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (-1.0f/nbrReplay)*(bankdlogpa*bankdelta), batchSize);
					trainerPA.accumulateGradient(bankXP[idx].S, (-1.0f/nbrReplay)*(bankdlogpa*bankdelta));
					/**/
					
					
					this->mXPrefresh[idxThread-1]->unlock();
				}


				//UPDATE OF THE NETWORKS now that a minibatch has been done :
				this->mPAupdate.lock();
				//std::cout << " THREAD : " << idxThread << " :::: " << " PA : " ;
				//trainerPA(GCSGD);
				trainerPA(GCSGDMomentum);
				this->mPAupdate.unlock();
			
				this->mFAupdate.lock();
				//std::cout << " THREAD : " << idxThread << " :::: " << " FA : " ;
				//trainerFA(GCSGD);
				trainerFA(GCSGDMomentum);
				
				for(int t=this->nbrThread;t--;)
				{
					if( t != idxThread)
					{
						this->NNupdateRequirement[t] = true;
					}
				}
				
				this->mFAupdate.unlock();
			
			
				/*
				//SOFT UPDATE TARGET TO REAL NETWORK :
				if( iteration % freqUpdate == 0)
				{
					this->mSOFTupdate.lock();
					std::cout << " THREAD : " << idxThread << " :::: SOFT UPDATE : " ;
					this->updateTARGET();
					this->mSOFTupdate.unlock();
				}
				*/
				
				//SOFT UPDATE TARGET TO REAL NETWORK :
				if( this->counterUpdate >= this->freqUpdate )
				{
					//this->mSOFTupdate.lock();
					this->counterUpdate = 0;
					std::cout << " THREAD : " << idxThread << " :::: SOFT UPDATE : " ;
					this->updateTARGET();
					//this->mSOFTupdate.unlock();
				}
				else
				{
					//this->mSOFTupdate.lock();
					this->counterUpdate++;
					//this->mSOFTupdate.unlock();
				}
							
			}
			else
			{
				iteration--;
			}

		}
		
	} 
	
	
	
	void runThreadEpisode()
	{
		mInit.lock();
		this->nbrThread++;
		int idxThread = this->nbrThread;
		this->NNupdateRequirement.push_back(false);
		this->mXPrefresh.push_back( new std::mutex() );
		mInit.unlock();
		
		NNTrainer<float> trainerPA( ((QPANN<float>*)this->pa)->getNetPointer());
		NNTrainer<float> trainerFA( ((QFANN<float>*)this->fa)->getNetPointer());
		
		//CARTPOLE :
		SimulatorRKCARTPOLE envt(5.0f);
		
		//CARTPOLEDISCRETE :
		//SimulatorRKCARTPOLE_Discrete envt(5.0f);
		
		for(int i=0;i<=nbrIteration;i++)
		{
			envt.initialize(false);	//random initial state or predefined...
			
		    unsigned int iteration = 0;
			while( !envt.isTerminal() )
			{
				clock_t time = clock();
		
				Mat<float> S(envt.getCurrentState() );
				//Mat<float> A(this->pa->estimateAction(S));
				Mat<float> A(this->pa->eps_greedy(S));
				Mat<float> R(envt.executeAction(A));
				Mat<float> S1(envt.getCurrentState());
			
				//NN UPDATE :
				//TODO:estimate the need of a mutex or not ?
				if(this->NNupdateRequirement[idxThread-1])
				{
					trainerPA.updateToward( ((QPANN<float>*)this->pa)->getNetPointer(), 1.0f );
					trainerFA.updateToward( ((QFANN<float>*)this->fa)->getNetPointer(), 1.0f );
					this->NNupdateRequirement[idxThread-1] = false;
				}
					
										
					//METHOD DDPG :
					/*
					Mat<float> bankAfromS1( this->pa->estimateAction(S1) ); 
					Mat<float> bankQs1a1(  this->fa->getQvalue(bankXP[idx].S1, bankAfromS1) );
					Mat<float> bankQsa( this->fa->getQvalue(bankXP[idx].S,bankXP[idx].A) );
					this->fa->updateDelta(operatorC(bankXP[idx].S,bankXP[idx].A), bankXP[idx].R+gamma*bankQs1a1-bankQsa );
					*/
				
					//METHOD DDPG or DSPG: fixed target :
					Mat<float> AfromS1TARGET( this->targetPA->estimateAction(S1) ); 
					Mat<float> Qs1a1TARGET(  this->targetFA->getQvalue(S1, AfromS1TARGET) );
					Mat<float> Qsa( this->fa->getQvalue(S,A) );
					Mat<float> error(R+gamma*Qs1a1TARGET-Qsa);
					//this->fa->updateDeltaBATCH(operatorC(bankXP[idx].S,bankXP[idx].A), (-1.0f)*bankerror, batchSize );
					trainerFA.accumulateGradient( operatorC(S,A), (-1.0f)*error);
				
				
					//POLICY GRADIENT : METHOD 4: TD 0 ACtor Critic
					/*
					Mat<float> bankdelta( bankXP[idx].R+gamma*bankQs1a1 - bankQsa);				
					Mat<float> bankdlogpa( inverseM(bankXP[idx].A) );
					this->pa->updateDelta(bankXP[idx].S, bankdlogpa*bankdelta);
					*/
					
					//POLICY GRADIENT : METHOD DDPG : fixed target : idem for the policy...
					/**/
					//this->fa->getQvalue( bankXP[idx].S, this->pa->estimateAction(bankXP[idx].S) ); 
					Mat<float> input( operatorC( S, this->pa->estimateAction(S) ) );
					Mat<float> dQda( this->fa->getNetPointer()->getGradientWRTinput( &input ) );
					Mat<float> satoa( 0.0f, dQda.getColumn(), A.getLine() );
					for(int k=1;k<=A.getLine();k++)	satoa.set( 1.0f, S.getLine()+k, k);
					dQda = dQda * satoa;
					//if(i==0)	std::cout << " DELTA Q da : " << bankdQda.get(1,1) << std::endl;
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (-1.0f/nbrReplay) * bankdQda, batchSize);
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (1.0f/nbrReplay) * bankdQda, batchSize);
					trainerPA.accumulateGradient( S, dQda);
					/**/
				
					//POLICY GRADIENT : METHOD 6: TD 0 ACtor Critic : DSPG+fixed target
					/*
					Mat<float> delta(R+gamma*Qs1a1TARGET - Qsa);				
					Mat<float> logpa( 2.0f,1,1 );
					//inverse of probability of 0.5
					//TODO : this is not stochastic : not a probability...
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (-1.0f/nbrReplay)*(bankdlogpa*bankdelta), batchSize);
					trainerPA.accumulateGradient(S, (-1.0f/freqUpdate)*(logpa*delta));
					/**/



				//UPDATE of the real network :
				this->mPAupdate.lock();
				//std::cout << " THREAD : " << idxThread << " :::: " << " PA : " ;
				trainerPA(GCSGD);
				this->mPAupdate.unlock();
			
				this->mFAupdate.lock();
				//std::cout << " THREAD : " << idxThread << " :::: " << " FA : " ;
				trainerFA(GCSGD);
				
				for(int t=this->nbrThread;t--;)
				{
					if( t != idxThread)
					{
						this->NNupdateRequirement[t] = true;
					}
				}
				
				this->mFAupdate.unlock();
			
			
				//SOFT UPDATE TARGET TO REAL NETWORK :
				if( iteration % freqUpdate == 0)
				{
					this->mSOFTupdate.lock();
					std::cout << " THREAD : " << idxThread << " :::: SOFT UPDATE : " ;
					this->updateTARGET();
					this->mSOFTupdate.unlock();
				}
				iteration++;
			}					
		}
		
	}
	
	
	void runThreadEpisode2()
	{
		mInit.lock();
		this->nbrThread++;
		int idxThread = this->nbrThread;
		this->NNupdateRequirement.push_back(false);
		this->mXPrefresh.push_back( new std::mutex() );
		mInit.unlock();
		
		NNTrainer<float> trainerPA( ((QPANN<float>*)this->pa)->getNetPointer());
		NNTrainer<float> trainerFA( ((QFANN<float>*)this->fa)->getNetPointer());
		
		//CARTPOLE :
		SimulatorRKCARTPOLE envt(this->env->getEOE());
		
		//CARTPOLEDISCRETE :
		//SimulatorRKCARTPOLE_Discrete envt(5.0f);
		
		int counterBATCH = 0;
		int valBATCH = 3;
		
		for(int it=0;it<this->nbrIteration;it++)
		{
			envt.initialize(false);	//random initial state or predefined...
			
			std::vector<XPActorCritic> bankXP;
			Mat<float> Rt(1,1);

			while( !envt.isTerminal() )
			{
				clock_t time = clock();
		
				Mat<float> S(envt.getCurrentState() );
				//Mat<float> A(this->pa->estimateAction(S));
				Mat<float> A(this->pa->eps_greedy(S));
				Rt = envt.executeAction(A);
				Mat<float> S1(envt.getCurrentState());
				Mat<float> A1(this->pa->eps_greedy(S1));
				
				bankXP.push_back( XPActorCritic(S,A,S1,A1,Rt));
			}
			
			int nbrXP = bankXP.size();
			//if the state is terminal, we do not bootstrap.
			if( Rt.get(1,1) >= 0.45f )
			{
				 Rt *= 0.0f;
			}
			else
			{
				Rt = this->fa->getQvalue( bankXP[nbrXP-1].S, bankXP[nbrXP-1].A);
			}
			
			for(int i=bankXP.size();i--;)
			{
			
				//NN UPDATE :
				//TODO:estimate the need of a mutex or not ?
				if(this->NNupdateRequirement[idxThread-1])
				{
					trainerPA.updateToward( ((QPANN<float>*)this->pa)->getNetPointer(), 1.0f );
					trainerFA.updateToward( ((QFANN<float>*)this->fa)->getNetPointer(), 1.0f );
					this->NNupdateRequirement[idxThread-1] = false;
				}
					
										
					//METHOD DDPG :
					/*
					Mat<float> bankAfromS1( this->pa->estimateAction(S1) ); 
					Mat<float> bankQs1a1(  this->fa->getQvalue(bankXP[idx].S1, bankAfromS1) );
					Mat<float> bankQsa( this->fa->getQvalue(bankXP[idx].S,bankXP[idx].A) );
					this->fa->updateDelta(operatorC(bankXP[idx].S,bankXP[idx].A), bankXP[idx].R+gamma*bankQs1a1-bankQsa );
					*/
				
					//METHOD DDPG or DSPG: fixed target :
					Mat<float> AfromS1TARGET( this->targetPA->estimateAction(bankXP[i].S1) ); 
					Mat<float> Qs1a1TARGET(  this->targetFA->getQvalue(bankXP[i].S1, AfromS1TARGET) );
					Mat<float> Qsa( this->fa->getQvalue(bankXP[i].S,bankXP[i].A) );
					Rt = gamma*Rt+bankXP[i].R;
					//Mat<float> error(Rt+gamma*Qs1a1TARGET-Qsa);
					
					Mat<float> error(Rt-Qsa);
					
					//this->fa->updateDeltaBATCH(operatorC(bankXP[idx].S,bankXP[idx].A), (-1.0f)*bankerror, batchSize );
					#ifndef Vvalues
					trainerFA.accumulateGradient( operatorC(bankXP[i].S,bankXP[i].A), (-1.0f/bankXP.size())*error);
					#else
					trainerFA.accumulateGradient( bankXP[i].S, (-2.0f/bankXP.size())*error);
					#endif
				
				
					//POLICY GRADIENT : METHOD 4: TD 0 ACtor Critic
					/*
					Mat<float> bankdelta( bankXP[idx].R+gamma*bankQs1a1 - bankQsa);				
					Mat<float> bankdlogpa( inverseM(bankXP[idx].A) );
					this->pa->updateDelta(bankXP[idx].S, bankdlogpa*bankdelta);
					*/
					
					//POLICY GRADIENT : METHOD DDPG : fixed target : idem for the policy...
					/*
					//this->fa->getQvalue( bankXP[idx].S, this->pa->estimateAction(bankXP[idx].S) ); 
					Mat<float> input( operatorC( bankXP[i].S, this->pa->estimateAction(bankXP[i].S) ) );
					Mat<float> dQda( this->fa->getNetPointer()->getGradientWRTinput( &input ) );
					Mat<float> satoa( 0.0f, dQda.getColumn(), bankXP[i].A.getLine() );
					for(int k=1;k<=bankXP[i].A.getLine();k++)	satoa.set( 1.0f, bankXP[i].S.getLine()+k, k);
					dQda = dQda * satoa;
					if(isnanM(dQda))
					{
						throw;
					}
					//if(i==0)	std::cout << " DELTA Q da : " << bankdQda.get(1,1) << std::endl;
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (-1.0f/nbrReplay) * bankdQda, batchSize);
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (1.0f/nbrReplay) * bankdQda, batchSize);
					//trainerPA.accumulateGradient( bankXP[i].S, dQda);//EXPLODE!!!!
					trainerPA.accumulateGradient( bankXP[i].S, (-1.0f/bankXP.size())*dQda);
					/**/
				
					//POLICY GRADIENT : METHOD 6: TD 0 ACtor Critic : DSPG+fixed target
					/**/
					Mat<float> delta(error);				
					Mat<float> dlogpa( 1.0f/0.5f,1,1 );
					float normedlogpadelta = sqrt( norme2(dlogpa*delta) );
					//inverse of probability of 0.5
					//TODO : this is not stochastic : not a probability...
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (-1.0f/nbrReplay)*(bankdlogpa*bankdelta), batchSize);
					//trainerPA.accumulateGradient(bankXP[i].S, (-1.0f/(freqUpdate*bankXP.size()))*(logpa*delta));
					trainerPA.accumulateGradient(bankXP[i].S, (-2.0f/(normedlogpadelta*bankXP.size()))*(dlogpa*delta));
					/**/
			}


			//UPDATE of the real network :
			if( counterBATCH > valBATCH)
			{
				counterBATCH=0;
				
				
				this->mPAupdate.lock();
				//std::cout << " THREAD : " << idxThread << " :::: " << " PA : " ;
				//trainerPA(GCSGD);
				trainerPA(GCSGDMomentum);
				this->mPAupdate.unlock();
		
				this->mFAupdate.lock();
				//std::cout << " THREAD : " << idxThread << " :::: " << " FA : " ;
				//trainerFA(GCSGD);
				trainerFA(GCSGDMomentum);
			
				for(int t=this->nbrThread;t--;)
				{
					if( t != idxThread)
					{
						this->NNupdateRequirement[t] = true;
					}
				}
			
				this->mFAupdate.unlock();
			}
			else
			{
				counterBATCH++;
			}
		
			//SOFT UPDATE TARGET TO REAL NETWORK :
			if( this->counterUpdate >= this->freqUpdate )
			{
				//this->mSOFTupdate.lock();
				this->counterUpdate = 0;
				std::cout << " THREAD : " << idxThread << " :::: SOFT UPDATE : " ;
				this->updateTARGET();
				//this->mSOFTupdate.unlock();
			}
			else
			{
				//this->mSOFTupdate.lock();
				this->counterUpdate++;
				//this->mSOFTupdate.unlock();
			}
								
		}
		
	} 
};





















//---------------------------------------------------------------------------------------------------------








class DDPGBA2C
{
	private :
	
	PA<float>* pa;				//ACTOR		//Function Approximator from which we retrieve the policy.
	FA<float>* fa;				//CRITIC	//Function Approximator from which we retrieve the Q values.
	PA<float>* targetPA;
	FA<float>* targetFA;
	Environment<float>* env;	//environnement from which we retrieve the rewards and the state.
	
	float gamma;				//discount factor
	unsigned int nbrepisode;
	
	std::vector<Mat<float> > totalReturn;
	
	unsigned int nbrReplay;
	std::vector<XPActorCritic> bankXP;
	std::mutex mutexXP;
	
	XPActorCritic meanXP;
	XPActorCritic stdXP;
	bool batchNormInit;
	
	int nbrEpisodeBankXPHolder;	
	float momentumUpdate;
	
	int nbrThread;
	int nbrThreadNeeded;
	std::mutex mInit;
	std::mutex mPAupdate;
	std::mutex mFAupdate;
	std::mutex mSOFTupdate;
	std::vector<std::mutex*> mXPrefresh;
	std::vector<std::thread> threadsA3C;
	int freqUpdate;
	int counterUpdate;
	int nbrIteration;
	std::vector<bool> NNupdateRequirement;
	bool needXP;
	
	int nbraction;
	
	public :
	
	DDPGBA2C(const unsigned int& nbrepi, const float& gamma_, Environment<float>* env_, FA<float>* fa_, PA<float>* pa_, const float& momentumUpdate_, int freqUpdate_, int nbraction_ = 1) : gamma(gamma_), env(env_), fa(fa_), pa(pa_), nbrepisode(nbrepi),nbrReplay(100), nbrEpisodeBankXPHolder(20), momentumUpdate(momentumUpdate_), batchNormInit(false), nbrThread(0), freqUpdate(freqUpdate_), counterUpdate(0), needXP(false), nbraction(nbraction_)
	{
		//TODO : evaluate the number of REPLAY NEEDED.
		targetFA = (FA<float>*) new QFANN<float>( fa_->getLR(), fa_->getEPS(), gamma_, ((QFANN<float>*)fa_)->dimActionSpace, fa_->getNetPointer()->topology, ((QFANN<float>*)fa)->filepathRSNN+std::string(".FAtarget"));
		((QFANN<float>*)targetFA)->VvaluesOnly = ((QFANN<float>*)fa)->VvaluesOnly;
		
		//*(((QFANN<float>*)targetFA)->getNetPointer()) = *(((QFANN<float>*)fa)->getNetPointer());
		targetFA->updateToward( fa, 1.0f);
		
		targetPA = (PA<float>*) new QPANN<float>( pa_->getLR(), pa_->getEPS(), gamma_, ((QPANN<float>*)pa_)->dimActionSpace, pa_->getNetPointer()->topology, ((QPANN<float>*)pa)->filepathRSNN+std::string(".PAtarget") );
		//*(((QPANN<float>*)targetPA)->getNetPointer()) = *(((QPANN<float>*)pa)->getNetPointer());
		targetPA->updateToward( pa, 1.0f);
	}
	
	~DDPGBA2C()
	{
		delete targetFA;
		delete targetPA;
	}
	
	void run(unsigned int nbrepisode = 100, unsigned int nbrThreadNeeded = 8)
	{
		this->nbrIteration = nbrepisode;
		
		this->nbrThreadNeeded = nbrThreadNeeded;
		this->nbrepisode = nbrepisode;
		
		this->fa->initialize();
		this->pa->initialize();
		
		std::vector<Mat<float> > QerrorBATCH;
		std::vector<Mat<float> > testTOTALRETURN;
		std::vector<Mat<float> > testMEANRETURN;
		std::vector<Mat<float> > testLOSS;
		std::vector<Mat<float> > testMINRETURN;
		std::vector<Mat<float> > testQSA_STD;
		std::vector<Mat<float> > testQSA_MEAN;
		std::vector<Mat<float> > testA_STD;
		std::vector<Mat<float> > testA_MEAN;
		
		int counterTest = 0;
		int valTest = 1;
		
		int bankSizeNeeded = 1000;
		
		int batchSize = nbrReplay;
		
		int itLOG = 0;
		int freqLOG = 100;
		
	#ifndef EVALONLY
		//CREATION OF THE THREADS :
		#ifdef ACTORCRITICDDPG
		this->needXP=true;
		for(int k=nbrThreadNeeded;k--;)	threadsA3C.push_back( std::thread( &DDPGBA2C::runThread, std::ref(*this) ) ); 
		#endif
		//for(int k=nbrThreadNeeded;k--;)	threadsA3C.push_back( std::thread( &DDPGBA2C::runThreadEpisode, std::ref(*this) ) ); 
		
		#ifdef ACTORCRITICVVALUES
		for(int k=nbrThreadNeeded;k--;)	threadsA3C.push_back( std::thread( &DDPGBA2C::runThreadEpisode2, std::ref(*this) ) ); 
		#endif
	#endif
			
		bool continuer = true;
				
		for(int i=0;i<=nbrepisode;i++)
		{
			#ifdef ENVnorandomINIT
			this->env->initialize(false);
			#else
			this->env->initialize(true);
			#endif
			
		    unsigned int iteration = 0;
			while( !this->env->isTerminal() )
			{
				clock_t time = clock();
		
				Mat<float> S(this->env->getCurrentState() );
				//Mat<float> A(this->pa->estimateAction(S));
				Mat<float> A(this->pa->eps_greedy(S));
				Mat<float> Qsa( this->fa->getQvalue(S,A) );


				Mat<float> R(this->env->executeAction(A));
				Mat<float> S1(this->env->getCurrentState());
			
				
				//METHOD 2 : DDPG..
				
				Mat<float> A1( this->pa->estimateAction(S1) ); 
				/*
				Mat<float> Qs1a1(  this->fa->getQvalue(S1,A1) );
				Mat<float> error(R+gamma*Qs1a1-Qsa);
				//this->fa->updateDelta(operatorC(S,A), (-1.0f)*error );
				*/
				
				
				/*
				Mat<float> Aprevious(A);
				Mat<float> Rprevious(R);
				for(int kk=1000;kk--;)
				{
				this->env->initialize(false);
				Mat<float> St(this->env->getCurrentState());
				Mat<float> At( this->pa->estimateAction(St) ); 
				Mat<float> Qsat( this->fa->getQvalue(St,At) );
				Mat<float> Rt(this->env->executeAction(At));
				Mat<float> S1t(this->env->getCurrentState());
				Mat<float> A1t( this->pa->estimateAction(S1t) ); 
				Mat<float> Qs1a1t(  this->fa->getQvalue(S1t,A1t) );
				Mat<float> errort(Qsat-(R+gamma*Qs1a1t) );
				this->fa->updateDelta(operatorC(St,At), errort );
				
				std::cout << " iteration : " << kk << " ; ERROR NORM FA : " << norme2(errort) << std::endl;
				
				Mat<float> delta( Rt+gamma*Qs1a1t - Qsat);				
				//Mat<float> delta( Rt);				
				//Mat<float> dlogpa( inverseM(At) );
				Mat<float> dlogpa( 1.0f,1,1 );
				Mat<float> error( (1.0f)*(dlogpa*delta));
				this->pa->updateDelta(St, error);
			
				//std::cout << " iteration : " << kk << " ; ERROR NORM PA: " << norme2(error) << " dif A1 - A  " << (At-Aprevious).get(1,1) << " REWARD : " << Rt.get(1,1) << " diff : " << (Rt-Rprevious).get(1,1) << " theta = " << S1t.get(2,1)*180.0f/PI <<  std::endl;
				
				Rprevious = Rt;
				Aprevious = At;
				
				}
				
				throw;
				*/
				
				
				
				
				//POLICY GRADIENT : METHOD 3 : DDPG : fixed target : idem for the policy...
				/*
				this->fa->getQvalue(S,A); 
				Mat<float> dQda( this->fa->getNetPointer()->getGradientWRTinput() );
				Mat<float> satoa( 0.0f, dQda.getColumn(), A.getColumn() );
				for(int k=1;k<=A.getColumn();k++)	satoa.set( 1.0f, S.getLine()+k, k);
				dQda = dQda * satoa;
				//this->pa->updateDelta(S, (1.0f/nbrReplay)*dQda);
				/**/
				
				/*
				for(int kk=1000;kk--;)
				{
				this->fa->getQvalue(S,A); 
				Mat<float> dQda( this->fa->getNetPointer()->getGradientWRTinput() );
				Mat<float> satoa( 0.0f, dQda.getColumn(), A.getColumn() );
				for(int k=1;k<=A.getColumn();k++)	satoa.set( 1.0f, S.getLine()+k, k);
				dQda = dQda * satoa;
				Mat<float> error((1.0f/nbrReplay)*dQda);
				this->pa->updateDelta(S, error);
				
				std::cout << " iteration : " << kk << " ; ERROR NORM : " << norme2(error) << std::endl;
				
				}
				
				throw;
				*/
				
				
				
				//POLICY GRADIENT : METHOD 4: TD 0 ACtor Critic : stochastic gradient
				/**/
				//TODO : this is not a probability...
				/*
				Mat<float> delta( R+gamma*Qs1a1 - Qsa);				
				Mat<float> dlogpa( inverseM(A) );
				//this->pa->updateDelta(S, (-1.0f)*dlogpa*delta);
				/**/
				
				//VALIDATION OF THE ASCENT BY USING A MINUS DLOGDA * DELTA...
				/*
				//Mat<float> Aprevious(A);
				//Mat<float> Rprevious(R);
				for(int kk=1000;kk--;)
				{
					this->env->initialize(false);
					Mat<float> St(this->env->getCurrentState());
					Mat<float> At(this->pa->estimateAction(St));
					Mat<float> Qsat( this->fa->getQvalue(St,At) );
					Mat<float> Rt(this->env->executeAction(At));
					Mat<float> S1t(this->env->getCurrentState());
					Mat<float> A1t( this->pa->estimateAction(S1t) ); 
					Mat<float> Qs1a1t(  this->fa->getQvalue(S1t,A1t) );
					Mat<float> delta( Rt+gamma*Qs1a1t - Qsat);				
					//Mat<float> delta( Rt);				
					//Mat<float> dlogpa( inverseM(At) );
					Mat<float> dlogpa( 1.0f,1,1 );
					Mat<float> error( (-1.0f)*(dlogpa*delta));
					this->pa->updateDelta(St, error);
				
					std::cout << " iteration : " << kk << " ; ERROR NORM : " << norme2(error) << " dif A1 - A  " << (At-Aprevious).get(1,1) << " REWARD : " << Rt.get(1,1) << " diff : " << (Rt-Rprevious).get(1,1) << " theta = " << S1t.get(2,1)*180.0f/PI <<  std::endl;
					
					Rprevious = Rt;
					Aprevious = At;
				}
				
				throw;
				*/
				
				
				//totalReturn :
				if(iteration == 0)
				{
					this->totalReturn.push_back(R);
				}
				else
				{
					this->totalReturn[this->totalReturn.size()-1]+=R;
				}
				
				
				iteration++;
				
				//---------------------------------------
				//---------------------------------------
				//---------------------------------------
				//XP REPLAY :
				//add the new exp :
				if( this->needXP)
				{
					this->mutexXP.lock();
					this->bankXP.push_back( XPActorCritic(S,A,S1,A1,R) );
					this->mutexXP.unlock();
				}
				//---------------------------------------
				//---------------------------------------
				//---------------------------------------
				
				if( itLOG > freqLOG)
				{
					itLOG = 0;
					std::cout << " EXPLORER :::: " ;
					std::cout << " Iteration : " << iteration << " : " << (float)(clock()-time)/CLOCKS_PER_SEC << " seconds." << " Rt = " << R.get(1,1) << " ; Q(s,a) = " << Qsa.get(1,1) << " ; Action performed : " << std::endl;
					transpose(A).afficher();
				}
				else
				{
					itLOG++;
				}
				
			}
			
			
			//VALIDATION TEST :
			if(counterTest >= valTest)
			{
				counterTest = 0;
				Mat<float> tempTR(0.0f,1,1);
				Mat<float> tempLOSS(1.0f,1,1);
				Mat<float> tempTMR(-1e20f,1,1);
				Mat<float> tempQSAMEAN(0.0f,1,1);
				std::vector<Mat<float> > tempQSASTD;
				Mat<float> tempAMEAN(0.0f,nbraction,1);
				std::vector<Mat<float> > tempASTD;
				
				#ifdef ENVnorandomINIT
				this->env->initialize(false);
				#else
				this->env->initialize(true);
				#endif
				
				int nbrsteps = 1;
				while( !this->env->isTerminal() )
				{

					Mat<float> St(this->env->getCurrentState() );
					Mat<float> At(this->pa->estimateAction(St));
					Mat<float> Qsat( this->fa->getQvalue(St,At) );

					Mat<float> Rt(this->env->executeAction(At));
					Mat<float> S1t(this->env->getCurrentState());
					
					Mat<float> A1t(this->pa->estimateAction(S1t));
					//XP REPLAY :
					//add the new exp :
					//this->bankXP.push_back( XPActorCritic(St,At,S1t,A1t,Rt) );
					
					tempTR += Rt;
					Mat<float> Qs1a1t(  this->fa->getQvalue(S1t,A1t) );
					Mat<float> error(Rt+gamma*Qs1a1t-Qsat);
					tempLOSS += error % error;
					if( tempTMR.get(1,1) < Rt.get(1,1) )
					{
						tempTMR.set( Rt.get(1,1), 1,1);
					}
					
					tempQSAMEAN = ( ((float)(nbrsteps))/((float)(nbrsteps+1)) )*tempQSAMEAN + (1.0f/(nbrsteps+1))*Qsat;
					tempQSASTD.push_back( Qsat);
					
					tempAMEAN = ( ((float)(nbrsteps))/((float)(nbrsteps+1)) )*tempAMEAN + (1.0f/(nbrsteps+1))*At;
					tempASTD.push_back( At);
					
					nbrsteps++;
				}
				
				testMEANRETURN.push_back( (1.0f/nbrsteps)*tempTR );
				testTOTALRETURN.push_back( tempTR);
				tempLOSS.set( log(tempLOSS.get(1,1)),1,1);
				testLOSS.push_back( tempLOSS);
				testMINRETURN.push_back( tempTMR );
				testQSA_MEAN.push_back( tempQSAMEAN );
				testA_MEAN.push_back( transpose(tempAMEAN) );
				
				//compute std : QSA :
				Mat<float> tempQSASTDev(0.0f,1,1);
				for(int i=tempQSASTD.size();i--;)
				{
					tempQSASTDev += (tempQSASTD[i]-tempQSAMEAN) % (tempQSASTD[i]-tempQSAMEAN);
				}
				tempQSASTDev.set( sqrt( (1.0f/(nbrsteps-1)) * tempQSASTDev.get(1,1) ), 1,1);
				testQSA_STD.push_back(tempQSASTDev);
				
				//compute std : A :
				Mat<float> tempASTDev(0.0f,nbraction,nbraction);
				for(int i=tempASTD.size();i--;)
				{
					tempASTDev += (tempASTD[i]-tempAMEAN) * transpose(tempASTD[i]-tempAMEAN);
				}
				//tempASTDev.set( sqrt( (1.0f/(nbrsteps-1)) * tempASTDev.get(1,1) ), 1,1);
				Mat<float> tempASTDline( 1,nbraction);
				for(int l=1;l<=nbraction;l++)	tempASTDline.set( sqrt( (1.0f/(nbrsteps-1)) * tempASTDev.get(l,l) ), 1,l);
				testA_STD.push_back(tempASTDline);
				
				
				writeInFile(std::string("./TEST_MinReturn.txt"), testMINRETURN);
				writeInFile(std::string("./TEST_MeanReturn.txt"), testMEANRETURN);
				writeInFile(std::string("./TEST_TotalReturn.txt"), testTOTALRETURN);
				writeInFile(std::string("./TEST_QSA_MEAN.txt"), testQSA_MEAN);
				writeInFile(std::string("./TEST_QSA_STD.txt"), testQSA_STD);
				
				writeInFile(std::string("./TEST_A_MEAN.txt"), testA_MEAN);
				writeInFile(std::string("./TEST_A_STD.txt"), testA_STD);
				
				writeInFile(std::string("./TEST_Loss.txt"), testLOSS);
				
				
			}
			else
			{
				counterTest++;
			}
			
			
			std::cout << " EXPLORER :::: " ;
			std::cout << "EPISODE : " << i << " REWARD : " << this->totalReturn[this->totalReturn.size()-1].get(1,1) << std::endl;
			
			
			
			if( bankXP.size() > nbrEpisodeBankXPHolder*nbrReplay)
			{
				for(int t=this->nbrThread;t--;)	this->mXPrefresh[t]->lock();
				
				for(int t=nbrEpisodeBankXPHolder/2*nbrReplay;t--;)	this->bankXP.erase(this->bankXP.begin());
				
				for(int t=this->nbrThread;t--;)	this->mXPrefresh[t]->unlock();
				
				
				this->batchNormInit = false;
				this->batchNormalization();
			}
			
			writeInFile(std::string("./totalreturn.txt"), totalReturn);
			
			
			
			continuer = false;
			for(int t=nbrThread;t--;)
			{
				if( !threadsA3C[t].joinable())
				{
					continuer = true;
					break;
				}
			}
			
			if(continuer && i == nbrepisode)
			{
				nbrepisode++;
			}
		}		
		
		//let us print out the rewards of each exploration tests :
		for(int i=0;i<this->totalReturn.size();i++)
		{
			std::cout << " EXPLORER :::: " ;
			std::cout << "EPISODE : " << i << " REWARD : " << this->totalReturn[i].get(1,1) << std::endl;
		}
		
		for(int i=threadsA3C.size();i--;)
		{
			if(threadsA3C[i].joinable())
			{
				threadsA3C[i].join();
			}
		}
	}
	
	Mat<float> getTotalReturn(unsigned int idxEpisode)
	{
		if(idxEpisode < this->nbrepisode)
		{
			return this->totalReturn[idxEpisode];
		}
		
		return Mat<float>((float)0,1,1);
	}
	
	void updateTARGET()
	{
		bool retfa = this->targetFA->updateToward( this->fa, momentumUpdate);
		bool retpa = this->targetPA->updateToward( this->pa, momentumUpdate);
		
		std::cout << " UPDATE THE TARGETS : momentum : " << momentumUpdate << " ; results fa pa : " << retfa << " " << retpa << std::endl;
	}
	
	void batchNormalization()
	{
		if(!this->batchNormInit)
		{
			this->batchNormInit = true;
			int nbrxp = bankXP.size();
			float inbrxp = 1.0f/((float)(nbrxp)+numeric_limits<float>::epsilon());
			Mat<float> Sm( 0.0f*bankXP[0].S);
			Mat<float> Am( 0.0f*bankXP[0].A);
		
		
			for(int i=nbrxp;i--;)
			{
				Sm += inbrxp*bankXP[i].S;
				Am += inbrxp*bankXP[i].A;
			}
		
			float inobias = 1.0f/((float)(nbrxp-1)+numeric_limits<float>::epsilon());
			Mat<float> Sv( 0.0f*Sm);
			Mat<float> Av( 0.0f*Am);
		
		
			for(int i=nbrxp;i--;)
			{
				Sv += inobias * ( Sm-bankXP[i].S) % ( Sm-bankXP[i].S) ;
				Av += inobias * ( Am-bankXP[i].A) % ( Am-bankXP[i].A) ;			
			}
		
			meanXP = XPActorCritic( Sm, Am);
			stdXP = XPActorCritic( sqrt(Sv), sqrt(Av));
		
			
			this->pa->setInputNormalization(meanXP.S,stdXP.S);
			this->targetPA->setInputNormalization(meanXP.S,stdXP.S);
			this->fa->setInputNormalization( operatorC(meanXP.S,meanXP.A) , operatorC(stdXP.S, stdXP.A) );
			this->targetFA->setInputNormalization( operatorC(meanXP.S,meanXP.A) , operatorC(stdXP.S, stdXP.A) );
			
			/*
			meanXP.S.afficher();
			meanXP.A.afficher();
			stdXP.S.afficher();
			stdXP.A.afficher();
			std::cout << inbrxp << " " << inobias << std::endl;
			throw;
			*/
		}
		
		
	}
	
	
	void runThread()
	{
		mInit.lock();
		this->nbrThread++;
		int idxThread = this->nbrThread;
		this->NNupdateRequirement.push_back(false);
		this->mXPrefresh.push_back( new std::mutex() );
		mInit.unlock();
		
		NNTrainer<float> trainerPA( ((QPANN<float>*)this->pa)->getNetPointer());
		NNTrainer<float> trainerFA( ((QFANN<float>*)this->fa)->getNetPointer());
		
		for(int iteration=0;iteration<=nbrIteration;iteration++)
		{
			if(bankXP.size()>2*nbrReplay)
			{
				for(int i=(nbrReplay<bankXP.size()?nbrReplay:bankXP.size());i--;)
				{
					//NN UPDATE :
					//TODO:estimate the need of a mutex or not ?
					if(this->NNupdateRequirement[idxThread-1])
					{
						trainerPA.updateToward( ((QPANN<float>*)this->pa)->getNetPointer(), 1.0f );
						trainerFA.updateToward( ((QFANN<float>*)this->fa)->getNetPointer(), 1.0f );
						this->NNupdateRequirement[idxThread-1] = false;
					}
					
					
					
					this->mXPrefresh[idxThread-1]->lock();
					
					this->mutexXP.lock();
					int idx = rand()%bankXP.size();
					Mat<float> S(this->bankXP[idx].S);
					Mat<float> A(this->bankXP[idx].A);
					Mat<float> S1(this->bankXP[idx].S1);
					Mat<float> A1(this->bankXP[idx].A1);
					Mat<float> R(this->bankXP[idx].R);
					this->mutexXP.unlock();
					
					//METHOD DDPG :
					/*
					Mat<float> bankAfromS1( this->pa->estimateAction(S1) ); 
					Mat<float> bankQs1a1(  this->fa->getQvalue(bankXP[idx].S1, bankAfromS1) );
					Mat<float> bankQsa( this->fa->getQvalue(bankXP[idx].S,bankXP[idx].A) );
					this->fa->updateDelta(operatorC(bankXP[idx].S,bankXP[idx].A), bankXP[idx].R+gamma*bankQs1a1-bankQsa );
					*/
				
					//METHOD DDPG or DSPG: fixed target :
					//Mat<float> bankAfromS1TARGET( this->targetPA->estimateAction(bankXP[idx].S1) ); 
					Mat<float> bankAfromS1TARGET( this->targetPA->estimateAction(S1) ); 
					//Mat<float> bankQs1a1TARGET(  this->targetFA->getQvalue(bankXP[idx].S1, bankAfromS1TARGET) );
					Mat<float> bankQs1a1TARGET(  this->targetFA->getQvalue(S1, bankAfromS1TARGET) );
					//Mat<float> bankQsa( this->fa->getQvalue(bankXP[idx].S,bankXP[idx].A) );
					Mat<float> bankQsa( this->fa->getQvalue(S,A) );
					//Mat<float> bankerror(bankXP[idx].R+gamma*bankQs1a1TARGET-bankQsa);
					Mat<float> bankerror(R+gamma*bankQs1a1TARGET-bankQsa);
					//this->fa->updateDeltaBATCH(operatorC(bankXP[idx].S,bankXP[idx].A), (-1.0f)*bankerror, batchSize );
					//trainerFA.accumulateGradient( operatorC(bankXP[idx].S,bankXP[idx].A), (-1.0f)*bankerror);
					trainerFA.accumulateGradient( operatorC(S,A), (-2.0f)*bankerror);
				
				
					//POLICY GRADIENT : METHOD 4: TD 0 ACtor Critic
					/*
					Mat<float> bankdelta( bankXP[idx].R+gamma*bankQs1a1 - bankQsa);				
					Mat<float> bankdlogpa( inverseM(bankXP[idx].A) );
					this->pa->updateDelta(bankXP[idx].S, bankdlogpa*bankdelta);
					*/
					
					//POLICY GRADIENT : METHOD DDPG : fixed target : idem for the policy...
					/**/
					//this->fa->getQvalue( bankXP[idx].S, this->pa->estimateAction(bankXP[idx].S) ); 
					//Mat<float> input( operatorC( bankXP[idx].S, this->pa->estimateAction(bankXP[idx].S) ) );
					Mat<float> input( operatorC(S, this->pa->estimateAction(S) ) );
					Mat<float> bankdQda( this->fa->getNetPointer()->getGradientWRTinput( &input ) );
					//Mat<float> banksatoa( 0.0f, bankdQda.getColumn(), bankXP[idx].A.getColumn() );
					Mat<float> banksatoa( 0.0f, bankdQda.getColumn(),A.getLine() );
					//for(int k=1;k<=bankXP[idx].A.getColumn();k++)	banksatoa.set( 1.0f, bankXP[idx].S.getLine()+k, k);
					for(int k=1;k<=A.getLine();k++)	banksatoa.set( 1.0f, S.getLine()+k, k);
					
					
					bankdQda = bankdQda * banksatoa;
					
					
					float normedqda = sqrt( norme2(bankdQda) );
					//if(i==0)	std::cout << " DELTA Q da : " << bankdQda.get(1,1) << std::endl;
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (-1.0f/nbrReplay) * bankdQda, batchSize);
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (1.0f/nbrReplay) * bankdQda, batchSize);
					//trainerPA.accumulateGradient(bankXP[idx].S, (-1.0f/nbrReplay) * bankdQda);
					//trainerPA.accumulateGradient(S, (-1.0f/nbrReplay) * bankdQda);
					trainerPA.accumulateGradient(S, (-2.0f/normedqda) * bankdQda);
					
					float normeAcc = norme2( (-2.0f/normedqda) * bankdQda );
					
					/*
					//DEBUG :
					std::cout << " NORME grad : " << normedqda << std::endl;
					std::cout << " NORME ACCUMULATION : " << normeAcc << std::endl;
					*/
					
					/**/
				
					//POLICY GRADIENT : METHOD 6: TD 0 ACtor Critic : DSPG+fixed target
					/*
					Mat<float> bankdelta( bankXP[idx].R+gamma*bankQs1a1TARGET - bankQsa);				
					Mat<float> bankdlogpa( 2.0f,1,1 );
					//inverse of probability of 0.5;
					//TODO : this is not stochastic : not a probability...
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (-1.0f/nbrReplay)*(bankdlogpa*bankdelta), batchSize);
					trainerPA.accumulateGradient(bankXP[idx].S, (-1.0f/nbrReplay)*(bankdlogpa*bankdelta));
					/**/
					
					
					this->mXPrefresh[idxThread-1]->unlock();
				}


				//UPDATE OF THE NETWORKS now that a minibatch has been done :
				this->mPAupdate.lock();
				//std::cout << " THREAD : " << idxThread << " :::: " << " PA : " ;
				//trainerPA(GCSGD);
				trainerPA(GCSGDMomentum);
				this->mPAupdate.unlock();
			
				this->mFAupdate.lock();
				//std::cout << " THREAD : " << idxThread << " :::: " << " FA : " ;
				//trainerFA(GCSGD);
				trainerFA(GCSGDMomentum);
				
				for(int t=this->nbrThread;t--;)
				{
					if( t != idxThread)
					{
						this->NNupdateRequirement[t] = true;
					}
				}
				
				this->mFAupdate.unlock();
			
			
				/*
				//SOFT UPDATE TARGET TO REAL NETWORK :
				if( iteration % freqUpdate == 0)
				{
					this->mSOFTupdate.lock();
					std::cout << " THREAD : " << idxThread << " :::: SOFT UPDATE : " ;
					this->updateTARGET();
					this->mSOFTupdate.unlock();
				}
				*/
				
				//SOFT UPDATE TARGET TO REAL NETWORK :
				if( this->counterUpdate >= this->freqUpdate )
				{
					//this->mSOFTupdate.lock();
					this->counterUpdate = 0;
					std::cout << " THREAD : " << idxThread << " :::: SOFT UPDATE : " ;
					this->updateTARGET();
					//this->mSOFTupdate.unlock();
				}
				else
				{
					//this->mSOFTupdate.lock();
					this->counterUpdate++;
					//this->mSOFTupdate.unlock();
				}
							
			}
			else
			{
				iteration--;
			}

		}
		
	} 
	
	
	
	void runThreadEpisode()
	{
		mInit.lock();
		this->nbrThread++;
		int idxThread = this->nbrThread;
		this->NNupdateRequirement.push_back(false);
		this->mXPrefresh.push_back( new std::mutex() );
		mInit.unlock();
		
		NNTrainer<float> trainerPA( ((QPANN<float>*)this->pa)->getNetPointer());
		NNTrainer<float> trainerFA( ((QFANN<float>*)this->fa)->getNetPointer());
		
		//CARTPOLE :
		SimulatorRKCARTPOLE envt(5.0f);
		
		//CARTPOLEDISCRETE :
		//SimulatorRKCARTPOLE_Discrete envt(5.0f);
		
		for(int i=0;i<=nbrIteration;i++)
		{
			envt.initialize(false);	//random initial state or predefined...
			
		    unsigned int iteration = 0;
			while( !envt.isTerminal() )
			{
				clock_t time = clock();
		
				Mat<float> S(envt.getCurrentState() );
				//Mat<float> A(this->pa->estimateAction(S));
				Mat<float> A(this->pa->eps_greedy(S));
				Mat<float> R(envt.executeAction(A));
				Mat<float> S1(envt.getCurrentState());
			
				//NN UPDATE :
				//TODO:estimate the need of a mutex or not ?
				if(this->NNupdateRequirement[idxThread-1])
				{
					trainerPA.updateToward( ((QPANN<float>*)this->pa)->getNetPointer(), 1.0f );
					trainerFA.updateToward( ((QFANN<float>*)this->fa)->getNetPointer(), 1.0f );
					this->NNupdateRequirement[idxThread-1] = false;
				}
					
										
					//METHOD DDPG :
					/*
					Mat<float> bankAfromS1( this->pa->estimateAction(S1) ); 
					Mat<float> bankQs1a1(  this->fa->getQvalue(bankXP[idx].S1, bankAfromS1) );
					Mat<float> bankQsa( this->fa->getQvalue(bankXP[idx].S,bankXP[idx].A) );
					this->fa->updateDelta(operatorC(bankXP[idx].S,bankXP[idx].A), bankXP[idx].R+gamma*bankQs1a1-bankQsa );
					*/
				
					//METHOD DDPG or DSPG: fixed target :
					Mat<float> AfromS1TARGET( this->targetPA->estimateAction(S1) ); 
					Mat<float> Qs1a1TARGET(  this->targetFA->getQvalue(S1, AfromS1TARGET) );
					Mat<float> Qsa( this->fa->getQvalue(S,A) );
					Mat<float> error(R+gamma*Qs1a1TARGET-Qsa);
					//this->fa->updateDeltaBATCH(operatorC(bankXP[idx].S,bankXP[idx].A), (-1.0f)*bankerror, batchSize );
					trainerFA.accumulateGradient( operatorC(S,A), (-1.0f)*error);
				
				
					//POLICY GRADIENT : METHOD 4: TD 0 ACtor Critic
					/*
					Mat<float> bankdelta( bankXP[idx].R+gamma*bankQs1a1 - bankQsa);				
					Mat<float> bankdlogpa( inverseM(bankXP[idx].A) );
					this->pa->updateDelta(bankXP[idx].S, bankdlogpa*bankdelta);
					*/
					
					//POLICY GRADIENT : METHOD DDPG : fixed target : idem for the policy...
					/**/
					//this->fa->getQvalue( bankXP[idx].S, this->pa->estimateAction(bankXP[idx].S) ); 
					Mat<float> input( operatorC( S, this->pa->estimateAction(S) ) );
					Mat<float> dQda( this->fa->getNetPointer()->getGradientWRTinput( &input ) );
					Mat<float> satoa( 0.0f, dQda.getColumn(), A.getLine() );
					for(int k=1;k<=A.getLine();k++)	satoa.set( 1.0f, S.getLine()+k, k);
					dQda = dQda * satoa;
					//if(i==0)	std::cout << " DELTA Q da : " << bankdQda.get(1,1) << std::endl;
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (-1.0f/nbrReplay) * bankdQda, batchSize);
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (1.0f/nbrReplay) * bankdQda, batchSize);
					trainerPA.accumulateGradient( S, dQda);
					/**/
				
					//POLICY GRADIENT : METHOD 6: TD 0 ACtor Critic : DSPG+fixed target
					/*
					Mat<float> delta(R+gamma*Qs1a1TARGET - Qsa);				
					Mat<float> logpa( 2.0f,1,1 );
					//inverse of probability of 0.5
					//TODO : this is not stochastic : not a probability...
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (-1.0f/nbrReplay)*(bankdlogpa*bankdelta), batchSize);
					trainerPA.accumulateGradient(S, (-1.0f/freqUpdate)*(logpa*delta));
					/**/



				//UPDATE of the real network :
				this->mPAupdate.lock();
				//std::cout << " THREAD : " << idxThread << " :::: " << " PA : " ;
				trainerPA(GCSGD);
				this->mPAupdate.unlock();
			
				this->mFAupdate.lock();
				//std::cout << " THREAD : " << idxThread << " :::: " << " FA : " ;
				trainerFA(GCSGD);
				
				for(int t=this->nbrThread;t--;)
				{
					if( t != idxThread)
					{
						this->NNupdateRequirement[t] = true;
					}
				}
				
				this->mFAupdate.unlock();
			
			
				//SOFT UPDATE TARGET TO REAL NETWORK :
				if( iteration % freqUpdate == 0)
				{
					this->mSOFTupdate.lock();
					std::cout << " THREAD : " << idxThread << " :::: SOFT UPDATE : " ;
					this->updateTARGET();
					this->mSOFTupdate.unlock();
				}
				iteration++;
			}					
		}
		
	}
	
	
	void runThreadEpisode2()
	{
		mInit.lock();
		this->nbrThread++;
		int idxThread = this->nbrThread;
		this->NNupdateRequirement.push_back(false);
		this->mXPrefresh.push_back( new std::mutex() );
		mInit.unlock();
		
		NNTrainer<float> trainerPA( ((QPANN<float>*)this->pa)->getNetPointer());
		NNTrainer<float> trainerFA( ((QFANN<float>*)this->fa)->getNetPointer());
		
		//CARTPOLE :
		SimulatorRKCARTPOLE envt(this->env->getEOE());
		
		//CARTPOLEDISCRETE :
		//SimulatorRKCARTPOLE_Discrete envt(5.0f);
		
		int counterBATCH = 0;
		int valBATCH = 3;
		
		for(int it=0;it<this->nbrIteration;it++)
		{
			envt.initialize(false);	//random initial state or predefined...
			
			std::vector<XPActorCritic> bankXP;
			Mat<float> Rt(1,1);

			while( !envt.isTerminal() )
			{
				clock_t time = clock();
		
				Mat<float> S(envt.getCurrentState() );
				//Mat<float> A(this->pa->estimateAction(S));
				Mat<float> A(this->pa->eps_greedy(S));
				Rt = envt.executeAction(A);
				Mat<float> S1(envt.getCurrentState());
				Mat<float> A1(this->pa->eps_greedy(S1));
				
				bankXP.push_back( XPActorCritic(S,A,S1,A1,Rt));
			}
			
			int nbrXP = bankXP.size();
			//if the state is terminal, we do not bootstrap.
			if( Rt.get(1,1) >= 0.45f )
			{
				 Rt *= 0.0f;
			}
			else
			{
				Rt = this->fa->getQvalue( bankXP[nbrXP-1].S, bankXP[nbrXP-1].A);
			}
			
			for(int i=bankXP.size();i--;)
			{
			
				//NN UPDATE :
				//TODO:estimate the need of a mutex or not ?
				if(this->NNupdateRequirement[idxThread-1])
				{
					trainerPA.updateToward( ((QPANN<float>*)this->pa)->getNetPointer(), 1.0f );
					trainerFA.updateToward( ((QFANN<float>*)this->fa)->getNetPointer(), 1.0f );
					this->NNupdateRequirement[idxThread-1] = false;
				}
					
										
					//METHOD DDPG :
					/*
					Mat<float> bankAfromS1( this->pa->estimateAction(S1) ); 
					Mat<float> bankQs1a1(  this->fa->getQvalue(bankXP[idx].S1, bankAfromS1) );
					Mat<float> bankQsa( this->fa->getQvalue(bankXP[idx].S,bankXP[idx].A) );
					this->fa->updateDelta(operatorC(bankXP[idx].S,bankXP[idx].A), bankXP[idx].R+gamma*bankQs1a1-bankQsa );
					*/
				
					//METHOD DDPG or DSPG: fixed target :
					Mat<float> AfromS1TARGET( this->targetPA->estimateAction(bankXP[i].S1) ); 
					Mat<float> Qs1a1TARGET(  this->targetFA->getQvalue(bankXP[i].S1, AfromS1TARGET) );
					Mat<float> Qsa( this->fa->getQvalue(bankXP[i].S,bankXP[i].A) );
					Rt = gamma*Rt+bankXP[i].R;
					//Mat<float> error(Rt+gamma*Qs1a1TARGET-Qsa);
					
					Mat<float> error(Rt-Qsa);
					
					//this->fa->updateDeltaBATCH(operatorC(bankXP[idx].S,bankXP[idx].A), (-1.0f)*bankerror, batchSize );
					#ifndef Vvalues
					trainerFA.accumulateGradient( operatorC(bankXP[i].S,bankXP[i].A), (-1.0f/bankXP.size())*error);
					#else
					trainerFA.accumulateGradient( bankXP[i].S, (-2.0f/bankXP.size())*error);
					#endif
				
				
					//POLICY GRADIENT : METHOD 4: TD 0 ACtor Critic
					/*
					Mat<float> bankdelta( bankXP[idx].R+gamma*bankQs1a1 - bankQsa);				
					Mat<float> bankdlogpa( inverseM(bankXP[idx].A) );
					this->pa->updateDelta(bankXP[idx].S, bankdlogpa*bankdelta);
					*/
					
					//POLICY GRADIENT : METHOD DDPG : fixed target : idem for the policy...
					/*
					//this->fa->getQvalue( bankXP[idx].S, this->pa->estimateAction(bankXP[idx].S) ); 
					Mat<float> input( operatorC( bankXP[i].S, this->pa->estimateAction(bankXP[i].S) ) );
					Mat<float> dQda( this->fa->getNetPointer()->getGradientWRTinput( &input ) );
					Mat<float> satoa( 0.0f, dQda.getColumn(), bankXP[i].A.getLine() );
					for(int k=1;k<=bankXP[i].A.getLine();k++)	satoa.set( 1.0f, bankXP[i].S.getLine()+k, k);
					dQda = dQda * satoa;
					if(isnanM(dQda))
					{
						throw;
					}
					//if(i==0)	std::cout << " DELTA Q da : " << bankdQda.get(1,1) << std::endl;
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (-1.0f/nbrReplay) * bankdQda, batchSize);
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (1.0f/nbrReplay) * bankdQda, batchSize);
					//trainerPA.accumulateGradient( bankXP[i].S, dQda);//EXPLODE!!!!
					trainerPA.accumulateGradient( bankXP[i].S, (-1.0f/bankXP.size())*dQda);
					/**/
				
					//POLICY GRADIENT : METHOD 6: TD 0 ACtor Critic : DSPG+fixed target
					/**/
					Mat<float> delta(error);				
					Mat<float> dlogpa( 1.0f/0.5f,1,1 );
					float normedlogpadelta = sqrt( norme2(dlogpa*delta) );
					//inverse of probability of 0.5
					//TODO : this is not stochastic : not a probability...
					//this->pa->updateDeltaBATCH(bankXP[idx].S, (-1.0f/nbrReplay)*(bankdlogpa*bankdelta), batchSize);
					//trainerPA.accumulateGradient(bankXP[i].S, (-1.0f/(freqUpdate*bankXP.size()))*(logpa*delta));
					trainerPA.accumulateGradient(bankXP[i].S, (-2.0f/(normedlogpadelta*bankXP.size()))*(dlogpa*delta));
					/**/
			}


			//UPDATE of the real network :
			if( counterBATCH > valBATCH)
			{
				counterBATCH=0;
				
				
				this->mPAupdate.lock();
				//std::cout << " THREAD : " << idxThread << " :::: " << " PA : " ;
				//trainerPA(GCSGD);
				trainerPA(GCSGDMomentum);
				this->mPAupdate.unlock();
		
				this->mFAupdate.lock();
				//std::cout << " THREAD : " << idxThread << " :::: " << " FA : " ;
				//trainerFA(GCSGD);
				trainerFA(GCSGDMomentum);
			
				for(int t=this->nbrThread;t--;)
				{
					if( t != idxThread)
					{
						this->NNupdateRequirement[t] = true;
					}
				}
			
				this->mFAupdate.unlock();
			}
			else
			{
				counterBATCH++;
			}
		
			//SOFT UPDATE TARGET TO REAL NETWORK :
			if( this->counterUpdate >= this->freqUpdate )
			{
				//this->mSOFTupdate.lock();
				this->counterUpdate = 0;
				std::cout << " THREAD : " << idxThread << " :::: SOFT UPDATE : " ;
				this->updateTARGET();
				//this->mSOFTupdate.unlock();
			}
			else
			{
				//this->mSOFTupdate.lock();
				this->counterUpdate++;
				//this->mSOFTupdate.unlock();
			}
								
		}
		
	} 
};

#endif
