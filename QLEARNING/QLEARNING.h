#ifndef QLEARNING_H
#define QLEARNING_H

#include "../FA/FA.h"
#include "../ENVIRONMENT/Environment.h"
#include "../RunningStats/RunningStats.h"


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
			
				this->fa->update(S,A,S1,R);
				
				
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
					
					this->fa->update( bankXP[idx].S, bankXP[idx].A, bankXP[idx].S1, bankXP[idx].R );
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


#endif