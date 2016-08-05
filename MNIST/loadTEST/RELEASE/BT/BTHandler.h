#ifndef BTHANDLER_H
#define BTHANDLER_H

//#include <stdio.h>
#include <iostream>

#include <unistd.h>
#include <sys/socket.h>
#include <bluetooth/bluetooth.h>
#include <bluetooth/rfcomm.h>

#include <vector>
#include <string>
#include <map>

typedef struct sockaddr_rc SOCKADDR;


class BTHandler
{
	public :
	
	BTHandler() : IDXCOM(1)
	{
	
	}
	
	~BTHandler()
	{
	
	}
	
	int Open( const std::string& dest, std::string* first_message  = NULL)
	{
		this->idxSOCKETS.push_back( socket(AF_BLUETOOTH,SOCK_STREAM,BTPROTO_RFCOMM) );
		
		this->socketADDR.push_back( SOCKADDR() );
		int nbrSOCKADDR = this->socketADDR.size();
		this->socketADDR[nbrSOCKADDR-1].rc_family = AF_BLUETOOTH;
		this->socketADDR[nbrSOCKADDR-1].rc_channel = (uint8_t)1;
		
		str2ba( dest.c_str(),&( (this->socketADDR[nbrSOCKADDR-1]).rc_bdaddr) ); 
		
		this->status.push_back( connect( this->idxSOCKETS[nbrSOCKADDR-1], (struct sockaddr*)( & (this->socketADDR[nbrSOCKADDR-1]) ), sizeof(this->socketADDR[nbrSOCKADDR-1]) ) );
		
		if( this->status[nbrSOCKADDR-1] == 0)
		{
			std::cout << "\t\t BTHandler :: opening : OKAY." << std::endl;
			
			this->dest2idx[dest] = this->IDXCOM++;
			
			
			
			
			if(first_message != NULL)
			{
				status[nbrSOCKADDR-1] = write( this->idxSOCKETS[nbrSOCKADDR-1], first_message->c_str(), first_message->length() );
				std::cout << "\t\tBTHandler : first message sent : RESULT OF THE OPERATION : " << status[nbrSOCKADDR-1] << std::endl;
			}
			
			return 0;
		}
		else
		{
			std::cout << "\t\tBTHandler :: opening status : " << status[nbrSOCKADDR-1] << std::endl;
			return -1;
		}
		
	}
	
	void closeAll()
	{
		for(int i=this->idxSOCKETS.size();i--;)
		{
			close( this->idxSOCKETS[i] );
		}
	}
	
	void Send(const std::string& dest, const std::string& message)
	{
		this->messages[dest].push_back(message);
		
		if(this->dest2idx[dest] > 0)
		{
			this->status[this->dest2idx[dest]-1] = write( this->idxSOCKETS[this->dest2idx[dest]-1], message.c_str(), message.length() );			
			if(	this->status[this->dest2idx[dest]-1] > 0)
			{
				std::cout << "\t\tBTHandler : message sent : RESULT OF THE OPERATION : " << this->status[this->dest2idx[dest]-1] << std::endl;
			}
			else
			{
				std::cout << "\t\tBTHandler : message sent : RESULT OF THE OPERATION : " << this->status[this->dest2idx[dest]-1] << std::endl;
			}
		}
		else
		{
			//it has to be open : 
			std::string dum(message);
			this->Open(dest,&dum);
		}
	}
	
	private :
	
	std::map<std::string, std::vector<std::string> > messages;
	
	
	std::vector<std::string> destinationsADDR;
	std::vector<SOCKADDR> socketADDR;
	
	std::vector<int> status;
	std::vector<int> idxSOCKETS;
	
	int IDXCOM;
	std::map<std::string, int> dest2idx;
};

#endif


