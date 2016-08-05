#include "../Mat.h"
#include <iostream>

Mat<float> matmult(const Mat<float>& a, const Mat<float>& b)
{
	//Mat<float> aa(a);
	Mat<float> bb(transpose(b));
	int nbrL = a.getLine();
	int nbrC = bb.getLine();
	int nbrLb = bb.getColumn();
	//Mat<float> c(0.0f,nbrL,nbrC);
	Mat<float> c(nbrL,nbrC);
	
	for(size_t j=0;j<nbrC;j++)
	{
		for(size_t i=0;i<nbrL;i++)	c.mat[i][j] =  0.0f;
		
		for(size_t i=0;i<nbrL;i++)
		{
			float sum = 0;
			for( size_t k=0;k<nbrLb;k++)
			{
				sum+= a.mat[i][k] * bb.mat[j][k];
			}
			c.mat[i][j] += sum;
		}
	}
	
	return c;
}



Mat<float> matmult1(const Mat<float>& a, const Mat<float>& b)
{
	Mat<float> aa(transpose(a));
	//Mat<float> bb(b);
	int nbrL = aa.getColumn();
	int nbrC = b.getColumn();
	int nbrCa = aa.getLine();
	//Mat<float> c(0.0f,nbrC,nbrL);
	Mat<float> c(0.0f,nbrL,nbrC);
	
	for(size_t j=0;j<nbrC;j++)
	{
		for(size_t k=0;k<nbrCa;k++)
		{
			float bkj[nbrC];
			for( size_t i=0;i<nbrC;i++)
			{
				bkj[i] = b.mat[k][j];
			}
			for(size_t i=0;i<nbrC;i++)
			{
				//c.mat[j][i] += aa.mat[k][i] * bkj[i];
				c.mat[i][j] += aa.mat[k][i] * bkj[i];
			}
		}
	}
	
	//transpose(&c);
	
	return c;
}

static inline void matmult_kernel_sxr_c(Mat<float>& c, const Mat<float>& a, const Mat<float>& b, int r, int s)
{
	Mat<float> bb(transpose(b));
	
	for(size_t j=0;j<r;j++)
	{
		for(size_t i=0;i<r;i++)
		{
			float sum = 0.0f;
			for(size_t k=0;k<s;k++)
			{
				sum+= a.mat[i][k] * bb.mat[j][k];
			}
			c.mat[i][j] += sum;
		}
	}
}


static inline void set_block( Mat<float>& m, const Mat<float>& blockm, int ib, int jb, int sizei, int sizej)
{
	for(int i=ib;i<ib+sizei;i++)
	{
		for(int j=jb;j<jb+sizej;j++)
		{
			m.mat[i][j] = blockm.mat[i-ib][j-jb];
		}
	}
}

Mat<float> matmult2(const Mat<float>& a, const Mat<float>& b)
{
	int r= 4;
	int s = 4;
	int n = a.getLine();
	int m = b.getColumn();
	
	Mat<float> C(0.0f,n,m);
	
	for(size_t cj=0;cj<n/r;cj++)
	{
		for(size_t ci=0;ci<n/r;ci++)
		{
			//for each blockC in C
			//Mat<float> blockC( get_block(ci,cj,r,r,C,n) );
			Mat<float> blockC( extract(C,(ci*r+1),(cj*r+1),(ci*r+1)+r,(cj*r+1)+r) );
			
			//for each blockA and blockB :
			for(size_t is=0;is<n/s;is++)
			{
				Mat<float> blockA( extract(a,(ci*r+1),(is*s+1),(ci*r+1)+r,(is*s+1)+s) );
				Mat<float> blockB( extract(b,(is*s+1),(ci*r+1),(is*s+1)+s,(ci*r+1)+r) );
				
				matmult_kernel_sxr_c(blockC,blockA,blockB,r,s);
			}
			
			set_block(C,blockC,ci*r,cj*r, r, r);
		}
	}
	
	return C;
}



int main( int argc, char* argv[])
{
	int nbrLine = 100;
	int nbrColumn = 100;
	Mat<float> A(nbrLine,nbrLine,(char)1);
	Mat<float> B(nbrLine,nbrColumn,(char)1);
	
	clock_t time = clock();
	(A*B);
	
	std::cout << " THE MULT REGULAR TOOK : " << ((float)clock()-time)/((float)CLOCKS_PER_SEC) << " seconds." << std::endl;
	
	
	time = clock();
	matmult(A,B);
	
	std::cout << " THE MULT TOOK : " << ((float)clock()-time)/((float)CLOCKS_PER_SEC) << " seconds." << std::endl;
	
	
	time = clock();
	matmult1(A,B);
	
	std::cout << " THE MULT 1 TOOK : " << ((float)clock()-time)/((float)CLOCKS_PER_SEC) << " seconds." << std::endl;
	
	
	time = clock();
	matmult2(A,B);
	
	std::cout << " THE MULT 2 TOOK : " << ((float)clock()-time)/((float)CLOCKS_PER_SEC) << " seconds." << std::endl;
	
	return 0;
	
}
