
# include <stdio.h>
# include <math.h>
# include <iostream>
# include <random>


using namespace std;

mt19937 gen(time(0));
double m = 0.0;
double s = 1.0;
auto myNormal = normal_distribution<double>(m,s);
double normalDouble(){
  return double(myNormal(gen));
}

double normCDF(double x){
  return 0.5*(1+erf(x/sqrt(2.0)));
}

struct szhw{
  double kap, lam, gam, p, theta, sbar, eta;
  double x0[3];
  double corrStructure[6];
};

double schoebelZhuHullWhite0(szhw imod,double targetVar,double K){
  /*
  Give the zeroth order price of a timer option.
  Use equation 2.2 of http://arxiv.org/abs/0711.1272
  */
  double mu,sig,rv;
  double T=targetVar/imod.x0[2];
  mu = imod.x0[0]/(1-imod.x0[1]*T);
  sig = imod.x0[0]*pow(imod.x0[2],imod.p)*sqrt(T);
  rv = (mu-K)*normCDF((mu-K)/(sig));
  rv += sig*exp(-0.5*(mu-K)*(mu-K)/sig/sig)/sqrt(2.0*3.1415926535897932384626433);
  return rv;
};

void schoebelZhuHullWhiteEllStep(double dt,szhw imod,double *x,double *y,double *xa,double *ya,double *dW){
  //std :: cout << "Components " << x[0] << " " << x[1]  << " " << x[2]  << " " <<  xa[0]  << " " <<  xa[1]  << " " <<  xa[2] << "\n";
  //std :: cout << "dW " << dW[0] << " " << dW[1]  << " " << dW[2]<< "\n";
  y[2] = (x[2]+imod.kap*imod.sbar*dt)/(1+imod.kap*dt);
  y[1] = (x[1]+imod.lam*imod.theta*dt)/(1+imod.lam*dt);
  y[0] = x[0]/(1-x[1]*dt);
  ya[2] = (xa[2]+imod.kap*imod.sbar*dt)/(1+imod.kap*dt);
  ya[1] = (xa[1]+imod.lam*imod.theta*dt)/(1+imod.lam*dt);
  ya[0] = xa[0]/(1-xa[1]*dt);
  y[0] += pow(x[2],imod.p)*x[0]*dW[0];
  y[1] += imod.eta*dW[1];
  y[2] += imod.gam*pow(x[2],imod.p-1)*dW[2];
  ya[0] -= pow(xa[2],imod.p)*xa[0]*dW[0];
  ya[1] -= imod.eta*dW[1];
  ya[2] -= imod.gam*pow(xa[2],imod.p-1)*dW[2];
  for(unsigned int j=0;j<3;j++){
    x[j] = y[j];
    xa[j] = ya[j];
  }
  //std :: cout << "Components " << x[0] << " " << x[1]  << " " << x[2]  << " " <<  xa[0]  << " " <<  xa[1]  << " " <<  xa[2] << "\n";
}

void schoebelZhuHullWhiteUpdateCVar(double dt,double * x,double * cVar){
  cVar[0] += x[2]*x[2]*dt;
}

void schoebelZhuHullWhiteUpdateDiscount(double dt, double * x, double * disc){
  disc[0] += dt*x[1];
}

void schoebelZhuHullWhiteUpdateEvaluate(double * x, double * disc, double *cVar, double* targetVar, double *g,double *K){
  if ((g[0]<0.0)&&(cVar[0]>=targetVar[0])) g[0]= exp(-1*disc[0])*((x[0]-K[0]>0.0)?x[0]-K[0]:0.0);
}


double schoebelZhuHullWhiteEll(double dt,szhw imod,double targetVar,double K){
  
  unsigned int i;

  double t=0.0;
  
  double dW1[3];
  double dW2[3];
  double x1[3];
  double x2[3];
  double y1[3];
  double y2[3];

  // antithetic variables
  double x1a[3];
  double x2a[3];
  double y1a[3];
  double y2a[3];

  double cVar[4];
  double discount[4];
  double g[4];
  double sqrtDt;
  
  double retVal;
  
  /* Initial value */
  for(i=0;i<3;i++){
    x1[i] = imod.x0[i];
    x2[i] = imod.x0[i];
    x1a[i] = imod.x0[i];
    x2a[i] = imod.x0[i];
  }
  
  /* Initialisation */
  
  for(i=0;i<4;i++){
    discount[i] = 0.0;
    g[i] = -1.0;
    cVar[i] = 0.0;
  }

  sqrtDt = sqrt(dt);
  
  //std :: cout << "square root dt " << sqrtDt << "\n";
  
  while(((g[0]<0.0)||(g[1]<0.0))||((g[2]<0.0)||(g[3]<0.0))){
    //std :: cout << "t=" << t << "\n";
    //std :: cout << "gs " << g[0] << " " <<  g[1] << " " << g[2] << " " << g[3] << "\n";
    //std :: cout << "cumVar " << cVar[0] << " " << cVar[2] << " (" << targetVar << ")\n";
    // Generate the random variables
    for(i=0;i<3;i++) dW1[i] = sqrtDt*normalDouble();
    //dW1[0] = imod.corrStructure[0]*dW1[0]+imod.corrStructure[1]*dW1[1]+imod.corrStructure[2]*dW1[2];
    //dW1[1] = imod.corrStructure[3]*dW1[1]+imod.corrStructure[4]*dW1[2];
    //dW1[2] = imod.corrStructure[5]*dW1[2];
    //std :: cout << "The dW components 1 " << dW1[0] << " " << dW1[1]  << " " << dW1[2]  << "\n";
    // Store random increments for simulation of the coarse process
    for(i=0;i<3;i++) dW2[i]=dW1[i];
    // Update cumulative variance and discount
    schoebelZhuHullWhiteUpdateCVar(dt,x1,cVar);
    schoebelZhuHullWhiteUpdateCVar(dt,x1a,cVar+2);
    schoebelZhuHullWhiteUpdateDiscount(dt,x1,discount);
    schoebelZhuHullWhiteUpdateDiscount(dt,x1a,discount+2);
    schoebelZhuHullWhiteEllStep(dt,imod,x1,y1,x1a,y1a,dW1);
    // Check if volatility target is reached
    schoebelZhuHullWhiteUpdateEvaluate(x1,discount,cVar,&targetVar,g,&K);
    schoebelZhuHullWhiteUpdateEvaluate(x1a,discount+2,cVar+2,&targetVar,g+2,&K);
    // Draw new random variables
    for(i=0;i<3;i++) dW1[i] = sqrtDt*normalDouble();
    //dW1[0] = imod.corrStructure[0]*dW1[0]+imod.corrStructure[1]*dW1[1]+imod.corrStructure[2]*dW1[2];
    //dW1[1] = imod.corrStructure[3]*dW1[1]+imod.corrStructure[4]*dW1[2];
    //W1[2] = imod.corrStructure[5]*dW1[2];
    //std :: cout << "The dW components 2 " << dW1[0] << " " << dW1[1]  << " " << dW1[2]  << "\n";
    // Increment the random increments for the coarse process
    for(i=0;i<3;i++) dW2[i]+=dW1[i];
    // Update compounded variance and discount factor
    schoebelZhuHullWhiteUpdateCVar(dt,x1,cVar);
    schoebelZhuHullWhiteUpdateCVar(dt,x1a,cVar+2);
    schoebelZhuHullWhiteUpdateDiscount(dt,x1,discount);
    schoebelZhuHullWhiteUpdateDiscount(dt,x1a,discount+2);
    schoebelZhuHullWhiteEllStep(dt,imod,x1,y1,x1a,y1a,dW1);
    //std :: cout << "Components " << x1[0] << " " << x1[1]  << " " << x1[2]  << " " <<  x1a[0]  << " " <<  x1a[1]  << " " <<  x1a[2] << "\n";
    //std :: cout << "cumVar " << cVar[0] << " " << cVar[2] << " (" << targetVar << ")\n";
    
    // Re-check if volatility budget is full
    schoebelZhuHullWhiteUpdateEvaluate(x1,discount,cVar,&targetVar,g,&K);
    schoebelZhuHullWhiteUpdateEvaluate(x1a,discount+2,cVar+2,&targetVar,g+2,&K);
    // Update the compounded variance and discount for the coarse process
    schoebelZhuHullWhiteUpdateCVar(dt*2,x2,cVar+1);
    schoebelZhuHullWhiteUpdateCVar(dt*2,x2a,cVar+3);
    schoebelZhuHullWhiteUpdateDiscount(dt*2,x2,discount+1);
    schoebelZhuHullWhiteUpdateDiscount(dt*2,x2a,discount+3);
    schoebelZhuHullWhiteEllStep(dt*2,imod,x2,y2,x2a,y2a,dW2);
    // Evaluate the payoff, if condition satisfied
    schoebelZhuHullWhiteUpdateEvaluate(x2,discount+1,cVar+1,&targetVar,g+1,&K);
    schoebelZhuHullWhiteUpdateEvaluate(x2a,discount+3,cVar+3,&targetVar,g+3,&K);
    t += 2*dt;
    //std :: cout << "cumVar2 " << cVar[0] << " " << cVar[2] << " (" << targetVar << ")\n";
    //std :: cout << "gs " << g[0] << " " << g[1] << " " << g[2] << " " << g[3] << "\n";

    //std :: cout << t << " " << x1[0] << " " << x1[1] << " " << x1[2] << " " << " " << cVar[0]<< " " << g[0]  << " " << x2[0] << " " << x2[1] << " " << x2[2]<< " " << g[1] << " " << cVar[1]<< " " << targetVar << "\n";
    //std :: cout << "Components " << x1[0] << " " << x1[1]  << " " << x1[2]  << " " <<  x1a[0]  << " " <<  x1a[1]  << " " <<  x1a[2] << "\n";
  }

  retVal = 0.5*(g[0]+g[2]-g[1]-g[3]);
  //retVal = g[0]-g[1];
  //retVal = 0.5*(g[0]+g[2]);
  return retVal;
}

double bsModel0(double S,double sigma,double r, double T,double K){
  szhw testModel;
  testModel.kap = 0.0;
  testModel.lam = 0.0;
  testModel.gam = 0.0;
  testModel.p = 1.0;
  testModel.theta = 0.0;
  testModel.sbar = 0.1;
  testModel.eta = 0.0;
  testModel.corrStructure[0] = 1.0;
  testModel.corrStructure[1] = 0.0;
  testModel.corrStructure[2] = 0.0;
  testModel.corrStructure[3] = 1.0;
  testModel.corrStructure[4] = 0.0;
  testModel.corrStructure[5] = 1.0;
  testModel.x0[0] = S;
  testModel.x0[1] = r;
  testModel.x0[2] = sigma;
  return schoebelZhuHullWhite0(testModel,T*sigma,K);
}

double bsModelDt(double dt, double S,double sigma, double r, double T, double K){
  szhw testModel;
  testModel.kap = 0.0;
  testModel.lam = 0.0;
  testModel.gam = 0.0;
  testModel.p = 1.0;
  testModel.theta = 0.0;
  testModel.sbar = 0.1;
  testModel.eta = 0.0;
  testModel.corrStructure[0] = 1.0;
  testModel.corrStructure[1] = 0.0;
  testModel.corrStructure[2] = 0.0;
  testModel.corrStructure[3] = 1.0;
  testModel.corrStructure[4] = 0.0;
  testModel.corrStructure[5] = 1.0;
  testModel.x0[0] = S;
  testModel.x0[1] = r;
  testModel.x0[2] = sigma;
  return schoebelZhuHullWhiteEll(dt,testModel,T*sigma*sigma,K);
}

double bsPrice(double S,double sigma, double r, double T, double K,double TOL){

  std :: cout << "Evaluating BS model S=" << S << " s=" << sigma << " r=" << r << " K=" << K << " T=" <<T<< " TOL=" << TOL <<"\n";

  szhw testModel;
  testModel.kap = 0.0;
  testModel.lam = 0.0;
  testModel.gam = 0.0;
  testModel.p = 1.0;
  testModel.theta = 0.0;
  testModel.sbar = 0.1;
  testModel.eta = 0.0;
  testModel.corrStructure[0] = 1.0;
  testModel.corrStructure[1] = 0.0;
  testModel.corrStructure[2] = 0.0;
  testModel.corrStructure[3] = 1.0;
  testModel.corrStructure[4] = 0.0;
  testModel.corrStructure[5] = 1.0;
  testModel.x0[0] = S;
  testModel.x0[1] = r;
  testModel.x0[2] = sigma;
  
  double bias = 100*TOL;
  double maxBias = TOL/2;
  double varErr = 100*TOL;
  varErr *= varErr;
  double maxVarErr = TOL-maxBias;
  maxVarErr *= maxVarErr;
  double levelVar;

  unsigned int M[100];
  double lV[100];
  double lmu[100];
  unsigned int minM=20;
  unsigned int i,j,k;
  unsigned int L = 1;
  double mu,var,temp;
  for(i=0;i<100;i++) M[i] = 0.0;
  for(i=0;i<100;i++) lV[i] = 0.0;
  
  M[0] = minM;
  mu = 0.0;
  var = 0.0;
  for(j=0;j<M[0];j++){
    temp = schoebelZhuHullWhiteEll(T/2,testModel,testModel.x0[2]*T,K);
    mu += temp;
    var += temp*temp;
  }
  mu /= M[0];
  var /= M[0];
  var -= mu*mu;
  lV[0] = var/M[0];
  bias = 2*TOL;

  while(abs(bias)+sqrt(varErr)>TOL){
    // Improve bias
    if(bias>maxBias){
      if (sqrt(lV[L-1])>0.5*bias){
	// We are in the case where bulk of the bias comes from
	// Statistical uncertainty in the last level
	std :: cout << "Bias is " << bias << " adding another level. L=" << L << "\n";
	// Add another level
	mu = 0.0;
	var = 0.0;
	for(j=0;j<minM;j++){
	  temp = schoebelZhuHullWhiteEll(T/pow(2.0,L+1),testModel,testModel.x0[2]*T,K);
	  mu += temp;
	  var+= temp*temp;
	  //std :: cout << "sample " << temp << "\n";
	}
	mu /=minM;
	var /= minM;
	var -= mu*mu;
	lV[L] = var/minM;
	lmu[L] = mu;
	M[L] = minM;
	//std :: cout << "debug print, mu of new level " << mu << " std of new level " << sqrt(var) << "\n";
	bias = abs(lmu[L])+sqrt(lV[L]);
	L++;
	varErr = 0.0;
	for(k=0;k<L;k++) varErr += lV[k]*lV[k];
      }
      else{
	std :: cout << "Reducing variance on the last level L=" << L<< " M[L]=" << M[L-1] << "\n";
	// Enlarge the last level sample size
        mu = 0;
        var = 0.0;
	for(j=0;j<M[L-1];j++){
          temp = schoebelZhuHullWhiteEll(T/pow(2.0,L),testModel,testModel.x0[2]*T,K);
          mu += temp;
          var+= temp*temp;
        }
	mu /= M[L-1];
	mu /=M[L-1];
        var /=M[L-1];
        var -= mu*mu;
	var /= M[L-1];
        lV[L-1] = 0.25*(lV[L-1]+var);
        lmu[L-1] = 0.5*(mu+lmu[L-1]);
	bias = abs(lmu[L-1])+sqrt(lV[L-1]);
        M[L-1] *= 2;
	varErr = 0.0;
	for(k=0;k<L;k++) varErr += lV[k]*lV[k];
      } 
    }
    // Improve variance
    //std :: cout << "debugging bias " << bias << "\n";
    if(varErr>maxVarErr){
      levelVar = maxVarErr/L;
      for(k=0;k<L;k++){
	if(lV[k]>levelVar){
	  std :: cout << "Reducing variance on level k=" << k<< " M[k]=" << M[k] << "\n";
	  mu = 0;
	  var = 0.0;
	  for(j=0;j<M[k];j++){
	    temp = schoebelZhuHullWhiteEll(T/pow(2.0,L),testModel,testModel.x0[2]*T,K);
	    mu += temp;
	    var+= temp*temp;
	  }
	  mu /= M[k];
	  mu /=M[k];
	  var /=M[k];
	  var -= mu*mu;
	  var /= M[k];
	  lV[k] = 0.25*(lV[k]+var);
	  lmu[k] = 0.5*(mu+lmu[k]);
	  M[k] *= 2;
	}
      }
      varErr = 0.0;
      for(k=0;k<L;k++) varErr += lV[k]*lV[k];
      std :: cout << "Variance error " << varErr << " bound " << maxVarErr << "\n";
    }

    std :: cout  << "Iteration ended L = " << L << " bias = " << bias << " , stat err. =  " << sqrt(varErr) << " TOL = " << TOL << "\n";
    
  }
  mu = 0.0;
  for(k=0;k<L;k++) mu += lmu[k];
  temp = schoebelZhuHullWhite0(testModel,sigma*T,K);
  std :: cout << "Zeroth order expectation " << temp << "\n";
  mu += temp;
  return mu;
}

extern "C" {
  double bsDt(double dt, double S,double sigma, double r, double T, double K){
    return bsModelDt(dt,S,sigma,r,T,K);
  };
  double bsNoStep(double S,double sigma, double r, double T, double K){
    return bsModel0(S,sigma,r,T,K);
  };
  double randn(){ return normalDouble();}
}

//int main(){
//  double price = bsPrice(100.0,0.2,0.00,0.25,100.0,0.005);
//  std :: cout << "BS price " << price << "\n";
//  return 0;
//}
