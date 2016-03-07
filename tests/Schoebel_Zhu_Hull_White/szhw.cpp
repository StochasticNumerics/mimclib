
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <random>
#include <quadmath.h>

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

double randTest(unsigned int Nt,double dt){
  double x1,x2,x1a,x2a;
  double t = 0.0;
  double sig = 0.2;
  x1=100.0;
  x2=x1;
  x1a=x1;
  x2a=x1;
  double r = 0.05;
  double K = x1;
  double temp,temp2;
  for(unsigned int i=0;i<Nt;i++){
    t += 2*dt;
    temp = sqrt(dt)*normalDouble();
    x1  += x1 *r*dt+x1*sig*temp;
    x1a += x1a*r*dt-x1a*sig*temp;
    temp2 = temp;
    temp = sqrt(dt)*normalDouble();
    x1 += x1*r*dt+x1*sig*temp;
    x1a += x1a*r*dt-x1a*sig*temp;
    temp2 += temp;
    x2 += x2*r*dt*2+x2*sig*temp2;
    x2a += x2a*r*dt*2-x2a*sig*temp2;
  }
  //std :: cout << "tt " << t << "\n";

  //return 0.5*(((x1-K>0.0)?x1-K:0.0 ) - ((x2-K>0.0)?x2-K:0.0 ) + ((x1a-K>0.0)?x1a-K:0.0 ) - ((x2a-K>0.0)?x2a-K:0.0 ));
  //return ((x1-K>0.0)?x1-K:0.0 ) ;
  //return ((x1a-K>0.0)?x1a-K:0.0 ) - ((x2a-K>0.0)?x2a-K:0.0 );
  //return ((x1-K>0.0)?x1-K:0.0 ) - ((x2-K>0.0)?x2-K:0.0 );
  //return x1-x2;
  return exp(-1*r)*0.5*((x1a>K?x1a-K:0.0)+(x1>K?x1-K:0.0));
  return exp(-1*r)*0.5*(x1a+x1);
}

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
  rv *= exp(-1*T*imod.x0[1]);
  return rv;
};

void schoebelZhuHullWhiteEllStep(double dt,szhw imod,double *x,double *y,double *xa,double *ya,double *dW){
  y[2] = (x[2]+imod.kap*imod.sbar*dt)/(1+imod.kap*dt);
  y[1] = (x[1]+imod.lam*imod.theta*dt)/(1+imod.lam*dt);
  y[0] = x[0]/(1-x[1]*dt);
  //y[0] = x[0] + x[1]*x[0]*dt;
  //std :: cout << "y x " << y[0] << " " << x[0] << "\n";
  ya[2] = (xa[2]+imod.kap*imod.sbar*dt)/(1+imod.kap*dt);
  ya[1] = (xa[1]+imod.lam*imod.theta*dt)/(1+imod.lam*dt);
  ya[0] = xa[0]/(1-xa[1]*dt);
  y[0] += pow(x[2],imod.p)*x[0]*dW[0];
  y[1] += imod.eta*dW[1];
  y[2] += imod.gam*pow(x[2],imod.p-1)*dW[2];
  ya[0] -= pow(xa[2],imod.p)*xa[0]*dW[0];
  ya[1] -= imod.eta*dW[1];
  ya[2] -= imod.gam*pow(xa[2],imod.p-1)*dW[2];
  
  //std :: cout << "y x " << y[0] << " " << x[0] << "\n";  
  for(unsigned int j=0;j<3;j++){
    x[j] = y[j];
    xa[j] = ya[j];
  }
  //std :: cout << "Components " << x[0] << " " << x[1]  << " " << x[2]  << " " <<  xa[0]  << " " <<  xa[1]  << " " <<  xa[2] << "\n";
}

void schoebelZhuHullWhiteUpdateCVar(double dt,double * x,double * cVar){
  cVar[0] += x[2]*x[2]*dt;
}

void schoebelZhuHullWhiteUpdateDiscount(double dt,double * x,double * disc){
  disc[0] += dt*x[1];
}

void schoebelZhuHullWhiteUpdateEvaluate(double * x, double * disc, double *cVar, double * targetVar, double *g,double *K,double * dt){
  if (g[0] < 0.0){
    if ((cVar[0]>=targetVar[0])||abs(targetVar[0]-cVar[0])<=0.5*x[2]*x[2]*dt[0] ) {
      g[0] = ((x[0]-K[0])>0.0) ? (x[0]-K[0])  : 0.0 ;
      //std :: cout << "exercise x " << x[0] << "\n";
    }
    g[0] *= exp(-1*disc[0]);
  }
}

double schoebelZhuHullWhiteEll(double dt,szhw imod,double targetVar,double K){
  
  //std :: cout << "targetVar vittu " << targetVar << "\n";

  unsigned int i;

  double t=0.0,tp;
  
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
  double cVarp[4];
  double discount[4];
  double g[4];
  double sqrtDt;
  
  double retVal;

  //std :: cout << "sigma " << imod.x0[2] << "\n";
  
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
  
  //std :: cout << "square root dt " << sqrtDt << " dt " << dt  << "\n";
  
  while(((g[0]<0.0)||(g[1]<0.0))||((g[2]<0.0)||(g[3]<0.0))){
    //if (t>=1.0){    std :: cout << "a t " << t << " " << g[0] << " " << g[1] <<  " " << g[2] << " " << g[3] << " " << cVar[0] << " " << cVar[1] << " " << cVar[2] << " " << cVar[3] << "\n";}
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
    //if (x1[0]<100.0) std :: cout << "alarm a \n";
    schoebelZhuHullWhiteUpdateCVar(dt,x1,cVar);
    schoebelZhuHullWhiteUpdateCVar(dt,x1a,cVar+2);
    schoebelZhuHullWhiteUpdateDiscount(dt,x1,discount);
    schoebelZhuHullWhiteUpdateDiscount(dt,x1a,discount+2);
    //if (x1[0]<100.0) std :: cout << "alarm a2 \n";
    schoebelZhuHullWhiteEllStep(dt,imod,x1,y1,x1a,y1a,dW1);
    //if (x1[0]<100.0) std :: cout << "alarm a3 \n";
    //x1[0] += x1[0]*x1[2]*dW1[0];
    // Check if volatility target is reached
    schoebelZhuHullWhiteUpdateEvaluate(x1,discount,cVar,&targetVar,g,&K,&dt);
    schoebelZhuHullWhiteUpdateEvaluate(x1a,discount+2,cVar+2,&targetVar,g+2,&K,&dt);
    // Draw new random variables
    for(i=0;i<3;i++) dW1[i] = sqrtDt*normalDouble();
    //if (x1[0]<100.0) std :: cout << "alarm b \n";
    //dW1[0] = imod.corrStructure[0]*dW1[0]+imod.corrStructure[1]*dW1[1]+imod.corrStructure[2]*dW1[2];
    //dW1[1] = imod.corrStructure[3]*dW1[1]+imod.corrStructure[4]*dW1[2];
    //W1[2] = imod.corrStructure[5]*dW1[2];
    //std :: cout << "The dW components 2 " << dW1[0] << " " << dW1[1]  << " " << dW1[2]  << "\n";
    // Increment the random increments for the coarse process
    for(i=0;i<3;i++) dW2[i]+=dW1[i];
    // Update compounded variance and discount factor
    for(i=0;i<4;i++) cVarp[i] = cVar[i];
    schoebelZhuHullWhiteUpdateCVar(dt,x1,cVar);
    schoebelZhuHullWhiteUpdateCVar(dt,x1a,cVar+2);
    schoebelZhuHullWhiteUpdateDiscount(dt,x1,discount);
    schoebelZhuHullWhiteUpdateDiscount(dt,x1a,discount+2);
    //if (x1[0]<100.0) std :: cout << "alarm b2 \n";
    schoebelZhuHullWhiteEllStep(dt,imod,x1,y1,x1a,y1a,dW1);
    //if (x1[0]<100.0) std :: cout << "alarm b3 \n";
    //x1[0] += x1[0]*x1[2]*dW1[0];
    //std :: cout << "Components " << x1[0] << " " << x1[1]  << " " << x1[2]  << " " <<  x1a[0]  << " " <<  x1a[1]  << " " <<  x1a[2] << "\n";
    //std :: cout << "cumVar " << cVar[0] << " " << cVar[2] << " (" << targetVar << ")\n";
    
    // Re-check if volatility budget is full
    //schoebelZhuHullWhiteUpdateEvaluate(x1,discount,cVar,&targetVar,g,&K);
    //if (t>=1.0){    std :: cout << "k t " << t << " " << g[0] << " " << g[1] <<  " " << g[2] << " " << g[3] << " " << cVar[0] << " " << cVar[1] << " " << cVar[2] << " " << cVar[3] << "\n";}
    //schoebelZhuHullWhiteUpdateEvaluate(x1a,discount+2,cVar+2,&targetVar,g+2,&K);
    //if (t>=1.0){    std :: cout << "l t " << t << " " << g[0] << " " << g[1] <<  " " << g[2] << " " << g[3] << " " << cVar[0] << " " << cVar[1] << " " << cVar[2] << " " << cVar[3] << "\n";}
    // Update the compounded variance and discount for the coarse process
    schoebelZhuHullWhiteUpdateCVar(dt*2,x2,cVar+1);
    schoebelZhuHullWhiteUpdateCVar(dt*2,x2a,cVar+3);
    schoebelZhuHullWhiteUpdateDiscount(dt*2,x2,discount+1);
    schoebelZhuHullWhiteUpdateDiscount(dt*2,x2a,discount+3);
    schoebelZhuHullWhiteEllStep(dt*2,imod,x2,y2,x2a,y2a,dW2);
    //if (x1[0]<100.0) std :: cout << "alarm c \n";
    //x2[0] += x2[0]*x2[2]*dW2[0];
    // Evaluate the payoff, if condition satisfied
    //schoebelZhuHullWhiteUpdateEvaluate(x2,discount+1,cVar+1,&targetVar,g+1,&K);
    //schoebelZhuHullWhiteUpdateEvaluate(x2a,discount+3,cVar+3,&targetVar,g+3,&K);
    //if (t>=1.0){    std :: cout << "v t " << t << " " << g[0] << " " << g[1] <<  " " << g[2] << " " << g[3] << " " << cVar[0] << " " << cVar[1] << " " << cVar[2] << " " << cVar[3] << "\n";}
    schoebelZhuHullWhiteUpdateEvaluate(x1,discount,cVar,&targetVar,g,&K,&dt);
    schoebelZhuHullWhiteUpdateEvaluate(x2,discount+1,cVar+1,&targetVar,g+1,&K,&dt);
    schoebelZhuHullWhiteUpdateEvaluate(x1a,discount+2,cVar+2,&targetVar,g+2,&K,&dt);
    //if (x1[0]<100.0) std :: cout << "alarm d\n";
    schoebelZhuHullWhiteUpdateEvaluate(x2a,discount+3,cVar+3,&targetVar,g+3,&K,&dt);
    tp = t;
    t += 2*dt;
    //if (t>1.0){    std :: cout << "x t " << t << " tp " << tp << " " << g[0] << " " << g[1] <<  " " << g[2] << " " << g[3] << " " << cVar[0] << " " << cVar[1] << " " << cVar[2] << " " << cVar[3] << "\n";}
    //std :: cout << "cumVar2 " << cVar[0] << " " << cVar[2] << " (" << targetVar << ")\n";
    //std :: cout << "gs " << g[0] << " " << g[1] << " " << g[2] << " " << g[3] << "\n";
    //std :: cout << t << " " << x1[0] << " " << x1[1] << " " << x1[2] << " " << " " << cVar[0]<< " " << g[0]  << " " << x2[0] << " " << x2[1] << " " << x2[2]<< " " << g[1] << " " << cVar[1]<< " " << targetVar << "\n";
    //std :: cout << "Components " << x1[0] << " " << x1[1]  << " " << x1[2]  << " " <<  x1a[0]  << " " <<  x1a[1]  << " " <<  x1a[2] << "\n";
    //std :: cout << K << "\n";
  }
  //if (t<1.0){    std :: cout << "y t " << t << " tp " << tp << " dt " <<dt << " " << g[0] << " " << g[1] <<  " " << g[2] << " " << g[3] << " " << cVar[0] << " " << cVar[1] << " " << cVar[2] << " " << cVar[3] << "\n";}
  //std :: cout << "t " << t << "\n"; 
  retVal = 0.5*(g[0]+g[2]-g[1]-g[3]);
  //retVal = g[0]-g[1];
  //retVal = g[2]-g[3];
  //std :: cout << "y t " << t << " tp " << tp << " dt " <<dt << " " << g[0] << " " << g[1] <<  " " << g[2] << " " << g[3] << " " << cVar[0] << " " << cVar[1] << " " << cVar[2] << " " << cVar[3] << "\n";
  //std :: cout  << g[0] << " " << K << " disc " << discount[0]  << "\n";
  //retVal = g[0];
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
  //std :: cout << "sigma " << sigma << "\n";
  //std :: cout << "T " << T << "\n";
  //std :: cout << "targetVar " << T*sigma*sigma << "\n";
  return schoebelZhuHullWhiteEll(dt,testModel,T*sigma*sigma,K);
}


double szhwModelDt(double dt, double S,double sigma, double r, double T, double K,double kappa,double lambda,double gamma,double p,double theta, double sbar, double eta){
  szhw testModel;
  testModel.kap = kappa;
  testModel.lam = lambda;
  testModel.gam = gamma;
  testModel.p = p;
  testModel.theta = theta;
  testModel.sbar = sbar;
  testModel.eta = eta;
  testModel.corrStructure[0] = 1.0;
  testModel.corrStructure[1] = 0.0;
  testModel.corrStructure[2] = 0.0;
  testModel.corrStructure[3] = 1.0;
  testModel.corrStructure[4] = 0.0;
  testModel.corrStructure[5] = 1.0;
  testModel.x0[0] = S;
  testModel.x0[1] = r;
  testModel.x0[2] = sigma;
  //std :: cout << "sigma " << sigma << "\n";                                                                                                                                                                                     
  //std :: cout << "T " << T << "\n";                                                                                                                                                                                             
  //std :: cout << "targetVar " << T*sigma*sigma << "\n";                                                                                                                                                                         
  return schoebelZhuHullWhiteEll(dt,testModel,T*sigma*sigma,K);
}

extern "C" {
  double bsDt(double dt, double S,double sigma, double r, double T, double K){
    return bsModelDt(dt,S,sigma,r,T,K);
  };
  double szhwDt(double dt, double S,double sigma, double r, double T, double K,double kappa,double lambda,double gamma,double p,double theta, double sbar, double eta){
    return szhwModelDt(dt,S,sigma,r,T,K,kappa,lambda,gamma,p,theta,sbar,eta);
  };
  double bsNoStep(double S,double sigma, double r, double T, double K){
    return bsModel0(S,sigma,r,T,K);
  };
  double randn(){ return normalDouble();}
  double test(unsigned int N,double dt){return randTest(N,dt);}
}

//int main(){
//  double price = bsPrice(100.0,0.2,0.00,0.25,100.0,0.005);
//  std :: cout << "BS price " << price << "\n";
//  return 0;
//}
